import os
import time
import re
from typing import Dict, List, Tuple

import telebot
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from datetime import datetime, timezone, timedelta
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from openai import OpenAI


# ======================
# CONFIG
# ======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Railway Variables.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional, but recommended
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

bot = telebot.TeleBot(BOT_TOKEN)

ALLOWED_TICKERS = {"META", "SNAP", "PINS"}

HISTORY_PERIOD = "6mo"
INTERVAL = "1d"

NEWS_LOOKBACK_DAYS = 30
NEWS_LIMIT = 10

# ML model settings (Linear Regression on lagged returns)
LAGS = 5               # number of lag days used as features
TRAIN_WINDOW = 120     # last N trading days for training (<= data length)

# MAE Backtest settings (simple walk-forward)
MAE_TEST_POINTS = 20   # number of recent origins to evaluate MAE on (keep small for speed)

# Cache
CACHE_TTL_SECONDS = 120
_cache = {}

# HTTP for news excerpt
HTTP_TIMEOUT = 8
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Chat history (optional, small memory) — no "mode", always on
CHAT_HISTORY: Dict[int, List[Dict[str, str]]] = {}
CHAT_HISTORY_MAX_TURNS = 10


# ======================
# INLINE MENU
# ======================
def main_menu():
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("📊 Predict META (1D)", callback_data="predict_META_1d"),
        InlineKeyboardButton("📅 Predict META (7D)", callback_data="predict_META_7d"),
        InlineKeyboardButton("📊 Predict SNAP (1D)", callback_data="predict_SNAP_1d"),
        InlineKeyboardButton("📅 Predict SNAP (7D)", callback_data="predict_SNAP_7d"),
        InlineKeyboardButton("📊 Predict PINS (1D)", callback_data="predict_PINS_1d"),
        InlineKeyboardButton("📅 Predict PINS (7D)", callback_data="predict_PINS_7d"),
        InlineKeyboardButton("ℹ️ Status", callback_data="status"),
    )
    return kb


# ======================
# CACHE HELPERS
# ======================
def cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    ts, val = item
    if time.time() - ts > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return val


def cache_set(key: str, val):
    _cache[key] = (time.time(), val)


# ======================
# TEXT HELPERS
# ======================
def shorten(text: str, n: int = 240) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def escape_html(s: str) -> str:
    if not s:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def parse_predict_command(text: str):
    """
    /predict META 1d
    /predict META 7d
    """
    parts = text.split()
    if len(parts) != 3:
        return None, None
    ticker = parts[1].upper().strip()
    mode = parts[2].lower().strip()
    if mode not in {"1d", "7d"}:
        return ticker, None
    return ticker, mode


def openai_available() -> bool:
    return client is not None and bool(OPENAI_API_KEY)


# ======================
# DATA FROM YAHOO
# ======================
def fetch_price_df(ticker: str) -> pd.DataFrame:
    cache_key = f"prices:{ticker}:{HISTORY_PERIOD}:{INTERVAL}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    df = yf.download(ticker, period=HISTORY_PERIOD, interval=INTERVAL, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned empty data (try later).")

    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance data has no 'Close' column.")

    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 60:
        raise ValueError("Not enough data (need at least ~60 trading days).")

    cache_set(cache_key, df)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_volatility_proxy(series: pd.Series, window: int = 14) -> float:
    ret = series.pct_change().dropna()
    vol = ret.abs().rolling(window).mean()
    return float(vol.iloc[-1] * 100.0)


# ======================
# NEWS (Yahoo via yfinance) + excerpt scraping
# ======================
NEWS_RULES = [
    {"keywords": ["beats earnings", "earnings beat", "revenue beat", "raised guidance", "guidance raised",
                  "upgrade", "price target raised"], "score": +3, "tag": "Strong positive"},
    {"keywords": ["partnership", "deal", "contract", "acquisition", "buyback", "share repurchase"],
     "score": +2, "tag": "Positive catalyst"},
    {"keywords": ["launch", "released", "new product", "record revenue", "strong demand"],
     "score": +2, "tag": "Growth signal"},
    {"keywords": ["missed earnings", "earnings miss", "revenue miss", "cut guidance", "guidance cut",
                  "downgrade", "price target cut"], "score": -3, "tag": "Strong negative"},
    {"keywords": ["lawsuit", "probe", "investigation", "regulators", "fine", "ban", "antitrust"],
     "score": -2, "tag": "Legal/regulatory risk"},
    {"keywords": ["weak demand", "slowdown", "warning", "decline", "fell", "drops"],
     "score": -2, "tag": "Weakness signal"},
]


def fetch_news_yahoo(ticker: str) -> list[dict]:
    cache_key = f"newslist:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    t = yf.Ticker(ticker)
    try:
        news = t.news
    except Exception:
        try:
            news = t.get_news()
        except Exception:
            news = []

    news = news or []
    cache_set(cache_key, news)
    return news


def fetch_news_excerpt(url: str) -> str:
    if not url:
        return ""
    cache_key = f"excerpt:{url}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT, allow_redirects=True)
        if r.status_code >= 400:
            cache_set(cache_key, "")
            return ""
        soup = BeautifulSoup(r.text, "html.parser")

        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            ex = shorten(og.get("content", ""), 260)
            cache_set(cache_key, ex)
            return ex

        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            ex = shorten(desc.get("content", ""), 260)
            cache_set(cache_key, ex)
            return ex

    except Exception:
        pass

    cache_set(cache_key, "")
    return ""


def analyze_news(news: list[dict], lookback_days: int = 30, limit: int = 10):
    cutoff = int(time.time()) - lookback_days * 24 * 3600
    items = []

    for n in news:
        ts = n.get("providerPublishTime")
        if isinstance(ts, int) and ts < cutoff:
            continue

        raw_title = n.get("title") or ""
        title_l = raw_title.lower().strip()
        if not title_l:
            continue

        link = n.get("link") or n.get("url") or ""
        score = 0
        tags = []

        for rule in NEWS_RULES:
            for kw in rule["keywords"]:
                if kw in title_l:
                    score += rule["score"]
                    tags.append(rule["tag"])
                    break

        excerpt = fetch_news_excerpt(link)

        items.append({
            "title": raw_title,
            "link": link,
            "ts": ts if isinstance(ts, int) else 0,
            "score": int(score),
            "tags": list(dict.fromkeys(tags)),
            "excerpt": excerpt
        })

    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    items = items[:limit]

    if not items:
        return 0.0, [], 0

    avg_score = float(np.mean([x["score"] for x in items]))
    top_items = sorted(items, key=lambda x: abs(x["score"]), reverse=True)[:3]
    return avg_score, top_items, len(items)


# ======================
# ML PREDICTOR (AI MODEL): Linear Regression on lagged returns
# ======================
def train_lr_on_returns(returns: pd.Series, lags: int = 5) -> Tuple[LinearRegression, float]:
    """
    Build supervised dataset:
    X_t = [r_{t-1}, r_{t-2}, ... r_{t-lags}]
    y_t = r_t
    """
    r = returns.values
    X, y = [], []
    for i in range(lags, len(r)):
        X.append(r[i-lags:i][::-1])  # most recent first
        y.append(r[i])
    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y)) if len(y) > 5 else 0.0
    return model, r2


def backtest_mae_price(
    close: pd.Series,
    horizon_days: int,
    lags: int = 5,
    train_window: int = 120,
    test_points: int = 20,
) -> Tuple[float, float]:
    """
    Simple walk-forward MAE in PRICE units and percent.
    - For each origin i, we train on returns up to i (no lookahead),
      predict price at i + horizon_days using iterative returns,
      compare with actual close at i + horizon_days.
    Returns: (mae_price, mae_percent)
    """
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < (lags + 40 + horizon_days):
        return float("nan"), float("nan")

    # returns aligned to close index (return at t = close[t]/close[t-1] - 1)
    returns = close.pct_change().dropna()
    r = returns.values
    c = close.values  # close[0..N-1]

    N = len(close) - 1  # last index in close array

    # valid origins i: need i >= lags and i + horizon_days <= N
    last_origin = N - horizon_days
    first_origin = max(lags, last_origin - test_points + 1)
    if first_origin > last_origin:
        return float("nan"), float("nan")

    preds = []
    actuals = []

    # helper: predict horizon returns starting from state at origin i
    for i in range(first_origin, last_origin + 1):
        # known returns go from index 1..i (because return[1] uses close[0]->close[1])
        # in returns array r, its indices are 0..len(returns)-1 corresponding to close[1..]
        # close index i corresponds to returns index i-1
        end_r_idx = i - 1
        if end_r_idx < lags:
            continue

        # training returns slice ends at end_r_idx inclusive
        train_end = end_r_idx
        train_start = max(0, train_end - train_window + 1)
        train_returns = pd.Series(r[train_start:train_end + 1])

        if len(train_returns) < (lags + 10):
            continue

        model, _ = train_lr_on_returns(train_returns, lags=lags)

        # state = last lags returns ending at i (most recent first)
        state = list(r[train_end - lags + 1: train_end + 1])[::-1]

        # volatility clamp from training returns
        sigma = float(np.std(train_returns.values))
        clamp = 2.5 * sigma if sigma > 0 else 0.02

        cur_price = float(c[i])

        # iterative multi-step prediction
        for _ in range(horizon_days):
            x = np.array(state).reshape(1, -1)
            pred_r = float(model.predict(x)[0])
            pred_r = max(-clamp, min(clamp, pred_r))
            cur_price = cur_price * (1.0 + pred_r)
            state = [pred_r] + state[:-1]

        pred_price = float(cur_price)
        actual_price = float(c[i + horizon_days])

        preds.append(pred_price)
        actuals.append(actual_price)

    if len(actuals) < 5:
        return float("nan"), float("nan")

    mae_price = float(mean_absolute_error(actuals, preds))
    mae_percent = float(np.mean(np.abs((np.array(preds) - np.array(actuals)) / np.array(actuals))) * 100.0)
    return mae_price, mae_percent


def predict_prices_ml_lr(
    ticker: str, days: int
) -> Tuple[List[float], float, float, float, float, List[pd.Timestamp], float, float, float, float]:
    """
    Returns:
    preds_prices, last_price, r2, mu, sigma, future_dates, rsi, vol_proxy, mae_price, mae_percent
    """
    df = fetch_price_df(ticker)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()

    # Use last TRAIN_WINDOW returns for training
    returns = close.pct_change().dropna()
    if len(returns) < (LAGS + 30):
        raise ValueError(f"Not enough returns to train ML model (need > {LAGS+30} days).")

    returns = returns.iloc[-min(TRAIN_WINDOW, len(returns)):]
    model, r2 = train_lr_on_returns(returns, lags=LAGS)

    mu = float(returns.mean())
    sigma = float(returns.std()) if float(returns.std()) > 0 else 0.01

    last_price = float(close.iloc[-1])

    # Start state: last LAGS returns
    state = list(returns.values[-LAGS:])[::-1]  # most recent first

    preds_prices = []
    cur = last_price

    # Simple safety clamp to avoid insane steps (still ML, just risk control)
    clamp = 2.5 * sigma

    for _ in range(days):
        x = np.array(state).reshape(1, -1)
        pred_r = float(model.predict(x)[0])
        pred_r = max(-clamp, min(clamp, pred_r))

        cur = cur * (1.0 + pred_r)
        preds_prices.append(float(cur))

        # update state
        state = [pred_r] + state[:-1]

    last_date = close.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days).to_list()

    rsi = compute_rsi(close, 14)
    vol_proxy = compute_volatility_proxy(close, 14)

    # ✅ MAE backtest for this horizon (1d or 7d)
    mae_price, mae_percent = backtest_mae_price(
        close=close,
        horizon_days=days,
        lags=LAGS,
        train_window=TRAIN_WINDOW,
        test_points=MAE_TEST_POINTS,
    )

    return preds_prices, last_price, r2, mu, sigma, future_dates, rsi, vol_proxy, mae_price, mae_percent


# ======================
# OPENAI: explain forecast + general assistant replies
# ======================
def llm_explain_prediction(
    ticker: str,
    mode: str,
    last_price: float,
    preds: List[float],
    r2: float,
    mu: float,
    sigma: float,
    rsi: float,
    vol_proxy: float,
    mae_price: float,
    mae_percent: float,
    news_top: List[dict],
) -> str:
    if not openai_available():
        return ""

    dayN = preds[-1]
    direction = "UP" if dayN > last_price else "DOWN"
    delta = dayN - last_price

    news_lines = []
    for it in news_top[:3]:
        title = it.get("title", "")
        excerpt = it.get("excerpt", "")
        score = it.get("score", 0)
        tags = ", ".join(it.get("tags", [])) if it.get("tags") else "Unclassified"
        news_lines.append(
            f"- title: {title}\n  tags: {tags}\n  score: {score}\n  excerpt: {shorten(excerpt, 220)}"
        )
    news_block = "\n".join(news_lines) if news_lines else "NO RECENT NEWS ITEMS AVAILABLE."

    mae_line = "MAE backtest: unavailable" if (np.isnan(mae_price) or np.isnan(mae_percent)) else \
        f"MAE backtest (price): {mae_price:.2f} | MAE%: {mae_percent:.2f}%"

    prompt = f"""
You are helping explain a stock forecast for an educational Telegram bot.
IMPORTANT:
- Use ONLY the data provided below.
- Do NOT invent news or numbers.
- Be honest about limitations.

Write in English. Keep it concise.

DATA:
ticker: {ticker}
horizon: {mode}
last_close: {last_price:.2f}
prediction_end: {dayN:.2f}
delta: {delta:+.2f}
direction: {direction}

AI model used for forecast:
- Model: Linear Regression (trained on lagged daily returns, autoregressive features)
- R2 on training: {r2*100:.1f}%
- mean daily return (mu): {mu*100:.3f}%
- daily volatility (sigma): {sigma*100:.3f}%
- {mae_line}

Indicators:
- RSI(14): {rsi:.1f}
- avg_abs_return_14d: {vol_proxy:.2f}%

News (titles + excerpts):
{news_block}

Output format:
1) Summary (2 lines)
2) Why it may move (4-7 bullets)
3) Risks/limits (3-5 bullets)
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be precise. No hallucinations. Educational tone."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def llm_assistant_reply(user_id: int, user_text: str) -> str:
    if not openai_available():
        return "OpenAI is not configured. Add OPENAI_API_KEY in Railway Variables."

    hist = CHAT_HISTORY.get(user_id, [])
    hist.append({"role": "user", "content": user_text})

    if len(hist) > CHAT_HISTORY_MAX_TURNS * 2:
        hist = hist[-CHAT_HISTORY_MAX_TURNS * 2 :]

    messages = [{"role": "system", "content": "You are a helpful assistant inside a Telegram bot. Keep answers clear and not too long."}]
    messages += hist

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
        )
        ans = resp.choices[0].message.content.strip()
        hist.append({"role": "assistant", "content": ans})
        CHAT_HISTORY[user_id] = hist
        return ans
    except Exception as e:
        return f"OpenAI error: {e}"


# ======================
# OUTPUT TEXT
# ======================
def build_predict_text_html(
    ticker: str,
    mode: str,
    preds: List[float],
    last_price: float,
    r2: float,
    mae_price: float,
    mae_percent: float,
    future_dates: List[pd.Timestamp],
    rule_reasoning: str,
    ai_reasoning: str,
) -> str:
    kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

    mae_txt = "N/A" if (np.isnan(mae_price) or np.isnan(mae_percent)) else f"{mae_price:.2f} ({mae_percent:.2f}%)"

    if mode == "1d":
        predicted = preds[0]
        delta = predicted - last_price
        direction = "UP" if predicted > last_price else "DOWN"

        header = f"<b>{ticker} Predict (1D)</b>\n\n"
        stats = (
            f"Last close: <code>{last_price:.2f}</code>\n"
            f"Predicted next close: <code>{predicted:.2f}</code>\n"
            f"Move (approx): <code>{delta:+.2f}</code>\n"
            f"Direction: <b>{direction}</b>\n"
            f"Model (Linear Regression) R²: <code>{r2*100:.1f}%</code>\n"
            f"Backtest MAE (1D): <code>{mae_txt}</code>\n\n"
        )
    else:
        dayN = preds[-1]
        deltaN = dayN - last_price
        direction = "UP" if dayN > last_price else "DOWN"

        header = f"<b>{ticker} Predict (7D)</b>\n\n"
        stats = (
            f"Last close: <code>{last_price:.2f}</code>\n"
            f"Day 7 prediction: <code>{dayN:.2f}</code>\n"
            f"Move (approx): <code>{deltaN:+.2f}</code>\n"
            f"Direction: <b>{direction}</b>\n"
            f"Model (Linear Regression) R²: <code>{r2*100:.1f}%</code>\n"
            f"Backtest MAE (7D): <code>{mae_txt}</code>\n\n"
        )

        path_lines = []
        for d, p in zip(future_dates, preds):
            path_lines.append(f"• {d.strftime('%Y-%m-%d')}: <code>{p:.2f}</code>")
        path = "<b>Next predicted closes:</b>\n" + "\n".join(path_lines) + "\n\n"

        stats += path

    text = header + stats
    text += "<b>Logic (rules + indicators):</b>\n" + escape_html(rule_reasoning) + "\n"

    if ai_reasoning.strip():
        text += "\n<b>AI explanation (OpenAI):</b>\n" + escape_html(ai_reasoning) + "\n"

    text += f"\n⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
    return text


def rule_based_reasoning(mu: float, sigma: float, rsi: float, vol_proxy: float, news_avg: float, news_count: int) -> str:
    trend = "upward" if mu > 0 else "downward" if mu < 0 else "flat"
    trend_score = 1 if mu > 0 else -1 if mu < 0 else 0

    rsi_score = -1 if rsi >= 70 else 1 if rsi <= 30 else 0
    rsi_txt = "overbought (pullback risk)" if rsi >= 70 else "oversold (bounce potential)" if rsi <= 30 else "neutral"

    news_score = 0
    if news_count == 0:
        news_txt = "no usable recent news (neutral)"
    else:
        if news_avg >= 1.0:
            news_score = 2
            news_txt = f"bullish average score ({news_avg:+.2f})"
        elif news_avg <= -1.0:
            news_score = -2
            news_txt = f"bearish average score ({news_avg:+.2f})"
        else:
            news_txt = f"mixed/neutral average score ({news_avg:+.2f})"

    total = trend_score + rsi_score + news_score

    if total >= 2:
        concl = "Bullish bias"
    elif total <= -2:
        concl = "Bearish bias"
    else:
        concl = "Mixed / uncertain"

    return (
        f"- Trend (mu): {mu*100:.3f}% daily → {trend}\n"
        f"- Volatility (sigma): {sigma*100:.3f}% daily\n"
        f"- RSI(14): {rsi:.1f} → {rsi_txt}\n"
        f"- Volatility proxy (avg abs return 14d): {vol_proxy:.2f}%\n"
        f"- News: {news_txt}\n"
        f"Conclusion: {concl}"
    )


# ======================
# CORE PREDICT
# ======================
def do_predict(ticker: str, mode: str) -> str:
    days = 1 if mode == "1d" else 7

    preds, last_price, r2, mu, sigma, future_dates, rsi, vol_proxy, mae_price, mae_percent = predict_prices_ml_lr(ticker, days=days)

    news = fetch_news_yahoo(ticker)
    news_avg, top_items, used_count = analyze_news(news, lookback_days=NEWS_LOOKBACK_DAYS, limit=NEWS_LIMIT)

    rule_reason = rule_based_reasoning(mu, sigma, rsi, vol_proxy, news_avg, used_count)

    ai_reason = llm_explain_prediction(
        ticker=ticker,
        mode=mode,
        last_price=last_price,
        preds=preds,
        r2=r2,
        mu=mu,
        sigma=sigma,
        rsi=rsi,
        vol_proxy=vol_proxy,
        mae_price=mae_price,
        mae_percent=mae_percent,
        news_top=top_items,
    )

    return build_predict_text_html(
        ticker=ticker,
        mode=mode,
        preds=preds,
        last_price=last_price,
        r2=r2,
        mae_price=mae_price,
        mae_percent=mae_percent,
        future_dates=future_dates,
        rule_reasoning=rule_reason,
        ai_reasoning=ai_reason,
    )


# ======================
# COMMANDS
# ======================
@bot.message_handler(commands=["start"])
def start(message):
    text = (
        "⚠️ <b>DISCLAIMER</b> ⚠️\n\n"
        "<b>THIS BOT IS CREATED FOR EXPERIMENTAL AND EDUCATIONAL PURPOSES ONLY.</b>\n"
        "<b>IT DOES NOT PROVIDE FINANCIAL ADVICE.</b>\n"
        "<b>DO NOT MAKE INVESTMENT DECISIONS BASED SOLELY ON THIS BOT.</b>\n\n"
        "Commands:\n"
        "• <code>/predict META 1d</code>\n"
        "• <code>/predict META 7d</code>\n\n"
        "You can also ask questions in normal text (the bot will answer with OpenAI if configured).\n"
    )
    bot.send_message(message.chat.id, text, reply_markup=main_menu(), parse_mode="HTML")


@bot.message_handler(commands=["status"])
def status(message):
    ai_status = "✅ OpenAI enabled" if openai_available() else "❌ OpenAI not configured (set OPENAI_API_KEY)"
    txt = (
        f"✅ Bot RUNNING\n"
        f"Source: Yahoo Finance\n"
        f"Tickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
        f"Predict command: /predict <TICKER> <1d|7d>\n"
        f"Model: Linear Regression (lagged returns)\n"
        f"Backtest MAE: walk-forward (last {MAE_TEST_POINTS} points)\n"
        f"{ai_status}"
    )
    bot.send_message(message.chat.id, txt)


@bot.message_handler(commands=["predict"])
def predict_command(message):
    ticker, mode = parse_predict_command(message.text)

    if ticker is None and mode is None:
        bot.send_message(
            message.chat.id,
            "Use: /predict META 1d  OR  /predict META 7d\nAllowed: " + ", ".join(sorted(ALLOWED_TICKERS))
        )
        return

    if ticker not in ALLOWED_TICKERS:
        bot.send_message(message.chat.id, f"Allowed: {', '.join(sorted(ALLOWED_TICKERS))}")
        return

    if mode not in {"1d", "7d"}:
        bot.send_message(message.chat.id, "Mode must be 1d or 7d. Example: /predict META 7d")
        return

    try:
        text = do_predict(ticker, mode)
        bot.send_message(message.chat.id, text, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")


@bot.message_handler(func=lambda m: True)
def any_text(message):
    if message.text and message.text.strip().startswith("/predict"):
        bot.send_message(message.chat.id, "Format: /predict META 1d  OR  /predict META 7d")
        return

    reply = llm_assistant_reply(message.from_user.id, message.text or "")
    bot.send_message(message.chat.id, reply)


# ======================
# BUTTON HANDLER
# ======================
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        if call.data == "status":
            ai_status = "✅ OpenAI enabled" if openai_available() else "❌ OpenAI not configured (set OPENAI_API_KEY)"
            bot.send_message(
                call.message.chat.id,
                f"✅ Bot RUNNING\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
                f"Model: Linear Regression (lagged returns)\n"
                f"Backtest MAE: walk-forward (last {MAE_TEST_POINTS} points)\n"
                f"{ai_status}"
            )
            return

        if call.data.startswith("predict_"):
            _, ticker, mode = call.data.split("_", 2)
            ticker = ticker.upper()
            mode = mode.lower()

            if ticker not in ALLOWED_TICKERS or mode not in {"1d", "7d"}:
                bot.send_message(call.message.chat.id, "Invalid ticker/mode.")
                return

            text = do_predict(ticker, mode)
            bot.send_message(call.message.chat.id, text, parse_mode="HTML", disable_web_page_preview=True)
            return

    except Exception as e:
        bot.send_message(call.message.chat.id, f"Error: {e}")


# ======================
# RUN
# ======================
if __name__ == "__main__":
    print("Bot started ✅ (ML Linear Regression predictor + OpenAI assistant + MAE backtest)")
    bot.infinity_polling()
