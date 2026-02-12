# bot.py
import os
import time
import re

import telebot
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup

from datetime import datetime, timezone, timedelta
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton


# ======================
# CONFIG
# ======================
API_TOKEN = os.getenv("BOT_TOKEN")
if not API_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Variables (BOT_TOKEN).")

bot = telebot.TeleBot(API_TOKEN)

ALLOWED_TICKERS = {"META", "SNAP", "PINS"}

HISTORY_PERIOD = "6mo"
INTERVAL = "1d"

# News settings
NEWS_LOOKBACK_DAYS = 30
NEWS_LIMIT = 10

# Returns model settings (fixes unrealistic jumps)
RETURNS_WINDOW_DAYS = 60      # use last ~60 trading days of returns
CLAMP_SIGMA_MULT = 2.0        # max daily move = 2*sigma

# Cache
CACHE_TTL_SECONDS = 120
_cache = {}

# HTTP for excerpt
HTTP_TIMEOUT = 8
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


# ======================
# NEWS RULES (event-based)
# ======================
NEWS_RULES = [
    {"keywords": ["beats earnings", "earnings beat", "revenue beat", "raised guidance", "guidance raised",
                 "upgrade", "price target raised"], "score": +3, "tag": "Strong positive"},
    {"keywords": ["partnership", "deal", "contract", "acquisition", "buyback", "share repurchase"],
     "score": +2, "tag": "Positive catalyst"},
    {"keywords": ["launch", "released", "new product", "record revenue", "strong demand"],
     "score": +2, "tag": "Growth signal"},

    {"keywords": ["expects", "plans", "considering", "announced", "update"], "score": 0, "tag": "Neutral"},

    {"keywords": ["missed earnings", "earnings miss", "revenue miss", "cut guidance", "guidance cut",
                 "downgrade", "price target cut"], "score": -3, "tag": "Strong negative"},
    {"keywords": ["lawsuit", "probe", "investigation", "regulators", "fine", "ban", "antitrust"],
     "score": -2, "tag": "Legal/regulatory risk"},
    {"keywords": ["weak demand", "slowdown", "warning", "decline", "fell", "drops"],
     "score": -2, "tag": "Weakness signal"},
    {"keywords": ["layoffs", "cuts jobs", "cost cutting"], "score": -1, "tag": "Cost-cutting (mixed)"},
]


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
def sanitize_md(text: str) -> str:
    if not text:
        return ""
    return (text.replace("*", "")
                .replace("_", "")
                .replace("`", "")
                .replace("[", "(")
                .replace("]", ")"))


def shorten(text: str, n: int = 240) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


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


# ======================
# INLINE MENU
# ======================
def main_menu():
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("📊 Predict META (1D)", callback_data="predict_META_1d"),
        InlineKeyboardButton("📅 Predict META (7D)", callback_data="predict_META_7d"),
        InlineKeyboardButton("📊 Predict SNAP (1D)", callback_data="predict_SNAP_1d"),
        InlineKeyboardButton("📅 Predict SNAP (7D)", callback_data="predict_SNAP_7d"),
        InlineKeyboardButton("📊 Predict PINS (1D)", callback_data="predict_PINS_1d"),
        InlineKeyboardButton("📅 Predict PINS (7D)", callback_data="predict_PINS_7d"),
        InlineKeyboardButton("ℹ️ Status", callback_data="status"),
    )
    return keyboard


# ======================
# PRICES
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
    if len(df) < 40:
        raise ValueError("Not enough data (need at least 40 trading days).")

    cache_set(cache_key, df)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_volatility(series: pd.Series, window: int = 14) -> float:
    ret = series.pct_change().dropna()
    vol = ret.abs().rolling(window).mean()
    return float(vol.iloc[-1] * 100.0)


# ======================
# NEWS + EXCERPT
# ======================
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

    if not news:
        news = []

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
            excerpt = shorten(og.get("content", ""), 260)
            cache_set(cache_key, excerpt)
            return excerpt

        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            excerpt = shorten(desc.get("content", ""), 260)
            cache_set(cache_key, excerpt)
            return excerpt

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
        title = raw_title.lower().strip()
        if not title:
            continue

        link = n.get("link") or n.get("url") or ""
        publisher = n.get("publisher") or ""

        score = 0
        tags = []
        for rule in NEWS_RULES:
            for kw in rule["keywords"]:
                if kw in title:
                    score += rule["score"]
                    tags.append(rule["tag"])
                    break

        excerpt = fetch_news_excerpt(link)

        items.append({
            "title": raw_title,
            "publisher": publisher,
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
# RETURNS-BASED PREDICTOR (FIXED)
# ======================
def predict_prices(ticker: str, days: int):
    """
    Predicts future prices using recent daily returns (NOT direct price line).
    This avoids unrealistic jumps like 739 -> 617 in one day.
    """
    df = fetch_price_df(ticker)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()

    returns = close.pct_change().dropna()
    if len(returns) < 20:
        raise ValueError("Not enough returns data.")

    window = min(RETURNS_WINDOW_DAYS, len(returns))
    r = returns.iloc[-window:]

    mu = float(r.mean())       # mean daily return
    sigma = float(r.std())     # daily volatility (std)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.01

    last_price = float(close.iloc[-1])

    # Clamp daily step to +/- (CLAMP_SIGMA_MULT*sigma)
    max_step = CLAMP_SIGMA_MULT * sigma

    preds = []
    cur = last_price
    for _ in range(days):
        step = mu
        step = max(-max_step, min(max_step, step))  # clamp
        cur = cur * (1.0 + step)
        preds.append(float(cur))

    # Dates
    last_date = close.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days).to_list()

    # Indicators for explanation
    rsi = compute_rsi(close, 14)
    vol_pct = compute_volatility(close, 14)

    # "Confidence" as stability score: lower sigma -> higher confidence
    # map roughly to 0..1
    stability = float(max(0.0, min(1.0, 1.0 - (sigma * 10.0))))

    # Use "drift" for trend explanation
    drift = mu

    return preds, last_price, stability, drift, future_dates, rsi, vol_pct


# ======================
# LOGIC ENGINE
# ======================
def score_trend(drift: float) -> tuple[int, str]:
    # drift = mean daily return
    if drift > 0:
        return 2, f"Average daily return is positive ({drift*100:.2f}%) → upward bias."
    if drift < 0:
        return -2, f"Average daily return is negative ({drift*100:.2f}%) → downward bias."
    return 0, "Average daily return is near zero → no clear trend."


def score_rsi(rsi: float) -> tuple[int, str]:
    if rsi >= 70:
        return -1, f"RSI is high ({rsi:.1f}) → overbought (pullback risk)."
    if rsi <= 30:
        return +1, f"RSI is low ({rsi:.1f}) → oversold (bounce potential)."
    return 0, f"RSI is normal ({rsi:.1f}) → neutral."


def score_vol(vol_pct: float) -> tuple[int, str]:
    if vol_pct >= 3.0:
        return 0, f"Volatility is high (~{vol_pct:.2f}% daily) → higher uncertainty."
    return 0, f"Volatility is moderate (~{vol_pct:.2f}% daily)."


def score_news(avg_score: float, used_count: int) -> tuple[int, str]:
    if used_count == 0:
        return 0, "No usable recent news → neutral."
    if avg_score >= 1.0:
        return +2, f"News looks bullish (avg score {avg_score:+.2f})."
    if avg_score <= -1.0:
        return -2, f"News looks bearish (avg score {avg_score:+.2f})."
    return 0, f"News is mixed/neutral (avg score {avg_score:+.2f})."


def build_news_excerpts(top_items: list[dict]) -> str:
    if not top_items:
        return ""  # IMPORTANT: don't show "none"

    lines = ["*Key news excerpts:*"]
    for it in top_items:
        title = sanitize_md(it["title"])
        excerpt = sanitize_md(it.get("excerpt") or "")
        link = it.get("link", "")
        tags = ", ".join(it["tags"]) if it["tags"] else "Unclassified"

        if it["score"] >= 2:
            impact = "Possible positive catalyst → can support growth."
        elif it["score"] <= -2:
            impact = "Possible negative catalyst → can pressure price down."
        else:
            impact = "Likely weak/neutral impact."

        lines.append(f"• {title}")
        lines.append(f"  Tags: {sanitize_md(tags)}")
        if excerpt:
            lines.append(f"  Excerpt: _{shorten(excerpt, 220)}_")
        lines.append(f"  {impact}")
        if link:
            lines.append(f"  {link}")

    return "\n".join(lines)


def logical_conclusion(total_score: int) -> str:
    if total_score >= 3:
        return "✅ Conclusion: *Bullish bias* (more factors support growth)."
    if total_score <= -3:
        return "✅ Conclusion: *Bearish bias* (more factors support decline)."
    return "✅ Conclusion: *Mixed / uncertain* (signals conflict or are weak)."


def build_explanation(drift: float, rsi: float, vol_pct: float,
                      avg_news: float, used_news: int, news_excerpts: str) -> str:
    t_score, t_txt = score_trend(drift)
    r_score, r_txt = score_rsi(rsi)
    v_score, v_txt = score_vol(vol_pct)
    n_score, n_txt = score_news(avg_news, used_news)

    total = t_score + r_score + v_score + n_score

    lines = [
        "🧠 *Logical reasoning (factors):*",
        f"• Trend factor: {t_txt} (score {t_score:+d})",
        f"• RSI factor: {r_txt} (score {r_score:+d})",
        f"• Volatility factor: {v_txt} (score {v_score:+d})",
        f"• News factor: {n_txt} (score {n_score:+d})",
        "",
        logical_conclusion(total),
    ]

    if news_excerpts.strip():
        lines.extend(["", news_excerpts])

    return "\n".join(lines)


# ======================
# OUTPUT
# ======================
def build_predict_text(ticker: str, mode: str, preds: list[float], last_price: float, stability: float,
                      future_dates: list, explanation: str) -> str:
    kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

    if mode == "1d":
        predicted = preds[0]
        direction = "📈 UP" if predicted > last_price else "📉 DOWN"
        delta = predicted - last_price
        return (
            f"*{ticker} Predict (1D)*\n\n"
            f"Last close: `{last_price:.2f}`\n"
            f"Predicted next close: `{predicted:.2f}`\n"
            f"Move (approx): `{delta:+.2f}`\n"
            f"Direction: {direction}\n"
            f"Stability score: `{stability*100:.1f}%`\n\n"
            f"{explanation}\n\n"
            f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
        )

    dayN = preds[-1]
    direction = "📈 UP" if dayN > last_price else "📉 DOWN"
    deltaN = dayN - last_price

    rows = [f"• {d.strftime('%Y-%m-%d')}: `{p:.2f}`" for d, p in zip(future_dates, preds)]
    path = "\n".join(rows)

    return (
        f"*{ticker} Predict (7D)*\n\n"
        f"Last close: `{last_price:.2f}`\n"
        f"Day 7 prediction: `{dayN:.2f}`\n"
        f"Move (approx): `{deltaN:+.2f}`\n"
        f"Direction (vs last): {direction}\n"
        f"Stability score: `{stability*100:.1f}%`\n\n"
        f"*Next predicted closes:*\n{path}\n\n"
        f"{explanation}\n\n"
        f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
    )


def usage_text() -> str:
    return (
        "✅ Use:\n"
        "• `/predict META 1d`\n"
        "• `/predict META 7d`\n\n"
        f"Allowed: {', '.join(sorted(ALLOWED_TICKERS))}"
    )


# ======================
# CORE
# ======================
def do_predict(ticker: str, mode: str) -> str:
    days = 1 if mode == "1d" else 7
    preds, last_price, stability, drift, future_dates, rsi, vol_pct = predict_prices(ticker, days=days)

    news = fetch_news_yahoo(ticker)
    avg_score, top_items, used_count = analyze_news(news, lookback_days=NEWS_LOOKBACK_DAYS, limit=NEWS_LIMIT)
    news_excerpts = build_news_excerpts(top_items)

    explanation = build_explanation(
        drift=drift,
        rsi=rsi,
        vol_pct=vol_pct,
        avg_news=avg_score,
        used_news=used_count,
        news_excerpts=news_excerpts
    )

    return build_predict_text(ticker, mode, preds, last_price, stability, future_dates, explanation)


# ======================
# COMMANDS
# ======================
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "**⚠️ DISCLAIMER ⚠️**\n\n"
        "**THIS BOT IS CREATED FOR EXPERIMENTAL AND EDUCATIONAL PURPOSES ONLY.**\n"
        "**IT DOES NOT PROVIDE FINANCIAL ADVICE.**\n"
        "**DO NOT MAKE INVESTMENT DECISIONS BASED SOLELY ON THIS BOT.**\n\n"
        "Market predictions are uncertain and may be inaccurate.\n"
        "Always do your own research and consult professional advisors.\n\n"
        "----------------------------------\n\n"
        "👋 *Welcome to Predict AI*\n\n"
        "This bot uses:\n"
        "• Historical prices (Yahoo Finance)\n"
        "• Returns-based forecasting (more realistic step-by-step path)\n"
        "• Technical indicators (RSI, volatility)\n"
        "• Yahoo Finance news + short excerpts\n\n"
        "*Commands:*\n"
        "• `/predict META 1d`\n"
        "• `/predict META 7d`\n\n"
        "Type `/status` to check bot status.",
        reply_markup=main_menu(),
        parse_mode="Markdown"
    )


@bot.message_handler(commands=["status"])
def status(message):
    bot.reply_to(
        message,
        f"✅ Bot RUNNING\nSource: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
        "Command: /predict <TICKER> <1d|7d>\n"
        f"News lookback: {NEWS_LOOKBACK_DAYS} days"
    )


@bot.message_handler(commands=["predict"])
def predict_command(message):
    ticker, mode = parse_predict_command(message.text)

    if ticker is None and mode is None:
        bot.send_message(message.chat.id, usage_text(), parse_mode="Markdown")
        return

    if ticker not in ALLOWED_TICKERS:
        bot.send_message(message.chat.id, f"Allowed: {', '.join(sorted(ALLOWED_TICKERS))}")
        return

    if mode not in {"1d", "7d"}:
        bot.send_message(message.chat.id, "Mode must be `1d` or `7d`.\n\n" + usage_text(), parse_mode="Markdown")
        return

    try:
        text = do_predict(ticker, mode)
        bot.send_message(message.chat.id, text, parse_mode="Markdown")
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")


# ======================
# BUTTON HANDLER
# ======================
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        if call.data == "status":
            bot.send_message(
                call.message.chat.id,
                f"✅ Bot RUNNING\nSource: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
                "Command: /predict <TICKER> <1d|7d>\n"
                f"News lookback: {NEWS_LOOKBACK_DAYS} days"
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
            bot.send_message(call.message.chat.id, text, parse_mode="Markdown")
            return

    except Exception as e:
        bot.send_message(call.message.chat.id, f"Error: {e}")


# ======================
# RUN
# ======================
if __name__ == "__main__":
    print("Bot started ✅ (returns-based predictor + logic + news excerpts)")
    bot.infinity_polling()
