# bot.py
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

from datetime import datetime, timezone, timedelta
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from openai import OpenAI


# ======================
# CONFIG
# ======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Variables (BOT_TOKEN).")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional but recommended
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

bot = telebot.TeleBot(BOT_TOKEN)

ALLOWED_TICKERS = {"META", "SNAP", "PINS"}

HISTORY_PERIOD = "6mo"
INTERVAL = "1d"

NEWS_LOOKBACK_DAYS = 30
NEWS_LIMIT = 10

# Returns model settings (fixes unrealistic jumps)
RETURNS_WINDOW_DAYS = 60
CLAMP_SIGMA_MULT = 2.0

# Cache
CACHE_TTL_SECONDS = 120
_cache = {}

# HTTP for excerpt
HTTP_TIMEOUT = 8
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Chat mode memory (per user)
CHAT_MODE_USERS = set()
CHAT_HISTORY: Dict[int, List[Dict[str, str]]] = {}  # user_id -> [{"role":"user/assistant","content":...}, ...]
CHAT_HISTORY_MAX_TURNS = 12  # keep last messages


# ======================
# NEWS RULES (simple scoring)
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
        InlineKeyboardButton("💬 Chat mode", callback_data="chat_on"),
        InlineKeyboardButton("🛑 Exit chat", callback_data="chat_off"),
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
        title_l = raw_title.lower().strip()
        if not title_l:
            continue

        link = n.get("link") or n.get("url") or ""
        publisher = n.get("publisher") or ""

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
# RETURNS-BASED PREDICTOR
# ======================
def predict_prices(ticker: str, days: int):
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

    max_step = CLAMP_SIGMA_MULT * sigma

    preds = []
    cur = last_price
    for _ in range(days):
        step = max(-max_step, min(max_step, mu))
        cur = cur * (1.0 + step)
        preds.append(float(cur))

    last_date = close.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days).to_list()

    rsi = compute_rsi(close, 14)
    vol_pct = compute_volatility(close, 14)

    stability = float(max(0.0, min(1.0, 1.0 - (sigma * 10.0))))
    drift = mu  # mean return used as "trend"

    return preds, last_price, stability, drift, future_dates, rsi, vol_pct, sigma, mu


# ======================
# OPENAI: explanation + chat
# ======================
def openai_available() -> bool:
    return client is not None and bool(OPENAI_API_KEY)


def llm_explain_prediction(
    ticker: str,
    mode: str,
    last_price: float,
    preds: List[float],
    drift: float,
    sigma: float,
    rsi: float,
    vol_pct: float,
    news_top: List[dict],
) -> str:
    """
    Uses OpenAI to generate a human explanation based ONLY on provided data.
    """
    if not openai_available():
        return ""  # no LLM text

    dayN = preds[-1]
    direction = "UP" if dayN > last_price else "DOWN"
    delta = dayN - last_price

    # Prepare compact news context (no hallucinations)
    news_block_lines = []
    for it in news_top[:3]:
        title = sanitize_md(it.get("title", ""))
        excerpt = sanitize_md(it.get("excerpt", ""))
        score = it.get("score", 0)
        tag = ", ".join(it.get("tags", [])) if it.get("tags") else "Unclassified"
        news_block_lines.append(f"- title: {title}\n  tag: {tag}\n  score: {score}\n  excerpt: {shorten(excerpt, 220)}")

    news_block = "\n".join(news_block_lines) if news_block_lines else "NO RECENT NEWS ITEMS AVAILABLE."

    prompt = f"""
You are a finance assistant for an educational Telegram bot.
Your job: explain the bot's forecast in a logical, honest way, using ONLY the data provided below.
Do NOT invent news or numbers. If news is missing, say it clearly.

Output format:
1) "Summary" (2-3 lines)
2) "Why it may move" (3-6 bullet points)
3) "Risks / limits" (2-4 bullet points)
Keep it concise.

DATA:
ticker: {ticker}
horizon: {mode}
last_close: {last_price:.2f}
dayN_prediction: {dayN:.2f}
delta: {delta:+.2f}
direction: {direction}

returns_model:
- drift (mean daily return): {drift*100:.3f}%
- sigma (daily volatility std): {sigma*100:.3f}%
- clamp: +/- {CLAMP_SIGMA_MULT} * sigma

indicators:
- RSI(14): {rsi:.1f}
- avg_abs_return_14d (volatility proxy): {vol_pct:.2f}%

news_items (top, with excerpts):
{news_block}
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be precise, do not hallucinate. Educational tone."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception:
        return ""


def llm_chat_reply(user_id: int, user_text: str) -> str:
    if not openai_available():
        return "OpenAI API is not configured. Add OPENAI_API_KEY in Railway Variables."

    history = CHAT_HISTORY.get(user_id, [])
    history.append({"role": "user", "content": user_text})

    # Trim history
    if len(history) > CHAT_HISTORY_MAX_TURNS * 2:
        history = history[-CHAT_HISTORY_MAX_TURNS * 2 :]

    messages = [
        {"role": "system", "content": "You are a helpful assistant inside a Telegram bot. Keep answers short and clear."},
        *history
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
        )
        answer = resp.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": answer})
        CHAT_HISTORY[user_id] = history
        return answer
    except Exception as e:
        return f"OpenAI error: {e}"


# ======================
# LOGIC ENGINE (scores)
# ======================
def score_trend(drift: float) -> Tuple[int, str]:
    if drift > 0:
        return 2, f"Average daily return is positive ({drift*100:.2f}%) → upward bias."
    if drift < 0:
        return -2, f"Average daily return is negative ({drift*100:.2f}%) → downward bias."
    return 0, "Average daily return is near zero → no clear trend."


def score_rsi(rsi: float) -> Tuple[int, str]:
    if rsi >= 70:
        return -1, f"RSI is high ({rsi:.1f}) → overbought (pullback risk)."
    if rsi <= 30:
        return +1, f"RSI is low ({rsi:.1f}) → oversold (bounce potential)."
    return 0, f"RSI is normal ({rsi:.1f}) → neutral."


def score_vol(vol_pct: float) -> Tuple[int, str]:
    if vol_pct >= 3.0:
        return 0, f"Volatility is high (~{vol_pct:.2f}% daily) → higher uncertainty."
    return 0, f"Volatility is moderate (~{vol_pct:.2f}% daily)."


def score_news(avg_score: float, used_count: int) -> Tuple[int, str]:
    if used_count == 0:
        return 0, "No usable recent news → neutral."
    if avg_score >= 1.0:
        return +2, f"News looks bullish (avg score {avg_score:+.2f})."
    if avg_score <= -1.0:
        return -2, f"News looks bearish (avg score {avg_score:+.2f})."
    return 0, f"News is mixed/neutral (avg score {avg_score:+.2f})."


def logical_conclusion(total_score: int) -> str:
    if total_score >= 3:
        return "✅ Conclusion: *Bullish bias* (more factors support growth)."
    if total_score <= -3:
        return "✅ Conclusion: *Bearish bias* (more factors support decline)."
    return "✅ Conclusion: *Mixed / uncertain* (signals conflict or are weak)."


def build_rule_based_reasoning(drift: float, rsi: float, vol_pct: float, avg_news: float, used_news: int) -> str:
    t_score, t_txt = score_trend(drift)
    r_score, r_txt = score_rsi(rsi)
    v_score, v_txt = score_vol(vol_pct)
    n_score, n_txt = score_news(avg_news, used_news)
    total = t_score + r_score + v_score + n_score

    return "\n".join([
        "🧠 *Rule-based factors:*",
        f"• Trend: {t_txt} (score {t_score:+d})",
        f"• RSI: {r_txt} (score {r_score:+d})",
        f"• Volatility: {v_txt} (score {v_score:+d})",
        f"• News: {n_txt} (score {n_score:+d})",
        "",
        logical_conclusion(total),
    ])


# ======================
# OUTPUT
# ======================
def build_predict_text(
    ticker: str,
    mode: str,
    preds: List[float],
    last_price: float,
    stability: float,
    future_dates: List,
    rule_reasoning: str,
    llm_reasoning: str,
) -> str:
    kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

    if mode == "1d":
        predicted = preds[0]
        direction = "📈 UP" if predicted > last_price else "📉 DOWN"
        delta = predicted - last_price
        text = (
            f"*{ticker} Predict (1D)*\n\n"
            f"Last close: `{last_price:.2f}`\n"
            f"Predicted next close: `{predicted:.2f}`\n"
            f"Move (approx): `{delta:+.2f}`\n"
            f"Direction: {direction}\n"
            f"Stability score: `{stability*100:.1f}%`\n\n"
        )
    else:
        dayN = preds[-1]
        direction = "📈 UP" if dayN > last_price else "📉 DOWN"
        deltaN = dayN - last_price
        rows = [f"• {d.strftime('%Y-%m-%d')}: `{p:.2f}`" for d, p in zip(future_dates, preds)]
        path = "\n".join(rows)

        text = (
            f"*{ticker} Predict (7D)*\n\n"
            f"Last close: `{last_price:.2f}`\n"
            f"Day 7 prediction: `{dayN:.2f}`\n"
            f"Move (approx): `{deltaN:+.2f}`\n"
            f"Direction (vs last): {direction}\n"
            f"Stability score: `{stability*100:.1f}%`\n\n"
            f"*Next predicted closes:*\n{path}\n\n"
        )

    text += rule_reasoning

    # Add LLM reasoning only if present (prevents empty blocks)
    if llm_reasoning.strip():
        text += "\n\n🤖 *AI explanation (OpenAI):*\n" + sanitize_md(llm_reasoning)

    text += f"\n\n⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
    return text


def usage_text() -> str:
    return (
        "✅ Use:\n"
        "• `/predict META 1d`\n"
        "• `/predict META 7d`\n"
        "• `/chat` to talk to the AI assistant\n\n"
        f"Allowed: {', '.join(sorted(ALLOWED_TICKERS))}"
    )


# ======================
# CORE
# ======================
def do_predict(ticker: str, mode: str) -> str:
    days = 1 if mode == "1d" else 7
    preds, last_price, stability, drift, future_dates, rsi, vol_pct, sigma, mu = predict_prices(ticker, days=days)

    news = fetch_news_yahoo(ticker)
    avg_score, top_items, used_count = analyze_news(news, lookback_days=NEWS_LOOKBACK_DAYS, limit=NEWS_LIMIT)

    rule_reasoning = build_rule_based_reasoning(
        drift=drift, rsi=rsi, vol_pct=vol_pct, avg_news=avg_score, used_news=used_count
    )

    llm_reasoning = llm_explain_prediction(
        ticker=ticker,
        mode=mode,
        last_price=last_price,
        preds=preds,
        drift=drift,
        sigma=sigma,
        rsi=rsi,
        vol_pct=vol_pct,
        news_top=top_items,
    )

    return build_predict_text(
        ticker=ticker,
        mode=mode,
        preds=preds,
        last_price=last_price,
        stability=stability,
        future_dates=future_dates,
        rule_reasoning=rule_reasoning,
        llm_reasoning=llm_reasoning,
    )


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
        "Commands:\n"
        "• `/predict META 1d`\n"
        "• `/predict META 7d`\n"
        "• `/chat` (talk to AI assistant)\n"
        "• `/exit` (exit chat)\n\n"
        "Or use the menu buttons below 👇",
        reply_markup=main_menu(),
        parse_mode="Markdown"
    )


@bot.message_handler(commands=["status"])
def status(message):
    ai_status = "✅ OpenAI enabled" if openai_available() else "❌ OpenAI not configured (set OPENAI_API_KEY)"
    bot.reply_to(
        message,
        f"✅ Bot RUNNING\nSource: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
        "Command: /predict <TICKER> <1d|7d>\n"
        f"News lookback: {NEWS_LOOKBACK_DAYS} days\n"
        f"{ai_status}"
    )


@bot.message_handler(commands=["chat"])
def chat_on(message):
    CHAT_MODE_USERS.add(message.from_user.id)
    bot.send_message(
        message.chat.id,
        "💬 Chat mode is ON.\nSend any message and I will answer like ChatGPT.\nType /exit to stop."
    )


@bot.message_handler(commands=["exit"])
def chat_off(message):
    CHAT_MODE_USERS.discard(message.from_user.id)
    bot.send_message(message.chat.id, "🛑 Chat mode is OFF. Use /predict to get forecasts.")


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


@bot.message_handler(func=lambda m: True)
def any_message(message):
    # If user is in chat mode -> OpenAI
    if message.from_user.id in CHAT_MODE_USERS:
        reply = llm_chat_reply(message.from_user.id, message.text)
        bot.send_message(message.chat.id, reply)
        return

    # Otherwise show a hint
    bot.send_message(
        message.chat.id,
        "Use `/predict META 1d` or `/predict META 7d`.\n"
        "Or type `/chat` to talk to the AI assistant."
    )


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
                f"✅ Bot RUNNING\nSource: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
                "Command: /predict <TICKER> <1d|7d>\n"
                f"News lookback: {NEWS_LOOKBACK_DAYS} days\n"
                f"{ai_status}"
            )
            return

        if call.data == "chat_on":
            CHAT_MODE_USERS.add(call.from_user.id)
            bot.send_message(call.message.chat.id, "💬 Chat mode is ON. Type /exit to stop.")
            return

        if call.data == "chat_off":
            CHAT_MODE_USERS.discard(call.from_user.id)
            bot.send_message(call.message.chat.id, "🛑 Chat mode is OFF.")
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
    print("Bot started ✅ (returns-based predictor + Yahoo news + OpenAI chat/explain)")
    bot.infinity_polling()
