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
        InlineKeyboardButton("💬 Chat mo
