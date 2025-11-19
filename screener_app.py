#!/usr/bin/env python3
"""
Streamlit UI for a VALUE-FOCUSED stock/fund screener with optional MOMENTUM mode.

Fundamentals (stocks only):
    - TTM net income > 0
    - Total debt / total assets < 0.5
    - Source: yfinance (quarterly), fallback to Financial Modeling Prep (FMP_API_KEY / secrets)

Market regime:
    - SPY last close > SPY 200-day SMA

Styles:

1) VALUE:
    - Stocks:
        * RSI(14) in configurable range (mildly oversold / value zone)
        * Optional: price < 20-day SMA (mid Bollinger band)  [discounted vs recent mean]
        * Optional: price > lower Bollinger band            [not a falling knife]
    - Funds / ETFs / indices / preferreds / bonds:
        * RSI(14) < configurable max (default 30)

2) MOMENTUM (volatility breakout):
    - Stocks:
        * Close > upper 20-day Bollinger band (BBU_20) [volatility run breakout]
    - Funds / ETFs / indices / preferreds / bonds:
        * RSI(14) < configurable max (same as value-style funds)

Usage:
    python -m streamlit run screener_app.py
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st


SPY_TICKER = "SPY"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


# ---------- Dataclasses ----------

@dataclass
class FundamentalResult:
    ttm_net_income: Optional[float]
    total_debt: Optional[float]
    total_assets: Optional[float]
    debt_asset_ratio: Optional[float]
    fundamentals_ok: bool
    source: str
    reason: str


@dataclass
class MarketRegimeResult:
    spy_last_close: float
    spy_sma_200: float
    spy_trend_ok: bool


@dataclass
class TechnicalResult:
    is_fund_like: bool  # True for funds/ETFs/indices/prefs/bonds; False for common stock
    last_close: float
    rsi_14: Optional[float]
    bb_lower_20: Optional[float]
    bb_mid_20: Optional[float]
    technical_ok: bool
    reason: str


@dataclass
class ScreenResult:
    ticker: str
    is_fund_like: bool
    instrument_type: str  # e.g., "Common stock", "ETF", "Preferred / Bond"
    fundamentals: FundamentalResult
    market: MarketRegimeResult
    technical: TechnicalResult
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        if self.is_fund_like:
            # For fund-like instruments, we only gate on market + technicals
            return self.market.spy_trend_ok and self.technical.technical_ok
        # For true stocks, require fundamentals as well
        return (
            self.fundamentals.fundamentals_ok
            and self.market.spy_trend_ok
            and self.technical.technical_ok
        )


@dataclass
class StrategyConfig:
    """
    Configurable strategy parameters.
    style: "value" or "momentum"
    """
    profile_name: str
    style: str  # "value" or "momentum"
    stock_rsi_min: float
    stock_rsi_max: float
    require_stock_below_mid: bool
    require_stock_above_lower: bool
    fund_rsi_max: float


# ---------- Helper functions ----------

def detect_is_fund_symbol_heuristic(symbol: str) -> bool:
    """Simple heuristic for fund vs stock based on symbol only."""
    s = symbol.upper()
    if s.endswith("X"):
        return True
    etf_like = {
        "SPY", "GLD", "QQQ", "VTI", "VOO",
        "IWM", "XLK", "XLF", "XLV", "XLE", "XLU",
    }
    return s in etf_like


def classify_instrument(ticker_obj: yf.Ticker, symbol: str) -> Tuple[str, bool]:
    """
    Classify instrument using yfinance metadata and symbol patterns.

    Returns:
        (instrument_type_label, is_fund_like)

    - is_fund_like = False → treat as common stock (apply fundamentals)
    - is_fund_like = True  → treat as fund-like (skip fundamentals; use fund rules)
    """
    sym = symbol.upper()
    info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        info = {}

    quote_type = str(info.get("quoteType") or "").upper()
    type_disp = str(info.get("typeDisp") or "").upper()
    category = quote_type or type_disp

    # Preferred / baby bond pattern: e.g., PBIPRB, SEALPRB, XYZ.PR.A, XYZPRC
    if ("PR" in sym and len(sym) >= 4) or "PREF" in sym:
        return "Preferred / Bond (symbol pattern)", True

    # Use quoteType when available
    if category in ("EQUITY", "COMMONSTOCK", "COMMON_STOCK"):
        return "Common stock", False
    if category in ("ETF", "MUTUALFUND", "MUTUAL_FUND", "INDEX", "FUND", "CLOSEDENDFUND", "CLOSED_END_FUND"):
        return "Fund / ETF / Index", True
    if category in ("PREFERRED_STOCK", "PREFERRED", "PFD", "BOND", "CORPORATE_BOND"):
        return "Preferred / Bond", True

    # Fallback on symbol-based ETF/fund heuristic
    if detect_is_fund_symbol_heuristic(sym):
        return "Fund-like (heuristic)", True

    # Default: assume stock, but flag classification as heuristic
    return "Stock (heuristic)", False


# --- Fundamentals via yfinance ---

def get_fundamentals_yf(ticker_obj: yf.Ticker) -> Tuple[float, float, Optional[float], str]:
    try:
        inc = ticker_obj.quarterly_income_stmt
    except Exception as e:
        raise RuntimeError(f"yfinance income statement error: {e}")
    if inc is None or inc.empty:
        raise RuntimeError("yfinance: quarterly income statement empty.")

    net_income_row_candidates = [
        "Net Income",
        "NetIncome",
        "Net Income Applicable To Common Shares",
    ]
    net_income_series = None
    for label in net_income_row_candidates:
        if label in inc.index:
            net_income_series = inc.loc[label]
            break
    if net_income_series is None:
        raise RuntimeError("yfinance: net income row not found.")

    ttm_net_income = float(net_income_series.iloc[:4].sum())

    try:
        bs = ticker_obj.quarterly_balance_sheet
    except Exception as e:
        raise RuntimeError(f"yfinance balance sheet error: {e}")
    if bs is None or bs.empty:
        raise RuntimeError("yfinance: quarterly balance sheet empty.")

    assets_row_candidates = ["Total Assets", "TotalAssets"]
    total_assets = None
    for label in assets_row_candidates:
        if label in bs.index:
            total_assets = float(bs.loc[label].iloc[0])
            break

    debt_row_candidates = ["Total Debt", "TotalDebt"]
    total_debt = None
    for label in debt_row_candidates:
        if label in bs.index:
            total_debt = float(bs.loc[label].iloc[0])
            break
    if total_debt is None:
        st_labels = ["Short Long Term Debt", "Short Term Debt"]
        lt_labels = ["Long Term Debt", "LongTermDebt"]
        short_debt = next(
            (float(bs.loc[l].iloc[0]) for l in st_labels if l in bs.index),
            0.0,
        )
        long_debt = next(
            (float(bs.loc[l].iloc[0]) for l in lt_labels if l in bs.index),
            0.0,
        )
        total_debt = short_debt + long_debt

    reason = "Fundamentals from yfinance quarterly statements."
    return ttm_net_income, total_debt, total_assets, reason


# --- Fundamentals via FMP ---

def get_fmp_api_key() -> Optional[str]:
    """
    Prefer Streamlit secrets (for Cloud), fall back to environment variable (for local runs).
    """
    key = None
    try:
        key = st.secrets.get("FMP_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None

    if not key:
        key = os.environ.get("FMP_API_KEY")

    return key


def fmp_get_json(path: str, params: Optional[dict] = None) -> list:
    key = get_fmp_api_key()
    if not key:
        raise RuntimeError("FMP_API_KEY not set in Streamlit secrets or environment.")
    params = dict(params or {})
    params["apikey"] = key
    url = f"{FMP_BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected FMP response format: {data}")
    return data


def get_fundamentals_fmp(symbol: str) -> Tuple[float, float, Optional[float], str]:
    sym = symbol.upper()
    inc_data = fmp_get_json(
        f"/income-statement/{sym}",
        params={"period": "quarter", "limit": 4},
    )
    if not inc_data:
        raise RuntimeError("FMP: income statement data empty.")
    ttm_net_income = 0.0
    for item in inc_data[:4]:
        ni = item.get("netIncome")
        if ni is None:
            raise RuntimeError("FMP: 'netIncome' missing.")
        ttm_net_income += float(ni)

    bs_data = fmp_get_json(
        f"/balance-sheet-statement/{sym}",
        params={"period": "quarter", "limit": 1},
    )
    if not bs_data:
        raise RuntimeError("FMP: balance sheet data empty.")
    bs0 = bs_data[0]
    total_assets = bs0.get("totalAssets")
    total_assets = float(total_assets) if total_assets is not None else None

    total_debt = bs0.get("totalDebt")
    if total_debt is not None:
        total_debt = float(total_debt)
    else:
        short_debt = float(bs0.get("shortTermDebt", 0.0) or 0.0)
        long_debt = float(bs0.get("longTermDebt", 0.0) or 0.0)
        total_debt = short_debt + long_debt

    reason = "Fundamentals from Financial Modeling Prep (FMP)."
    return ttm_net_income, total_debt, total_assets, reason


# --- Unified fundamentals wrapper ---

def get_fundamentals(symbol: str, ticker_obj: yf.Ticker, is_fund_like: bool) -> FundamentalResult:
    # Skip fundamentals for fund-like instruments (funds/ETFs/indices/prefs/bonds)
    if is_fund_like:
        return FundamentalResult(
            ttm_net_income=None,
            total_debt=None,
            total_assets=None,
            debt_asset_ratio=None,
            fundamentals_ok=True,
            source="None (fund-like: fundamentals skipped).",
            reason="Fund-like instrument; fundamental rules not applied.",
        )

    ttm_net_income = None
    total_debt = None
    total_assets = None
    source = ""
    reason = ""

    try:
        ttm_net_income, total_debt, total_assets, source = get_fundamentals_yf(ticker_obj)
    except Exception as e_yf:
        try:
            ttm_net_income, total_debt, total_assets, source = get_fundamentals_fmp(symbol)
        except Exception as e_fmp:
            reason = f"Failed fundamentals (yfinance: {e_yf}; FMP: {e_fmp})"
            return FundamentalResult(
                ttm_net_income=None,
                total_debt=None,
                total_assets=None,
                debt_asset_ratio=None,
                fundamentals_ok=False,
                source="None",
                reason=reason,
            )

    if total_assets is None or total_assets == 0:
        debt_asset_ratio = None
    else:
        debt_asset_ratio = total_debt / total_assets

    fundamentals_ok = True
    notes: List[str] = []

    if ttm_net_income is None or ttm_net_income <= 0:
        fundamentals_ok = False
        notes.append(f"TTM net income <= 0 ({ttm_net_income}).")

    if debt_asset_ratio is None:
        fundamentals_ok = False
        notes.append("Debt/Assets ratio not available.")
    elif debt_asset_ratio >= 0.5:
        fundamentals_ok = False
        notes.append(f"Debt/Assets >= 0.5 ({debt_asset_ratio:.3f}).")

    if not notes:
        notes.append("Fundamentals pass all rules.")

    reason = " ".join(notes)
    return FundamentalResult(
        ttm_net_income=ttm_net_income,
        total_debt=total_debt,
        total_assets=total_assets,
        debt_asset_ratio=debt_asset_ratio,
        fundamentals_ok=fundamentals_ok,
        source=source,
        reason=reason,
    )


# --- SPY market regime ---

def get_spy_market_regime() -> MarketRegimeResult:
    spy = yf.Ticker(SPY_TICKER)
    hist = spy.history(period="1y")
    if hist.empty or "Close" not in hist:
        raise RuntimeError("Failed to fetch SPY history.")
    closes = hist["Close"]
    sma_200 = closes.rolling(200).mean().iloc[-1]
    last_close = float(closes.iloc[-1])
    spy_trend_ok = last_close > sma_200
    return MarketRegimeResult(
        spy_last_close=last_close,
        spy_sma_200=float(sma_200),
        spy_trend_ok=spy_trend_ok,
    )


# --- Technicals (VALUE + MOMENTUM) ---

def get_default_strategy() -> StrategyConfig:
    return StrategyConfig(
        profile_name="Balanced value (internal default)",
        style="value",
        stock_rsi_min=25,
        stock_rsi_max=45,
        require_stock_below_mid=True,
        require_stock_above_lower=True,
        fund_rsi_max=30,
    )


def get_technical(
    ticker_obj: yf.Ticker,
    is_fund_like: bool,
    strategy: Optional[StrategyConfig] = None,
) -> TechnicalResult:
    if strategy is None:
        strategy = get_default_strategy()

    hist = ticker_obj.history(period="1y")
    if hist.empty or "Close" not in hist:
        return TechnicalResult(
            is_fund_like=is_fund_like,
            last_close=float("nan"),
            rsi_14=None,
            bb_lower_20=None,
            bb_mid_20=None,
            technical_ok=False,
            reason="No price history.",
        )

    df = hist.copy()
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    bb = ta.bbands(df["Close"], length=20, std=2)

    last_close = float(df["Close"].iloc[-1])
    rsi_14 = float(df["RSI_14"].iloc[-1]) if pd.notna(df["RSI_14"].iloc[-1]) else None

    if bb is None or bb.empty:
        return TechnicalResult(
            is_fund_like=is_fund_like,
            last_close=last_close,
            rsi_14=rsi_14,
            bb_lower_20=None,
            bb_mid_20=None,
            technical_ok=False,
            reason="Failed to compute Bollinger Bands.",
        )

    lower_cols = [c for c in bb.columns if c.startswith("BBL_20")]
    mid_cols = [c for c in bb.columns if c.startswith("BBM_20")]
    upper_cols = [c for c in bb.columns if c.startswith("BBU_20")]

    if not lower_cols or not mid_cols or not upper_cols:
        return TechnicalResult(
            is_fund_like=is_fund_like,
            last_close=last_close,
            rsi_14=rsi_14,
            bb_lower_20=None,
            bb_mid_20=None,
            technical_ok=False,
            reason="One or more Bollinger Band columns (lower/mid/upper) not found.",
        )

    df["BBL_20"] = bb[lower_cols[0]]
    df["BBM_20"] = bb[mid_cols[0]]
    df["BBU_20"] = bb[upper_cols[0]]

    bb_lower_20 = float(df["BBL_20"].iloc[-1]) if pd.notna(df["BBL_20"].iloc[-1]) else None
    bb_mid_20 = float(df["BBM_20"].iloc[-1]) if pd.notna(df["BBM_20"].iloc[-1]) else None
    bb_upper_20 = float(df["BBU_20"].iloc[-1]) if pd.notna(df["BBU_20"].iloc[-1]) else None

    technical_ok = True
    notes: List[str] = []

    # --- MOMENTUM STYLE ---
    if strategy.style == "momentum":
        if is_fund_like:
            # Funds / ETFs / preferreds / bonds: use RSI oversold even in momentum mode
            if rsi_14 is None:
                technical_ok = False
                notes.append("RSI(14) not available.")
            elif rsi_14 >= strategy.fund_rsi_max:
                technical_ok = False
                notes.append(
                    f"RSI(14) >= fund max ({rsi_14:.2f} ≥ {strategy.fund_rsi_max})."
                )
            else:
                notes.append(
                    f"RSI(14) < fund max ({rsi_14:.2f} < {strategy.fund_rsi_max}) – passes (fund-style rule)."
                )
        else:
            # Common stocks: pure volatility breakout on upper band
            if bb_upper_20 is None:
                technical_ok = False
                notes.append("Upper Bollinger band not available.")
            else:
                if last_close > bb_upper_20:
                    notes.append(
                        f"Momentum breakout: Close {last_close:.2f} > Upper BB {bb_upper_20:.2f}."
                    )
                else:
                    technical_ok = False
                    notes.append(
                        f"Close {last_close:.2f} is not above Upper BB {bb_upper_20:.2f} "
                        "(no volatility breakout)."
                    )

            if rsi_14 is not None:
                notes.append(f"RSI(14) = {rsi_14:.2f} (informational in momentum mode).")

        reason = " ".join(notes) if notes else "Momentum rules pass."
        return TechnicalResult(
            is_fund_like=is_fund_like,
            last_close=last_close,
            rsi_14=rsi_14,
            bb_lower_20=bb_lower_20,
            bb_mid_20=bb_mid_20,
            technical_ok=technical_ok,
            reason=reason,
        )

    # --- VALUE STYLE ---
    if is_fund_like:
        # Funds / ETFs / preferreds / bonds: oversold rule with configurable max RSI
        if rsi_14 is None:
            technical_ok = False
            notes.append("RSI(14) not available.")
        elif rsi_14 >= strategy.fund_rsi_max:
            technical_ok = False
            notes.append(
                f"RSI(14) >= fund max ({rsi_14:.2f} ≥ {strategy.fund_rsi_max})."
            )
        else:
            notes.append(
                f"RSI(14) < fund max ({rsi_14:.2f} < {strategy.fund_rsi_max}) – passes."
            )
    else:
        # Stocks: VALUE RULES with configurable RSI range and BB conditions
        if rsi_14 is None:
            technical_ok = False
            notes.append("RSI(14) not available.")
        if bb_lower_20 is None or bb_mid_20 is None:
            technical_ok = False
            notes.append("Bollinger Bands not available.")
        if technical_ok:
            rsi_ok = strategy.stock_rsi_min <= rsi_14 <= strategy.stock_rsi_max
            discounted = (not strategy.require_stock_below_mid) or (last_close < bb_mid_20)
            not_crashing = (not strategy.require_stock_above_lower) or (last_close > bb_lower_20)

            if not rsi_ok:
                technical_ok = False
                notes.append(
                    f"RSI(14) not in [{strategy.stock_rsi_min}, {strategy.stock_rsi_max}] "
                    f"(value zone): {rsi_14:.2f}."
                )
            if strategy.require_stock_below_mid and not discounted:
                technical_ok = False
                notes.append(
                    f"Price is not below mid Bollinger band "
                    f"(Close {last_close:.2f} ≥ Mid {bb_mid_20:.2f})."
                )
            if strategy.require_stock_above_lower and not not_crashing:
                technical_ok = False
                notes.append(
                    f"Price is at/under lower Bollinger band "
                    f"(Close {last_close:.2f} ≤ Lower {bb_lower_20:.2f})."
                )

            if technical_ok:
                desc_parts = [
                    f"RSI in [{strategy.stock_rsi_min},{strategy.stock_rsi_max}]",
                ]
                if strategy.require_stock_below_mid:
                    desc_parts.append("price below mid BB (discounted)")
                else:
                    desc_parts.append("mid BB discount not required")
                if strategy.require_stock_above_lower:
                    desc_parts.append("price above lower BB (not a falling knife)")
                else:
                    desc_parts.append("lower BB safety not required")
                notes.append("Value conditions met: " + "; ".join(desc_parts) + ".")

    reason = " ".join(notes) if notes else "Technical rules pass."
    return TechnicalResult(
        is_fund_like=is_fund_like,
        last_close=last_close,
        rsi_14=rsi_14,
        bb_lower_20=bb_lower_20,
        bb_mid_20=bb_mid_20,
        technical_ok=technical_ok,
        reason=reason,
    )


# --- Orchestration ---

def screen_ticker(
    symbol: str,
    precomputed_market: Optional[MarketRegimeResult] = None,
    strategy: Optional[StrategyConfig] = None,
) -> ScreenResult:
    symbol = symbol.upper()
    ticker_obj = yf.Ticker(symbol)
    instrument_type, is_fund_like = classify_instrument(ticker_obj, symbol)
    try:
        fundamentals = get_fundamentals(symbol, ticker_obj, is_fund_like)
        market = precomputed_market or get_spy_market_regime()
        technical = get_technical(ticker_obj, is_fund_like, strategy=strategy)
        return ScreenResult(
            ticker=symbol,
            is_fund_like=is_fund_like,
            instrument_type=instrument_type,
            fundamentals=fundamentals,
            market=market,
            technical=technical,
            error=None,
        )
    except Exception as e:
        dummy_fund = FundamentalResult(
            ttm_net_income=None,
            total_debt=None,
            total_assets=None,
            debt_asset_ratio=None,
            fundamentals_ok=False,
            source="None",
            reason=f"Error during screening: {e}",
        )
        dummy_market = precomputed_market or MarketRegimeResult(
            spy_last_close=float("nan"),
            spy_sma_200=float("nan"),
            spy_trend_ok=False,
        )
        dummy_tech = TechnicalResult(
            is_fund_like=is_fund_like,
            last_close=float("nan"),
            rsi_14=None,
            bb_lower_20=None,
            bb_mid_20=None,
            technical_ok=False,
            reason="Technical not evaluated due to error.",
        )
        return ScreenResult(
            ticker=symbol,
            is_fund_like=is_fund_like,
            instrument_type=instrument_type,
            fundamentals=dummy_fund,
            market=dummy_market,
            technical=dummy_tech,
            error=str(e),
        )


def batch_screen_tickers(
    tickers: List[str],
    delay_seconds: float = 0.0,
    strategy: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
    rows: List[dict] = []

    if strategy is None:
        strategy = get_default_strategy()

    try:
        market = get_spy_market_regime()
    except Exception:
        market = MarketRegimeResult(
            spy_last_close=float("nan"),
            spy_sma_200=float("nan"),
            spy_trend_ok=False,
        )

    for i, sym in enumerate(tickers):
        res = screen_ticker(sym, precomputed_market=market, strategy=strategy)
        rows.append({
            "Symbol": res.ticker,
            "InstrumentType": res.instrument_type,
            "StockOrFundLike": "Fund-like" if res.is_fund_like else "Stock",
            "Style": strategy.style,
            "Profile": strategy.profile_name,
            "StockRSIMin": strategy.stock_rsi_min,
            "StockRSIMax": strategy.stock_rsi_max,
            "RequireBelowMid": strategy.require_stock_below_mid,
            "RequireAboveLower": strategy.require_stock_above_lower,
            "FundRSIMax": strategy.fund_rsi_max,
            "OverallPass": res.passed,
            "Error": res.error,
            "FundamentalsOK": res.fundamentals.fundamentals_ok,
            "FundSource": res.fundamentals.source,
            "FundNotes": res.fundamentals.reason,
            "TTMNetIncome": res.fundamentals.ttm_net_income,
            "TotalDebt": res.fundamentals.total_debt,
            "TotalAssets": res.fundamentals.total_assets,
            "DebtAssetsRatio": res.fundamentals.debt_asset_ratio,
            "SPYLastClose": res.market.spy_last_close,
            "SPYSMA200": res.market.spy_sma_200,
            "SPYTrendOK": res.market.spy_trend_ok,
            "LastClose": res.technical.last_close,
            "RSI14": res.technical.rsi_14,
            "BBLower20": res.technical.bb_lower_20,
            "BBMid20": res.technical.bb_mid_20,
            "TechnicalOK": res.technical.technical_ok,
            "TechNotes": res.technical.reason,
        })

        if delay_seconds > 0 and i < len(tickers) - 1:
            time.sleep(delay_seconds)

    df = pd.DataFrame(rows)

    if not df.empty:
        df.insert(
            5,
            "Overall",
            df["OverallPass"].map(lambda x: "👍 PASS" if x else "👎 NO"),
        )

    return df


# ---------- Strategy builder (UI) ----------

def build_strategy_from_sidebar() -> StrategyConfig:
    st.subheader("Strategy")

    style_choice = st.radio(
        "Trading style",
        ["Value", "Momentum (volatility breakout for stocks)"],
        index=0,
    )

    # MOMENTUM STYLE CONFIG
    if style_choice.startswith("Momentum"):
        st.caption(
            "Momentum breakout for stocks:\n"
            "- **Stocks**: Close must be above the upper 20-day Bollinger band.\n"
            "- **Funds / ETFs / preferreds / bonds**: still use RSI oversold (fund-style rule).\n"
            "- Fundamentals (for stocks) and SPY regime still apply."
        )
        return StrategyConfig(
            profile_name="Momentum breakout (stocks)",
            style="momentum",
            stock_rsi_min=0.0,
            stock_rsi_max=100.0,
            require_stock_below_mid=False,
            require_stock_above_lower=False,
            fund_rsi_max=30.0,  # still oversold threshold for fund-like instruments
        )

    # VALUE STYLE CONFIG
    st.subheader("Stock value profile")
    mode = st.radio(
        "Choose a value profile:",
        [
            "Balanced value (default)",
            "Conservative value",
            "Aggressive bargain",
            "Custom",
        ],
        index=0,
    )

    if mode == "Balanced value (default)":
        strategy = StrategyConfig(
            profile_name="Balanced value (default)",
            style="value",
            stock_rsi_min=25,
            stock_rsi_max=45,
            require_stock_below_mid=True,
            require_stock_above_lower=True,
            fund_rsi_max=30,
        )
        st.caption(
            "Balanced value: mildly oversold stocks in healthy trends.\n"
            "- Stocks: RSI 25–45, price between lower and mid Bollinger band.\n"
            "- Fund-like instruments: RSI < 30."
        )
        return strategy

    if mode == "Conservative value":
        strategy = StrategyConfig(
            profile_name="Conservative value",
            style="value",
            stock_rsi_min=35,
            stock_rsi_max=55,
            require_stock_below_mid=True,
            require_stock_above_lower=True,
            fund_rsi_max=35,
        )
        st.caption(
            "Conservative value: shallower dips in stronger stocks.\n"
            "- Stocks: RSI 35–55, still below mid band but not deeply oversold.\n"
            "- Fund-like instruments: RSI < 35."
        )
        return strategy

    if mode == "Aggressive bargain":
        strategy = StrategyConfig(
            profile_name="Aggressive bargain",
            style="value",
            stock_rsi_min=15,
            stock_rsi_max=35,
            require_stock_below_mid=True,
            require_stock_above_lower=True,
            fund_rsi_max=30,
        )
        st.caption(
            "Aggressive bargain: deeper value hunting.\n"
            "- Stocks: RSI 15–35, close to lower band but must remain above it.\n"
            "- Fund-like instruments: RSI < 30."
        )
        return strategy

    # Custom value profile
    st.markdown("**Custom stock parameters**")

    stock_rsi_min = st.slider(
        "Stock RSI min (oversold floor)",
        min_value=0,
        max_value=60,
        value=25,
        step=1,
        help="Lower values mean you only buy very oversold stocks.",
    )
    stock_rsi_max = st.slider(
        "Stock RSI max (recovery ceiling)",
        min_value=10,
        max_value=80,
        value=45,
        step=1,
        help="Upper bound for how far RSI can recover and still be considered a 'value' entry.",
    )

    if stock_rsi_max <= stock_rsi_min:
        st.warning("RSI max should be greater than RSI min; adjust your sliders.")

    require_below_mid = st.checkbox(
        "Require price below mid Bollinger band (discounted vs recent mean)",
        value=True,
    )
    require_above_lower = st.checkbox(
        "Require price above lower Bollinger band (avoid falling knife)",
        value=True,
    )

    fund_rsi_max = st.slider(
        "Fund-like RSI max (oversold threshold)",
        min_value=10,
        max_value=50,
        value=30,
        step=1,
        help="Fund-like instruments with RSI below this value are considered oversold.",
    )

    st.caption(
        f"Custom value profile: stocks with RSI in [{stock_rsi_min}, {stock_rsi_max}], "
        f"{'require' if require_below_mid else 'do not require'} price below mid BB, "
        f"{'require' if require_above_lower else 'do not require'} price above lower BB; "
        f"fund-like instruments oversold if RSI < {fund_rsi_max}."
    )

    return StrategyConfig(
        profile_name="Custom value",
        style="value",
        stock_rsi_min=float(stock_rsi_min),
        stock_rsi_max=float(stock_rsi_max),
        require_stock_below_mid=bool(require_below_mid),
        require_stock_above_lower=bool(require_above_lower),
        fund_rsi_max=float(fund_rsi_max),
    )


def render_strategy_legend():
    """
    Legend card explaining value vs momentum presets and instrument classification.
    """
    st.markdown("### Strategy & instrument legend")
    st.info(
        "- **Instrument types**  \n"
        "  • *Common stock*: fundamentals + technicals + market regime.  \n"
        "  • *Fund / ETF / Index*: treated as fund-like (technicals only).  \n"
        "  • *Preferred / Bond*: treated as fund-like (technicals only).  \n"
        "  • *Heuristic* labels use symbol patterns or fallback rules when Yahoo metadata is missing.\n\n"
        "- **Value style**  \n"
        "  • Stocks: RSI window + band placement (discounted but not a falling knife).  \n"
        "  • Fund-like: RSI below a configurable oversold threshold (mean reversion).  \n\n"
        "- **Momentum style (stocks)**  \n"
        "  • Stocks: Close above the upper 20-day Bollinger band (volatility breakout).  \n"
        "  • Fund-like: still use RSI oversold (fund-style rule) – no upper-band requirement.  \n"
        "  • Fundamentals (for stocks) and SPY > 200d SMA still enforced."
    )


# ---------- Streamlit UI ----------

def main():
    st.title("Stock / Fund Screener — Value & Momentum")

    # Legend card
    render_strategy_legend()

    with st.sidebar:
        st.header("Settings")
        fmp_key = get_fmp_api_key()
        st.markdown(f"**FMP_API_KEY set:** {'✅' if fmp_key else '❌'}")

        strategy = build_strategy_from_sidebar()

        st.markdown(f"**Active style:** `{strategy.style}`")
        st.markdown(f"**Profile:** `{strategy.profile_name}`")

        st.markdown("### Summary of rules")
        if strategy.style == "momentum":
            st.write(
                "- **Style**: Momentum (volatility breakout for stocks).  \n"
                "- **Stocks**: Close > Upper Bollinger Band (20d).  \n"
                "- **Fund-like instruments**: RSI(14) < "
                f"{strategy.fund_rsi_max} (fund-style oversold rule).  \n"
                "- **Stocks** still require TTM NI>0 and Debt/Assets<0.5.  \n"
                "- **Market**: SPY > 200d SMA."
            )
        else:
            st.write(
                f"- **Style**: Value.  \n"
                f"- **Stocks**: TTM NI>0, Debt/Assets<0.5; "
                f"RSI(14) ∈ [{strategy.stock_rsi_min},{strategy.stock_rsi_max}]."
            )
            bb_bits = []
            if strategy.require_stock_below_mid:
                bb_bits.append("price below mid BB (discounted)")
            else:
                bb_bits.append("mid BB discount not required")
            if strategy.require_stock_above_lower:
                bb_bits.append("price above lower BB (not a falling knife)")
            else:
                bb_bits.append("lower BB safety not required")
            st.write(f"  - Stock price conditions: {', '.join(bb_bits)}.")
            st.write(f"- **Fund-like**: RSI(14) < {strategy.fund_rsi_max}.")
            st.write("- **Market**: SPY > 200d SMA.")

    tab_single, tab_batch = st.tabs(["Single Ticker", "Batch"])

    with tab_single:
        ticker = st.text_input("Ticker symbol", value="LYB")
        if st.button("Run Screen", type="primary"):
            if not ticker.strip():
                st.warning("Please enter a ticker.")
            else:
                res = screen_ticker(ticker.strip(), strategy=strategy)
                st.subheader(
                    f"Result for {res.ticker} "
                    f"({res.instrument_type}) — "
                    f"{strategy.style.capitalize()} / {strategy.profile_name}"
                )
                if res.error:
                    st.error(f"Error: {res.error}")

                overall_label = "👍 PASS" if res.passed else "👎 DO NOT BUY"
                st.subheader(f"Overall: {overall_label}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Instrument type", res.instrument_type)
                with col2:
                    st.metric("Fund-like?", "Yes" if res.is_fund_like else "No")
                with col3:
                    st.metric("SPY > 200d SMA?", "Yes" if res.market.spy_trend_ok else "No")

                st.markdown("### Fundamentals (stocks only)")
                fund_df = pd.DataFrame({
                    "Metric": ["Source", "TTM Net Income", "Total Debt", "Total Assets",
                               "Debt/Assets", "Fundamentals OK", "Notes"],
                    "Value": [
                        res.fundamentals.source,
                        res.fundamentals.ttm_net_income,
                        res.fundamentals.total_debt,
                        res.fundamentals.total_assets,
                        res.fundamentals.debt_asset_ratio,
                        res.fundamentals.fundamentals_ok,
                        res.fundamentals.reason,
                    ],
                })
                fund_df["Value"] = fund_df["Value"].astype(str)
                st.table(fund_df)

                st.markdown("### Market Regime (SPY)")
                market_df = pd.DataFrame({
                    "Metric": ["SPY Last Close", "SPY SMA(200)", "SPY > SMA(200)?"],
                    "Value": [
                        res.market.spy_last_close,
                        res.market.spy_sma_200,
                        res.market.spy_trend_ok,
                    ],
                })
                market_df["Value"] = market_df["Value"].astype(str)
                st.table(market_df)

                st.markdown("### Technicals")
                tech_df = pd.DataFrame({
                    "Metric": ["Last Close", "RSI(14)", "BB Lower 20", "BB Mid 20", "Technical OK", "Notes"],
                    "Value": [
                        res.technical.last_close,
                        res.technical.rsi_14,
                        res.technical.bb_lower_20,
                        res.technical.bb_mid_20,
                        res.technical.technical_ok,
                        res.technical.reason,
                    ],
                })
                tech_df["Value"] = tech_df["Value"].astype(str)
                st.table(tech_df)

    with tab_batch:
        st.write("Enter one ticker per line.")
        raw = st.text_area("Tickers", value="AAPL\nMSFT\nQQQ")

        delay_seconds = st.number_input(
            "Delay between tickers (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Use a small delay to avoid API rate limits when screening many symbols.",
        )

        if st.button("Run Batch Screen"):
            tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]
            if not tickers:
                st.warning("Please enter at least one ticker.")
            else:
                df = batch_screen_tickers(
                    tickers,
                    delay_seconds=delay_seconds,
                    strategy=strategy,
                )
                st.subheader(
                    f"Batch Results — Style: {strategy.style}, Profile: {strategy.profile_name}"
                )
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="screen_results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
