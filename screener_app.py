#!/usr/bin/env python3
"""
Streamlit UI for a VALUE-FOCUSED stock/fund screener.

Rules:

Fundamentals (stocks only):
    - TTM net income > 0
    - Total debt / total assets < 0.5
    - Source: yfinance (quarterly), fallback to Financial Modeling Prep (FMP_API_KEY)

Market regime:
    - SPY last close > SPY 200-day SMA

Technicals:
    - If stock (value style):
        * RSI(14) between 25 and 45 (mildly oversold)
        * Price < 20-day SMA (mid Bollinger band)  [discounted vs recent mean]
        * Price > lower Bollinger band            [not a falling knife]
    - If fund:
        * RSI(14) < 30

Usage:
    python -m streamlit run screener_app.py
"""

import os
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
    is_fund: bool
    last_close: float
    rsi_14: Optional[float]
    bb_lower_20: Optional[float]
    bb_mid_20: Optional[float]
    technical_ok: bool
    reason: str


@dataclass
class ScreenResult:
    ticker: str
    is_fund: bool
    fundamentals: FundamentalResult
    market: MarketRegimeResult
    technical: TechnicalResult
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        if self.is_fund:
            return self.market.spy_trend_ok and self.technical.technical_ok
        return (
            self.fundamentals.fundamentals_ok
            and self.market.spy_trend_ok
            and self.technical.technical_ok
        )


# ---------- Helper functions ----------

def detect_is_fund(symbol: str) -> bool:
    """Simple heuristic for fund vs stock."""
    s = symbol.upper()
    if s.endswith("X"):
        return True
    etf_like = {
        "SPY", "GLD", "QQQ", "VTI", "VOO",
        "IWM", "XLK", "XLF", "XLV", "XLE", "XLU",
    }
    return s in etf_like


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
    return st.secrets.get("FMP_API_KEY")


def fmp_get_json(path: str, params: Optional[dict] = None) -> list:
    key = get_fmp_api_key()
    if not key:
        raise RuntimeError("FMP_API_KEY environment variable not set.")
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

def get_fundamentals(symbol: str, ticker_obj: yf.Ticker, is_fund: bool) -> FundamentalResult:
    if is_fund:
        return FundamentalResult(
            ttm_net_income=None,
            total_debt=None,
            total_assets=None,
            debt_asset_ratio=None,
            fundamentals_ok=True,
            source="None (fund: fundamentals skipped).",
            reason="Fund detected; fundamental rules not applied.",
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
    notes = []

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


# --- Technicals (VALUE rules) ---

def get_technical(ticker_obj: yf.Ticker, is_fund: bool) -> TechnicalResult:
    hist = ticker_obj.history(period="1y")
    if hist.empty or "Close" not in hist:
        return TechnicalResult(
            is_fund=is_fund,
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
            is_fund=is_fund,
            last_close=last_close,
            rsi_14=rsi_14,
            bb_lower_20=None,
            bb_mid_20=None,
            technical_ok=False,
            reason="Failed to compute Bollinger Bands.",
        )

    lower_cols = [c for c in bb.columns if c.startswith("BBL_20")]
    mid_cols = [c for c in bb.columns if c.startswith("BBM_20")]

    if not lower_cols or not mid_cols:
        return TechnicalResult(
            is_fund=is_fund,
            last_close=last_close,
            rsi_14=rsi_14,
            bb_lower_20=None,
            bb_mid_20=None,
            technical_ok=False,
            reason="Lower or middle Bollinger Band columns not found.",
        )

    df["BBL_20"] = bb[lower_cols[0]]
    df["BBM_20"] = bb[mid_cols[0]]

    bb_lower_20 = float(df["BBL_20"].iloc[-1]) if pd.notna(df["BBL_20"].iloc[-1]) else None
    bb_mid_20 = float(df["BBM_20"].iloc[-1]) if pd.notna(df["BBM_20"].iloc[-1]) else None

    technical_ok = True
    notes = []

    if is_fund:
        # Funds: classic oversold rule
        if rsi_14 is None:
            technical_ok = False
            notes.append("RSI(14) not available.")
        elif rsi_14 >= 30:
            technical_ok = False
            notes.append(f"RSI(14) >= 30 ({rsi_14:.2f}).")
        else:
            notes.append(f"RSI(14) < 30 ({rsi_14:.2f}) – passes.")
    else:
        # Stocks: VALUE RULES
        #  - RSI between 25 and 45 (mildly oversold)
        #  - Price < mid BB (discounted vs recent mean)
        #  - Price > lower BB (avoid falling knife)
        if rsi_14 is None:
            technical_ok = False
            notes.append("RSI(14) not available.")
        if bb_lower_20 is None or bb_mid_20 is None:
            technical_ok = False
            notes.append("Bollinger Bands not available.")
        if technical_ok:
            rsi_ok = 25 <= rsi_14 <= 45
            discounted = last_close < bb_mid_20
            not_crashing = last_close > bb_lower_20

            if not rsi_ok:
                technical_ok = False
                notes.append(f"RSI(14) not in [25, 45] (value zone): {rsi_14:.2f}.")
            if not discounted:
                technical_ok = False
                notes.append(
                    f"Price is not below mid Bollinger band (Close {last_close:.2f} ≥ Mid {bb_mid_20:.2f})."
                )
            if not not_crashing:
                technical_ok = False
                notes.append(
                    f"Price is at/under lower Bollinger band (Close {last_close:.2f} ≤ Lower {bb_lower_20:.2f})."
                )

            if technical_ok:
                notes.append(
                    f"Value conditions met: RSI in [25,45], price between lower ({bb_lower_20:.2f}) and mid ({bb_mid_20:.2f}) bands."
                )

    reason = " ".join(notes) if notes else "Technical rules pass."
    return TechnicalResult(
        is_fund=is_fund,
        last_close=last_close,
        rsi_14=rsi_14,
        bb_lower_20=bb_lower_20,
        bb_mid_20=bb_mid_20,
        technical_ok=technical_ok,
        reason=reason,
    )


# --- Orchestration ---

def screen_ticker(symbol: str) -> ScreenResult:
    symbol = symbol.upper()
    ticker_obj = yf.Ticker(symbol)
    is_fund = detect_is_fund(symbol)
    try:
        fundamentals = get_fundamentals(symbol, ticker_obj, is_fund)
        market = get_spy_market_regime()
        technical = get_technical(ticker_obj, is_fund)
        return ScreenResult(
            ticker=symbol,
            is_fund=is_fund,
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
        dummy_market = MarketRegimeResult(
            spy_last_close=float("nan"),
            spy_sma_200=float("nan"),
            spy_trend_ok=False,
        )
        dummy_tech = TechnicalResult(
            is_fund=is_fund,
            last_close=float("nan"),
            rsi_14=None,
            bb_lower_20=None,
            bb_mid_20=None,
            technical_ok=False,
            reason="Technical not evaluated due to error.",
        )
        return ScreenResult(
            ticker=symbol,
            is_fund=is_fund,
            fundamentals=dummy_fund,
            market=dummy_market,
            technical=dummy_tech,
            error=str(e),
        )


def batch_screen_tickers(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for sym in tickers:
        res = screen_ticker(sym)
        rows.append({
            "Symbol": res.ticker,
            "Type": "Fund" if res.is_fund else "Stock",
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
    return pd.DataFrame(rows)


# ---------- Streamlit UI ----------

def main():
    st.title("Value-Focused Stock / Fund Screener")

    with st.sidebar:
        st.header("Settings")
        fmp_key = get_fmp_api_key()
        st.markdown(f"**FMP_API_KEY set:** {'✅' if fmp_key else '❌'}")
        st.write("Rules:")
        st.write("- **Stocks**: TTM NI>0, Debt/Assets<0.5; RSI(14) ∈ [25,45]; Close between lower & mid BB.")
        st.write("- **Funds**: RSI(14) < 30.")
        st.write("- **Market**: SPY > 200d SMA.")

    tab_single, tab_batch = st.tabs(["Single Ticker", "Batch"])

    with tab_single:
        ticker = st.text_input("Ticker symbol", value="LYB")
        if st.button("Run Screen", type="primary"):
            if not ticker.strip():
                st.warning("Please enter a ticker.")
            else:
                res = screen_ticker(ticker.strip())
                st.subheader(f"Result for {res.ticker} ({'Fund' if res.is_fund else 'Stock'})")
                if res.error:
                    st.error(f"Error: {res.error}")

                # >>> NEW: Big thumbs-up/thumbs-down Overall indicator
                overall_label = "👍 PASS" if res.passed else "👎 DO NOT BUY"
                st.subheader(f"Overall: {overall_label}")
                # <<<

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Type", "Fund" if res.is_fund else "Stock")
                with col2:
                    st.metric("SPY > 200d SMA?", "Yes" if res.market.spy_trend_ok else "No")
                with col3:
                    st.metric("Technical OK?", "Yes" if res.technical.technical_ok else "No")

                st.markdown("### Fundamentals")
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
                fund_ok = "Yes" if res.fundamentals.fundamentals_ok else "No"
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
                spy_ok = "Yes" if res.market.spy_trend_ok else "No"
                market_df["Value"] = market_df["Value"].astype(str)
                st.table(market_df)

                st.markdown("### Technicals (Value Rules)")
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
                tech_ok = "Yes" if res.technical.technical_ok else "No"
                tech_df["Value"] = tech_df["Value"].astype(str)
                st.table(tech_df)

    with tab_batch:
        st.write("Enter one ticker per line.")
        raw = st.text_area("Tickers", value="LYB\nGLD\nVTI")
        if st.button("Run Batch Screen"):
            tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]
            if not tickers:
                st.warning("Please enter at least one ticker.")
            else:
                df = batch_screen_tickers(tickers)
                st.subheader("Batch Results")
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

