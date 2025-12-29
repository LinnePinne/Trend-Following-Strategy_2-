import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from math import sqrt
import os

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {
        "name": "US500",
        "csv": "US500_1H_2012-now.csv",
    },
    {
        "name": "USTECH",
        "csv": "USTEC_1H_2012-now.csv",
    },
    {
        "name": "US30",
        "csv": "US30_1H_2012-now.csv",
    },
]


START_CAPITAL = 50_000
EXPOSURE_PCT = 0.33  # andel av (aktuell) equity per entry (portfölj-sim)
MAX_GROSS_EXPOSURE = 1.0  # 1.0 = max 100% av equity investerat samtidigt

# ============================================================
# COST MODEL (POINTS)
# ============================================================

HALF = 0.5
SLIPPAGE_POINTS = 0.5
FIXED_SPREAD_POINTS = 0.8
COMM_POINTS_PER_SIDE = 0.05  # per side (entry/exit)

def commission_round_turn_points() -> float:
    """Kommission per round-turn (entry+exit) i points."""
    return 2.0 * COMM_POINTS_PER_SIDE


# ============================================================
# DATA HELPERS
# ============================================================

def load_ohlc_csv(csv_path: str, market_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # timestamp/datetime -> index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError(f"[{market_name}] Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[{market_name}] CSV måste innehålla kolumnerna: {required_cols}")

    # Sanera duplicerade timestamps (krav för reindex i portfölj)
    if df.index.has_duplicates:
        dup_count = int(df.index.duplicated().sum())
        print(f"[{market_name}] VARNING: {dup_count} duplicerade timestamps. Tar bort dubletter (keep='last').")
        df = df[~df.index.duplicated(keep="last")].copy()

    return df


# ============================================================
# SINGLE-MARKET BACKTEST
# ============================================================

def run_backtest_for_market(market_name: str, csv_path: str):
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("=" * 70 + "\n")

    # 1) Ladda data
    df = load_ohlc_csv(csv_path, market_name)

    # 2) Indikatorer
    df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_medium"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=150, adjust=False).mean()

    USE_SPREAD_COLUMN = "spread_points" in df.columns

    def get_spread_points(row) -> float:
        if USE_SPREAD_COLUMN:
            return float(row["spread_points"])
        return FIXED_SPREAD_POINTS

    # 3) Backtest-loop (1 trade åt gången per marknad)
    equity = START_CAPITAL

    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None
    entry_fill_time = None
    pos_size = 0.0
    pos_exposure = 0.0

    trades = []

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx_list[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # ======================
        # EXIT: om vi är i position
        # ======================
        if in_position:
            exit_price = None
            exit_reason = None

            # Exit-regel: när ema_fast <= ema_medium, fyll på ema_medium-nivå med spread/slippage
            if row["ema_fast"] <= row["ema_medium"]:
                spread_now = get_spread_points(row)
                exit_price = float(row["ema_medium"]) - HALF * spread_now - SLIPPAGE_POINTS
                exit_reason = "ema_touch_exit"

            if exit_price is not None:
                exit_time = ts

                # PnL i points och cash (LONG only)
                pnl_points = (exit_price - entry_price) - commission_round_turn_points()
                pnl_cash = pnl_points * pos_size

                equity += pnl_cash

                trades.append({
                    "Market": market_name,
                    "Direction": pos_direction,
                    "Entry Time": entry_time,            # signal-time
                    "Entry Fill Time": entry_fill_time,  # faktiska fill-tiden (next bar)
                    "Exit Time": exit_time,
                    "Entry Price": float(entry_price),
                    "Exit Price": float(exit_price),
                    "Exit Reason": exit_reason,
                    "Size": float(pos_size),
                    "Exposure ($)": float(pos_exposure),
                    "Commission RT (points)": float(commission_round_turn_points()),
                    "PnL (points)": float(pnl_points),
                    "PnL ($)": float(pnl_cash),
                    "Equity After": float(equity),
                })

                # reset position
                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                entry_fill_time = None
                pos_size = 0.0
                pos_exposure = 0.0

            # om fortfarande i position -> inga nya entries
            if in_position:
                continue

        # ======================
        # ENTRY: om vi inte är i position
        # ======================
        ema_fast = row["ema_fast"]
        prev_ema_fast = prev_row["ema_fast"]
        ema_medium = row["ema_medium"]
        ema_slow = row["ema_slow"]

        if np.isnan(ema_slow):
            continue

        bullish_trend = ema_medium > ema_slow
        cross = (prev_ema_fast < ema_medium) and (ema_fast > ema_medium)
        long_entry_signal = cross and bullish_trend

        if long_entry_signal:
            # Fill på next bar open + half spread (ask)
            next_row = df.iloc[i + 1]
            next_open = float(next_row["open"])
            spread_next = get_spread_points(next_row)

            pos_direction = "LONG"
            entry_time = ts
            entry_fill_time = idx_list[i + 1]
            entry_price = next_open + HALF * spread_next

            # Fixed exposure sizing baserat på equity i denna marknad
            position_value = equity * EXPOSURE_PCT
            pos_size = position_value / entry_price if entry_price > 0 else 0.0
            pos_exposure = position_value

            in_position = True

    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print(f"[{market_name}] Inga trades hittades.")
        return None, trades_df, df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)

    # --- Statistik i CASH (rekommenderat) ---
    trades_df["is_win"] = trades_df["PnL ($)"] > 0

    gross_profit = trades_df.loc[trades_df["PnL ($)"] > 0, "PnL ($)"].sum()
    gross_loss = trades_df.loc[trades_df["PnL ($)"] < 0, "PnL ($)"].sum()  # negativt
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["PnL ($)"] > 0, "PnL ($)"].mean()
    avg_loss = trades_df.loc[trades_df["PnL ($)"] < 0, "PnL ($)"].mean()

    winrate = trades_df["is_win"].mean()
    expectancy = trades_df["PnL ($)"].mean()

    # Drawdown på Equity After
    eq = trades_df["Equity After"]
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = dd.min()  # negativ

    # Longest losing streak
    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    # “Sharpe” på trade-nivå (cash, ej tidsnormaliserad)
    pnl_std = trades_df["PnL ($)"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    stats = {
        "Market": market_name,
        "Trades": int(len(trades_df)),
        "Total PnL ($)": float(trades_df["PnL ($)"].sum()),
        "Gross Profit ($)": float(gross_profit),
        "Gross Loss ($)": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win ($)": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss ($)": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy ($/trade)": float(expectancy),
        "Max Drawdown (%)": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
        "Commission RT (points)": float(commission_round_turn_points()),
        "Spread model": "column(spread_points)" if USE_SPREAD_COLUMN else f"fixed({FIXED_SPREAD_POINTS})",
    }

    return stats, trades_df, df


# ============================================================
# MULTI-MARKET PORTFÖLJ (MTM) + PORTFÖLJ TRADE-LOG (NET $)
# ============================================================

def build_multi_mtm_portfolio(
    market_dfs: dict,              # {"US500": df, ...}
    portfolio_trades: pd.DataFrame,
    start_capital: float,
    exposure_pct: float,
    trading_days_per_year: int = 252,
):
    tr = portfolio_trades.copy()
    tr["Entry Time"] = pd.to_datetime(tr["Entry Time"])
    tr["Exit Time"] = pd.to_datetime(tr["Exit Time"])

    if "Entry Fill Time" not in tr.columns:
        raise ValueError("portfolio_trades saknar 'Entry Fill Time'. Logga den i backtesten.")
    tr["Entry Fill Time"] = pd.to_datetime(tr["Entry Fill Time"])

    required = {"Market", "Direction", "Entry Price", "Exit Price", "Entry Fill Time", "Exit Time"}
    missing = required - set(tr.columns)
    if missing:
        raise ValueError(f"portfolio_trades saknar kolumner: {missing}")

    # Bygg gemensam tidsaxel (union av alla marknadsindex)
    all_index = None
    for mkt, df in market_dfs.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")].copy()
        idx = dfx.index
        all_index = idx if all_index is None else all_index.union(idx)

    all_index = pd.DatetimeIndex(all_index.sort_values().unique())

    # Close-matris per marknad (ffill)
    closes = pd.DataFrame(index=all_index)
    for mkt, df in market_dfs.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")].copy()
        closes[mkt] = dfx["close"].reindex(all_index).ffill()

    # Events per timestamp
    entries = tr.sort_values(["Entry Fill Time", "Market"]).groupby("Entry Fill Time")
    exits = tr.sort_values(["Exit Time", "Market"]).groupby("Exit Time")

    # Position state per market
    positions = {}   # mkt -> {"size": float, "dir": "LONG", "entry": float}
    entry_meta = {}  # mkt -> {"Entry Time", "Entry Price", "Size", ...}

    portfolio_trade_log = []

    # --- NYTT: spåra exposure och antal öppna positioner över tid ---
    open_pos_count = []  # (ts, count)
    gross_exposure_pct_series = []  # (ts, gross_exposure_pct)

    cash = float(start_capital)
    equity_path = []

    for ts in all_index:
        # 1) Exits först (frigör kapital innan entries samma timestamp)
        if ts in exits.groups:
            block = exits.get_group(ts)
            for _, trade in block.iterrows():
                mkt = trade["Market"]
                if mkt not in positions:
                    continue

                pos = positions[mkt]
                size = float(pos["size"])
                entry_price = float(pos["entry"])
                exit_price = float(trade["Exit Price"])
                comm_rt_points = float(trade.get("Commission RT (points)", 0.0))

                pnl_gross = (exit_price - entry_price) * size
                pnl_comm = comm_rt_points * size
                pnl_net = pnl_gross - pnl_comm

                # Realisera i cash
                cash += size * exit_price
                cash -= pnl_comm

                meta = entry_meta.get(mkt, {})
                portfolio_trade_log.append({
                    "Market": mkt,
                    "Direction": pos["dir"],
                    "Entry Time": meta.get("Entry Time", pd.NaT),
                    "Exit Time": pd.to_datetime(trade["Exit Time"]),
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Size": size,
                    "PnL Gross ($)": pnl_gross,
                    "Commission ($)": pnl_comm,
                    "PnL Net ($)": pnl_net,
                })

                del positions[mkt]
                if mkt in entry_meta:
                    del entry_meta[mkt]

        # 2) Entries
        if ts in entries.groups:
            block = entries.get_group(ts)
            for _, trade in block.iterrows():
                mkt = trade["Market"]
                direction = trade.get("Direction", "LONG")
                if direction != "LONG":
                    raise NotImplementedError("SHORT ej implementerad i portfölj-sim.")

                if mkt in positions:
                    continue  # en position per marknad

                entry_price = float(trade["Entry Price"])

                # Mark-to-market equity just nu (innan entry)
                mtm_value = 0.0
                gross_exposure_value = 0.0
                for pmkt, pos in positions.items():
                    px = float(closes.loc[ts, pmkt])
                    pos_val = float(pos["size"]) * px
                    mtm_value += pos_val
                    gross_exposure_value += pos_val

                equity_now = cash + mtm_value

                # Cap på gross exposure
                gross_exposure_pct = gross_exposure_value / equity_now if equity_now > 0 else 0.0
                remaining_capacity = max(0.0, MAX_GROSS_EXPOSURE - gross_exposure_pct)

                position_value = equity_now * exposure_pct
                position_value = min(position_value, equity_now * remaining_capacity)

                # Ingen hävstång: kan inte investera mer än cash
                if position_value > cash:
                    position_value = max(0.0, cash)

                size = position_value / entry_price if entry_price > 0 else 0.0
                cash -= position_value

                positions[mkt] = {"size": size, "dir": "LONG", "entry": entry_price}
                entry_meta[mkt] = {
                    "Entry Time": pd.to_datetime(trade["Entry Fill Time"]),
                    "Entry Price": entry_price,
                    "Size": size,
                    "Commission RT (points)": float(trade.get("Commission RT (points)", 0.0)),
                }

        # 3) MTM på close
        mtm_value = 0.0
        for pmkt, pos in positions.items():
            px = float(closes.loc[ts, pmkt])
            mtm_value += float(pos["size"]) * px

        equity = cash + mtm_value
        equity_path.append((ts, equity))
        gross_exposure_value = 0.0
        for pmkt, pos in positions.items():
            px = float(closes.loc[ts, pmkt])
            gross_exposure_value += float(pos["size"]) * px

        gross_exposure_pct = gross_exposure_value / equity if equity > 0 else 0.0
        open_pos_count.append((ts, len(positions)))
        gross_exposure_pct_series.append((ts, gross_exposure_pct))

    equity_series = pd.Series(
        data=[v for _, v in equity_path],
        index=pd.DatetimeIndex([t for t, _ in equity_path]),
        name="Portfolio_Equity_MTM"
    )

    open_pos_series = pd.Series(
        data=[v for _, v in open_pos_count],
        index=pd.DatetimeIndex([t for t, _ in open_pos_count]),
        name="OpenPositions"
    )

    gross_exposure_series = pd.Series(
        data=[v for _, v in gross_exposure_pct_series],
        index=pd.DatetimeIndex([t for t, _ in gross_exposure_pct_series]),
        name="GrossExposurePct"
    )

    time_in_market = float((gross_exposure_series > 0).mean() * 100.0)
    max_concurrent = int(open_pos_series.max())
    avg_concurrent = float(open_pos_series.mean())
    avg_gross_exposure = float(gross_exposure_series.mean() * 100.0)
    max_gross_exposure = float(gross_exposure_series.max() * 100.0)

    # Daily returns via daily last
    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    # Metrics
    ret_mean = daily_returns.mean()
    ret_std = daily_returns.std(ddof=1)

    sharpe = np.nan
    if ret_std and ret_std > 0:
        sharpe = (ret_mean / ret_std) * np.sqrt(trading_days_per_year)

    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    max_dd = dd.min()

    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    cagr = np.nan
    if n_days > 0:
        years = n_days / 365.25
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0

    calmar = np.nan
    if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0:
        calmar = cagr / abs(max_dd)

    metrics = {
        "Equity Start": float(equity_series.iloc[0]),
        "Equity End": float(equity_series.iloc[-1]),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "Max Drawdown (%)": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        "Sharpe (ann.)": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "Avg Daily Return": float(ret_mean) if pd.notna(ret_mean) else np.nan,
        "Daily Vol": float(ret_std) if pd.notna(ret_std) else np.nan,
        "Open Positions (end)": int(len(positions)),
        "Time in Market (%)": time_in_market,
        "Max Concurrent Positions": max_concurrent,
        "Avg Concurrent Positions": avg_concurrent,
        "Avg Gross Exposure (%)": avg_gross_exposure,
        "Max Gross Exposure (%)": max_gross_exposure,
    }

    portfolio_trades_mtm = pd.DataFrame(portfolio_trade_log)
    return metrics, equity_series, daily_returns, portfolio_trades_mtm, open_pos_series, gross_exposure_series

# ============================================================
# MAIN: KÖR ALLA MARKNADER -> BYGG PORTFÖLJ -> RAPPORT + PLOT
# ============================================================

all_results = []
all_trades = []
market_dfs = {}

for m in markets:
    try:
        stats, trades_df, df = run_backtest_for_market(m["name"], m["csv"])
        if stats is not None and trades_df is not None and not trades_df.empty:
            all_results.append(stats)
            all_trades.append(trades_df)
            market_dfs[m["name"]] = df
    except Exception as e:
        print(f"\n*** FEL för {m['name']} ({m['csv']}): {e}\n")

if not all_trades:
    raise RuntimeError("Inga trades i någon marknad.")

portfolio_trades = pd.concat(all_trades, ignore_index=True)

# Portfölj MTM + portföljens egen trade-log (net $)
portfolio_metrics, portfolio_equity, portfolio_rets, portfolio_trades_mtm, open_pos_series, gross_exposure_series = build_multi_mtm_portfolio(
    market_dfs=market_dfs,
    portfolio_trades=portfolio_trades,
    start_capital=START_CAPITAL,
    exposure_pct=EXPOSURE_PCT,
    trading_days_per_year=252,
)

print("\n--- PORTFÖLJ METRICS (MULTI-MARKET MTM) ---")
for k, v in portfolio_metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

# Portfölj trade-stats (från portfölj-simen, dvs konsekvent med equity)
if not portfolio_trades_mtm.empty:
    t = portfolio_trades_mtm.copy()
    t["is_win"] = t["PnL Net ($)"] > 0

    gross_profit = t.loc[t["PnL Net ($)"] > 0, "PnL Net ($)"].sum()
    gross_loss = t.loc[t["PnL Net ($)"] < 0, "PnL Net ($)"].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    print("\n--- PORTFÖLJ TRADE-STATS (REALISERAD, NET $) ---")
    print("Trades:", len(t))
    print("Total Realized PnL ($):", float(t["PnL Net ($)"].sum()))
    print("Winrate:", float(t["is_win"].mean()))
    print("Profit Factor:", float(profit_factor))

    # Sanity check: om inga öppna positioner vid slutet ska detta matcha
    print("\nSanity (Equity End - Start):", float(portfolio_equity.iloc[-1] - portfolio_equity.iloc[0]))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(portfolio_equity.index, portfolio_equity.values)
plt.title("Portfolio Equity Curve (MTM, Multi-market, Fixed % Exposure per Entry)")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(open_pos_series.index, open_pos_series.values)
plt.title("Open Positions")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(gross_exposure_series.index, gross_exposure_series.values)
plt.title("Gross Exposure % (0-1)")
plt.grid(True)
plt.tight_layout()
plt.show()