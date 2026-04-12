"""
Sequence Optimizer — Hungarian algorithm for minimum-risk A→DFW→B assignment.

Given a pool of DFW arrivals (airport A, arrival time) and departures (airport B,
departure time), finds the one-to-one assignment that minimizes total weather risk.

Uses scipy.optimize.linear_sum_assignment (Jonker-Volgenant algorithm, O(n³)).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MIN_TURN = 30    # min turnaround minutes
MAX_TURN = 240   # max turnaround minutes
INFEASIBLE = 2.0 # penalty > max risk (1.0), forces infeasible pairs out


# ── Cost matrix ───────────────────────────────────────────────────────────────

def build_cost_matrix(
    arrivals: pd.DataFrame,       # cols: airport, time_min, flight, time_str
    departures: pd.DataFrame,     # cols: airport, time_min, flight, time_str
    scores_idx: pd.DataFrame,     # pair_risk_scores indexed by (airport_A, airport_B, Month)
    month: int,
    unknown_risk: float = 0.50,   # neutral score for pairs not in training data
) -> np.ndarray:
    """
    Return n_arrivals × n_departures cost matrix.
    Vectorized: cross-join → filter → batch-lookup via merge. O(n*m) pandas, not Python loops.
    """
    arr = arrivals.reset_index(drop=True)[["airport", "time_min"]].copy()
    dep = departures.reset_index(drop=True)[["airport", "time_min"]].copy()
    arr["_i"] = arr.index
    dep["_j"] = dep.index

    # Cross-join
    pairs = arr.merge(dep, how="cross", suffixes=("_a", "_b"))

    # Filter: turnaround window + no same-airport round-trip
    ta = pairs["time_min_b"] - pairs["time_min_a"]
    mask = (ta >= MIN_TURN) & (ta <= MAX_TURN) & (pairs["airport_a"] != pairs["airport_b"])
    pairs = pairs[mask].copy()
    pairs.rename(columns={"airport_a": "airport_A", "airport_b": "airport_B"}, inplace=True)
    pairs["Month"] = month

    # Batch risk lookup via merge against scores
    scores_flat = scores_idx.reset_index()[["airport_A", "airport_B", "Month", "avg_risk_score"]]
    pairs = pairs.merge(scores_flat, on=["airport_A", "airport_B", "Month"], how="left")
    pairs["avg_risk_score"] = pairs["avg_risk_score"].fillna(unknown_risk)

    # Fill cost matrix
    n, m = len(arrivals), len(departures)
    cost = np.full((n, m), INFEASIBLE, dtype=float)
    cost[pairs["_i"].values, pairs["_j"].values] = pairs["avg_risk_score"].values
    return cost


# ── Optimizer ─────────────────────────────────────────────────────────────────

def optimize_sequences(
    arrivals: pd.DataFrame,
    departures: pd.DataFrame,
    scores_idx: pd.DataFrame,
    month: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Run Hungarian algorithm → minimum-risk assignment.

    Returns:
        (sequences_df, stats_dict)
    """
    if arrivals.empty or departures.empty:
        return pd.DataFrame(), {"error": "No arrivals or departures in window"}

    cost = build_cost_matrix(arrivals, departures, scores_idx, month)
    row_ind, col_ind = linear_sum_assignment(cost)

    results = []
    for i, j in zip(row_ind, col_ind):
        c = cost[i][j]
        if c >= INFEASIBLE:
            continue
        arr = arrivals.iloc[i]
        dep = departures.iloc[j]
        ta  = int(dep["time_min"] - arr["time_min"])
        results.append({
            "Sequence":       f"{arr['airport']} → DFW → {dep['airport']}",
            "airport_A":      arr["airport"],
            "airport_B":      dep["airport"],
            "flight_in":      arr.get("flight", "—"),
            "arr_time":       arr.get("time_str", ""),
            "flight_out":     dep.get("flight", "—"),
            "dep_time":       dep.get("time_str", ""),
            "turnaround_min": ta,
            "risk_score":     c,
            "risk_label":     "HIGH" if c >= 0.70 else "MODERATE" if c >= 0.40 else "LOW",
        })

    df = pd.DataFrame(results).sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Worst-case benchmark: greedy highest-risk assignment (for comparison)
    worst_cost = _worst_case_risk(cost, row_ind, col_ind)

    feasible_costs = cost[row_ind, col_ind]
    feasible_mask  = feasible_costs < INFEASIBLE

    stats = {
        "n_arrivals":        len(arrivals),
        "n_departures":      len(departures),
        "n_matched":         int(feasible_mask.sum()),
        "feasible_pairs":    int((cost < INFEASIBLE).sum()),
        "optimal_total":     float(feasible_costs[feasible_mask].sum()),
        "optimal_avg":       float(feasible_costs[feasible_mask].mean()) if feasible_mask.any() else 0.0,
        "worst_total":       worst_cost,
        "risk_saved":        max(0.0, worst_cost - float(feasible_costs[feasible_mask].sum())),
        "pct_high":          float((feasible_costs[feasible_mask] >= 0.70).mean()) if feasible_mask.any() else 0.0,
        "cost_matrix":       cost,
        "row_ind":           row_ind,
        "col_ind":           col_ind,
    }
    return df, stats


def _worst_case_risk(cost: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> float:
    """Approximate worst-case by flipping the cost matrix (maximize = minimize negative)."""
    feasible = cost < INFEASIBLE
    if not feasible.any():
        return 0.0
    neg_cost = np.where(feasible, 1.0 - cost, INFEASIBLE)
    try:
        wr, wc = linear_sum_assignment(neg_cost)
        wc_vals = cost[wr, wc]
        return float(wc_vals[wc_vals < INFEASIBLE].sum())
    except Exception:
        return float(cost[row_ind, col_ind][cost[row_ind, col_ind] < INFEASIBLE].sum())


# ── Schedule builders ─────────────────────────────────────────────────────────

def bts_to_arrivals(day_df: pd.DataFrame, arr_start_h: int, arr_end_h: int) -> pd.DataFrame:
    """Extract arrivals → DFW from BTS day DataFrame, filtered to hour window."""
    df = day_df[day_df["Dest"] == "DFW"].copy()
    df["time_min"] = (df["CRSArrTime"] // 100) * 60 + (df["CRSArrTime"] % 100)
    df = df[(df["time_min"] >= arr_start_h * 60) & (df["time_min"] < arr_end_h * 60)]
    df["time_str"] = (df["time_min"] // 60).astype(int).astype(str).str.zfill(2) + ":" + \
                     (df["time_min"] % 60).astype(int).astype(str).str.zfill(2)
    df["flight"] = df.get("Reporting_Airline", "AA").fillna("AA").astype(str) + \
                   df["Flight_Number_Reporting_Airline"].fillna("").astype(str)
    return df.rename(columns={"Origin": "airport"})[
        ["airport", "time_min", "time_str", "flight", "Tail_Number"]
    ].dropna(subset=["airport", "time_min"]).reset_index(drop=True)


def bts_to_departures(day_df: pd.DataFrame, dep_start_h: int, dep_end_h: int) -> pd.DataFrame:
    """Extract departures from DFW from BTS day DataFrame, filtered to hour window."""
    df = day_df[day_df["Origin"] == "DFW"].copy()
    df["time_min"] = (df["CRSDepTime"] // 100) * 60 + (df["CRSDepTime"] % 100)
    df = df[(df["time_min"] >= dep_start_h * 60) & (df["time_min"] < dep_end_h * 60)]
    df["time_str"] = (df["time_min"] // 60).astype(int).astype(str).str.zfill(2) + ":" + \
                     (df["time_min"] % 60).astype(int).astype(str).str.zfill(2)
    df["flight"] = df.get("Reporting_Airline", "AA").fillna("AA").astype(str) + \
                   df["Flight_Number_Reporting_Airline"].fillna("").astype(str)
    return df.rename(columns={"Dest": "airport"})[
        ["airport", "time_min", "time_str", "flight", "Tail_Number"]
    ].dropna(subset=["airport", "time_min"]).reset_index(drop=True)


_DFW_TZ_OFFSET_H = -5   # DFW = CDT (UTC-5) Apr–Oct, CST (UTC-6) Nov–Mar
try:
    import pytz as _pytz
    _DFW_PYTZ = _pytz.timezone("America/Chicago")
except ImportError:
    _DFW_PYTZ = None


def _to_dfw_local(dt) -> tuple[int, str]:
    """Convert UTC datetime → DFW local (minutes-from-midnight, display string)."""
    if _DFW_PYTZ is not None:
        dt_local = dt.astimezone(_DFW_PYTZ)
    else:
        from datetime import timedelta, timezone
        offset = _DFW_TZ_OFFSET_H
        dt_local = dt.astimezone(timezone(timedelta(hours=offset)))
    t_min = dt_local.hour * 60 + dt_local.minute
    t_str = dt_local.strftime("%H:%M CDT")
    return t_min, t_str


def aviationstack_to_arrivals(raw: list[dict], start_h: int, end_h: int) -> pd.DataFrame:
    """Parse AviationStack arrivals → standard DataFrame. Times converted UTC→DFW local."""
    from datetime import datetime, timezone as _tz
    rows = []
    for f in raw:
        arr = f.get("arrival", {})
        dep = f.get("departure", {})
        origin = dep.get("iata")
        if not origin or origin == "DFW":
            continue
        # Prefer actual > estimated > scheduled
        sched = arr.get("actual") or arr.get("estimated") or arr.get("scheduled")
        if not sched:
            continue
        try:
            dt = datetime.fromisoformat(sched.replace("Z", "+00:00"))
            t_min, t_str = _to_dfw_local(dt)
        except Exception:
            continue
        if not (start_h * 60 <= t_min < end_h * 60):
            continue
        flt = f.get("flight", {})
        rows.append({
            "airport":     origin,
            "time_min":    t_min,
            "time_str":    t_str,
            "flight":      flt.get("iata", "AA?"),
            "Tail_Number": f.get("aircraft", {}).get("registration", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def aviationstack_to_departures(raw: list[dict], start_h: int, end_h: int) -> pd.DataFrame:
    """Parse AviationStack departures → standard DataFrame. Times converted UTC→DFW local."""
    from datetime import datetime
    rows = []
    for f in raw:
        dep_info = f.get("departure", {})
        arr_info = f.get("arrival", {})
        dest = arr_info.get("iata")
        if not dest or dest == "DFW":
            continue
        sched = dep_info.get("actual") or dep_info.get("estimated") or dep_info.get("scheduled")
        if not sched:
            continue
        try:
            dt = datetime.fromisoformat(sched.replace("Z", "+00:00"))
            t_min, t_str = _to_dfw_local(dt)
        except Exception:
            continue
        if not (start_h * 60 <= t_min < end_h * 60):
            continue
        flt = f.get("flight", {})
        rows.append({
            "airport":     dest,
            "time_min":    t_min,
            "time_str":    t_str,
            "flight":      flt.get("iata", "AA?"),
            "Tail_Number": f.get("aircraft", {}).get("registration", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
