"""
DART Congestion Arbitrage Engine.
Computes Day-Ahead vs. Real-Time (DART) spreads and flags underpriced constraints.
"""

from datetime import datetime, timedelta
import logging
import pandas as pd
from typing import List, Dict, Any

from src.storage.duckdb_store import store

logger = logging.getLogger(__name__)

def calculate_dart_spreads(iso: str = "spp", days: int = 30) -> List[Dict[str, Any]]:
    """
    Computes rolling N-day historical DART spreads per constraint.
    
    Formula:
        DART_Spread = RTBM_Shadow_Price - DAM_Shadow_Price
        (Where RTBM shadow prices are averaged hourly to align with DAM)
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days)
    
    logger.info(f"Computing DART spreads for ISO {iso} over the last {days} days ({start_date.date()} to {now.date()})")
    
    # 1. Fetch RTBM constraints
    try:
        rt_df = store.query_constraints(start=start_date, end=now, iso=iso)
    except Exception as e:
        logger.error(f"Error querying RTBM constraints for DART: {e}")
        rt_df = pd.DataFrame()
        
    # 2. Fetch DAM constraints
    try:
        da_df = store.query_da_constraints(start=start_date, end=now, iso=iso)
    except Exception as e:
        logger.error(f"Error querying DAM constraints for DART: {e}")
        da_df = pd.DataFrame()
        
    if rt_df.empty or da_df.empty:
        # A spread needs both sides. Treating a missing market as a shadow price of 0
        # would flag every constraint as "underpriced".
        logger.warning("DART needs both real-time and day-ahead constraint history; one is empty.")
        return []
        
    # 3. Process RTBM constraints
    rt_hourly = pd.DataFrame(columns=["hour", "constraint_name", "rtbm_sp"])
    if not rt_df.empty and "interval" in rt_df.columns and "constraint_name" in rt_df.columns:
        rt_df = rt_df.copy()
        rt_df["interval"] = pd.to_datetime(rt_df["interval"])
        # SPP interval labels are interval-END times (DA hour 01:00 = HE01 = 00:00-01:00),
        # so RT intervals 00:05..01:00 belong to the bucket labelled 01:00 (ceil, not floor).
        rt_df["hour"] = rt_df["interval"].dt.ceil("h")
        rt_df["shadow_price"] = pd.to_numeric(rt_df["shadow_price"], errors="coerce").fillna(0.0)
        
        # Group by hour and constraint to get hourly average shadow price
        rt_grouped = rt_df.groupby(["hour", "constraint_name"])["shadow_price"].mean().reset_index()
        rt_grouped = rt_grouped.rename(columns={"shadow_price": "rtbm_sp"})
        rt_hourly = rt_grouped

    # 4. Process DAM constraints
    da_hourly = pd.DataFrame(columns=["hour", "constraint_name", "dam_sp"])
    if not da_df.empty and "interval" in da_df.columns and "constraint_name" in da_df.columns:
        da_df = da_df.copy()
        da_df["interval"] = pd.to_datetime(da_df["interval"])
        da_df["hour"] = da_df["interval"].dt.ceil("h")
        da_df["shadow_price"] = pd.to_numeric(da_df["shadow_price"], errors="coerce").fillna(0.0)
        
        da_grouped = da_df.groupby(["hour", "constraint_name"])["shadow_price"].mean().reset_index()
        da_grouped = da_grouped.rename(columns={"shadow_price": "dam_sp"})
        da_hourly = da_grouped

    # 5. Merge on constraint_name and hour to align.
    # Only market days that actually have Day-Ahead data are comparable: for those days a
    # constraint absent from the DA file really did have a DA shadow price of 0, but for
    # days with no DA file at all we know nothing, and 0 would be fabricated.
    if rt_hourly.empty or da_hourly.empty:
        return []
    _market_day = lambda h: (h - pd.Timedelta(hours=1)).dt.normalize()   # HE24 (00:00) belongs to the prior day
    da_days = set(_market_day(da_hourly["hour"]))
    rt_hourly = rt_hourly[_market_day(rt_hourly["hour"]).isin(da_days)]
    if rt_hourly.empty:
        return []
        
    if rt_hourly.empty:
        merged = da_hourly.copy()
        merged["rtbm_sp"] = 0.0
    elif da_hourly.empty:
        merged = rt_hourly.copy()
        merged["dam_sp"] = 0.0
    else:
        merged = pd.merge(rt_hourly, da_hourly, on=["hour", "constraint_name"], how="outer")
        merged["rtbm_sp"] = merged["rtbm_sp"].fillna(0.0)
        merged["dam_sp"] = merged["dam_sp"].fillna(0.0)
        
    # Calculate DART Spread per aligned hour
    merged["spread"] = merged["rtbm_sp"] - merged["dam_sp"]
    
    # 6. Aggregate per constraint
    summary_data = []
    unique_constraints = merged["constraint_name"].unique()
    
    for name in unique_constraints:
        c_df = merged[merged["constraint_name"] == name]
        
        mean_rt = float(c_df["rtbm_sp"].mean())
        mean_da = float(c_df["dam_sp"].mean())
        mean_spread = float(c_df["spread"].mean())
        
        rt_binds = int((c_df["rtbm_sp"].abs() > 1e-4).sum())
        da_binds = int((c_df["dam_sp"].abs() > 1e-4).sum())
        
        # A constraint is underpriced in DAM if RTBM congestion is more severe (more negative shadow price)
        # i.e., abs(mean_rt) > abs(mean_da) AND mean_rt < 0.
        underpriced = abs(mean_rt) > abs(mean_da) and mean_rt < -0.01
        
        summary_data.append({
            "constraint_name": name,
            "mean_rtbm_shadow_price": round(mean_rt, 4),
            "mean_dam_shadow_price": round(mean_da, 4),
            "mean_dart_spread": round(mean_spread, 4),
            "rtbm_binding_hours": rt_binds,
            "dam_binding_hours": da_binds,
            "underpriced_in_dam": bool(underpriced),
        })
        
    # Sort: put constraints with the most negative DART spread first (meaning RTBM is much more negative than DAM)
    summary_data = sorted(summary_data, key=lambda x: x["mean_dart_spread"])
    
    return summary_data
