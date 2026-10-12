"""Rows must land in the file for their own market day (SPP labels intervals by END time)."""

from datetime import date

import pandas as pd
import pytest

from src.storage.duckdb_store import DuckDBStore


@pytest.fixture
def store(tmp_path):
    return DuckDBStore(data_dir=tmp_path)


def _files(store, iso="spp"):
    d = store._iso_dir(iso)
    return sorted(p.name for p in d.rglob("*.parquet"))


def _spp_day_constraints(day: str) -> pd.DataFrame:
    """A daily SPP file: first interval 00:05, last interval 00:00 of the next day."""
    ivs = pd.date_range(f"{day} 00:05", periods=288, freq="5min")
    return pd.DataFrame({"interval": ivs, "constraint_name": "C", "shadow_price": -1.0})


def test_daily_spp_file_is_stored_under_its_own_day(store):
    df = _spp_day_constraints("2026-10-10")
    assert df["interval"].max() == pd.Timestamp("2026-10-11 00:00")

    store.write_interval(df, iso="spp")

    assert _files(store) == ["constraints_20261010.parquet"]


def test_backfill_does_not_skip_the_following_day(store):
    """
    backfill_history skips a day when constraints_<day>.parquet exists. Storing day D
    under D+1 made it skip D+1, so every other day was never fetched.
    """
    from scripts.backfill_history import bc_already_stored

    store.write_interval(_spp_day_constraints("2026-10-10"), iso="spp")

    assert bc_already_stored(store, date(2026, 10, 10), iso="spp")
    assert not bc_already_stored(store, date(2026, 10, 11), iso="spp")


def test_lmp_batch_straddling_midnight_is_split(store):
    ivs = pd.date_range("2026-10-10 23:35", periods=12, freq="5min")   # 23:35 .. 00:30
    lmp = pd.DataFrame({"interval": ivs, "settlement_location": "N1", "mcc": 1.0})

    store.write_interval(pd.DataFrame(columns=["interval", "constraint_name"]), lmp_df=lmp, iso="spp")

    assert _files(store) == ["lmp_20261010.parquet", "lmp_20261011.parquet"]
    day1 = pd.read_parquet(store._lmp_dir_for_iso("spp") / "lmp_20261010.parquet")
    day2 = pd.read_parquet(store._lmp_dir_for_iso("spp") / "lmp_20261011.parquet")
    assert day1["interval"].max() == pd.Timestamp("2026-10-11 00:00")   # last interval of the 10th
    assert day2["interval"].min() == pd.Timestamp("2026-10-11 00:05")
    assert len(day1) + len(day2) == 12


def test_da_hour_ending_24_belongs_to_the_same_day(store):
    ivs = pd.date_range("2026-10-10 01:00", periods=24, freq="h")      # HE01 .. HE24 (= 00:00 next day)
    da = pd.DataFrame({"interval": ivs, "constraint_name": "C", "shadow_price": -5.0})

    store.write_da_interval(da, iso="spp")

    assert _files(store) == ["da_constraints_20261010.parquet"]


def test_non_spp_rows_are_not_shifted(store):
    """The end-of-interval shift is SPP-specific; other ISOs keep each row's own calendar day."""
    ivs = pd.to_datetime(["2026-10-10 23:55", "2026-10-11 00:00"])
    df = pd.DataFrame({"interval": ivs, "constraint_name": "C", "shadow_price": -1.0})

    store.write_interval(df, iso="miso")

    assert _files(store, "miso") == ["constraints_20261010.parquet", "constraints_20261011.parquet"]


def test_rewriting_a_day_is_idempotent(store):
    df = _spp_day_constraints("2026-10-10")
    store.write_interval(df, iso="spp")
    store.write_interval(df, iso="spp")

    out = pd.read_parquet(store._constraints_dir_for_iso("spp") / "constraints_20261010.parquet")
    assert len(out) == 288
