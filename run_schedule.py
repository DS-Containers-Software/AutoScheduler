from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyodbc

from cleaning_utils import load_and_clean
from exporters import export_timeline_json
from db_sqlserver import (
    input_df_from_cleaned,
    write_input_to_sqlserver,
    dataframes_from_plant,
    write_schedule_to_sqlserver,
)

from scheduling_core import (
    CalendarCursor,
    HourlyCalendarClock,
    Line,
    PlantSchedule,
    Scheduler,
    assign_jobs_to_lines_balanced,
    build_timeline_from_objects,
    item_prefix3,
    jobs_from_rows,
)


def read_line_hourly_status(
    conn_str: str,
    line: str,
    start_dt,
    end_dt,
) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_dt).to_pydatetime()
    end_dt = pd.to_datetime(end_dt).to_pydatetime()

    sql = """
    SELECT Line, HourStart, IsDown, IsRunning
    FROM dbo.vLineHourlyStatus
    WHERE Line = ?
      AND HourStart >= ?
      AND HourStart < ?
    ORDER BY HourStart;
    """

    with pyodbc.connect(conn_str) as cn:
        cur = cn.cursor()
        cur.execute(sql, (line, start_dt, end_dt))
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]

    df = pd.DataFrame.from_records(rows, columns=cols)

    if not df.empty:
        df["HourStart"] = pd.to_datetime(df["HourStart"])
        df["IsDown"] = df["IsDown"].astype(int)
        df["IsRunning"] = df["IsRunning"].astype(int)

    return df


def build_calendar_df(conn_str: str, line_name: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    cal_df = read_line_hourly_status(conn_str, line_name, start_dt, end_dt)
    cal_df = cal_df[["HourStart", "IsDown"]].copy()
    cal_df["HourStart"] = pd.to_datetime(cal_df["HourStart"])
    cal_df = cal_df.sort_values("HourStart").reset_index(drop=True)
    return cal_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_xlsx", help="Input Excel file path")
    args = ap.parse_args()

    in_path = Path(args.input_xlsx).expanduser().resolve()

    start_time_str = "2026-01-11 06:00"
    start_dt = pd.to_datetime(start_time_str)
    end_dt = start_dt + pd.Timedelta(days=21)

    data = load_and_clean(in_path)
    input_df = input_df_from_cleaned(data)

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=dscsqc;"
        "DATABASE=SchedulePlanning;"
        "Trusted_Connection=yes;"
    )

    # 1) Persist input snapshot
    write_input_to_sqlserver(conn_str, input_df)

    # 2) Convert rows -> Job objects
    all_jobs = jobs_from_rows(data.itertuples(index=False))

    # 3) Define lines (execution config lives here)
    lines = [
        Line(
            name="LINE4",
            daily_capacity=0,
            decorator_names=("A", "B"),
            can_run_fn=lambda j: item_prefix3(j.item_number) in {"710", "205"},
        )
    ]

    # 4) Assign jobs to lines
    buckets, unassigned = assign_jobs_to_lines_balanced(all_jobs, lines)
    if unassigned:
        # Optional: log / persist these somewhere
        pass

    # 5) Run scheduling per line (each line gets its own clock & schedule entries)
    plant = PlantSchedule()

    for ln in lines:
        cal_df = build_calendar_df(conn_str, ln.name, start_dt, end_dt)

        clock = HourlyCalendarClock(
            rate_cph=29_000.0,
            cal=CalendarCursor(calendar_df=cal_df, idx=0),
        )

        Scheduler(
            buckets[ln.name],
            ln,
            plant,
            clock=clock,
            start_time=start_dt.to_pydatetime(),
        ).run()

    # 6) Normalize + validate
    plant.normalize_sequences()
    plant.validate()

    # 7) Persist plant schedule tables
    work_orders_df, schedule_df = dataframes_from_plant(plant)
    write_schedule_to_sqlserver(conn_str, work_orders_df, schedule_df)

    # 8) Export per-line timeline JSON (re-simulate timeline for reporting)
    for ln in lines:
        a = plant.by_resource(ln.name, "A")
        b = plant.by_resource(ln.name, "B")

        cal_df = build_calendar_df(conn_str, ln.name, start_dt, end_dt)

        clock = HourlyCalendarClock(
            rate_cph=29_000.0,
            cal=CalendarCursor(calendar_df=cal_df, idx=0),
        )

        a_df, b_df, timeline_df = build_timeline_from_objects(
            a, b,
            start_time=start_time_str,
            clock=clock,
            setup_equiv_qty=60_000,
            anchor_slice_qty=60_000,
        )

        export_timeline_json(
            timeline_df,
            Path(f"output.{ln.name}.timeline.json"),
            start_time=start_time_str,
            deco_a_df=a_df,
            deco_b_df=b_df,
        )


if __name__ == "__main__":
    main()
