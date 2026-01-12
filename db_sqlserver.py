# db_sqlserver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
import pyodbc

@dataclass(frozen=True)
class SqlServerConfig:
    driver: str = "ODBC Driver 17 for SQL Server"
    server: str = "dscsqc"
    database: str = "SchedulePlanning"
    trusted_connection: bool = True

    def conn_str(self) -> str:
        parts = [
            f"DRIVER={{{self.driver}}};",
            f"SERVER={self.server};",
            f"DATABASE={self.database};",
        ]
        if self.trusted_connection:
            parts.append("Trusted_Connection=yes;")
        return "".join(parts)


def df_to_tuples(df: pd.DataFrame, cols: List[str]):
    out = df[cols].copy()
    out = out.where(pd.notna(out), None)
    return list(map(tuple, out.to_numpy()))


def input_df_from_cleaned(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    expected = {
        "WO": None,
        "QTY": 0,
        "COLOR_RANK": 0,
        "FAMILY": None,
        "PRIMARY_COLOR": None,
        "DESCRIPTION": None,
        "ITEM_NUMBER": None,
        "REQ_DECORATOR": None,
        "SEQ": 0,
        "REQ_DATE": None,
        "CAN_SIZE": None,
        "LINE": None,
    }
    for c, default in expected.items():
        if c not in df.columns:
            df[c] = default

    out = pd.DataFrame({
        "wo": df["WO"].astype(str),
        "qty": pd.to_numeric(df["QTY"], errors="coerce").fillna(0).astype(int),
        "color_rank": pd.to_numeric(df["COLOR_RANK"], errors="coerce").fillna(0).astype(int),
        "family": df["FAMILY"].where(pd.notna(df["FAMILY"]), None).astype(object),
        "primary_color": df["PRIMARY_COLOR"].where(pd.notna(df["PRIMARY_COLOR"]), None).astype(object),
        "description": df["DESCRIPTION"].where(pd.notna(df["DESCRIPTION"]), None).astype(object),
        "item_number": df["ITEM_NUMBER"].where(pd.notna(df["ITEM_NUMBER"]), None).astype(object),
        "req_decorator": df["REQ_DECORATOR"].where(pd.notna(df["REQ_DECORATOR"]), None).astype(object),
        "erp_seq": pd.to_numeric(df["SEQ"], errors="coerce").fillna(0).astype(int),
        "req_date": pd.to_datetime(df["REQ_DATE"], errors="coerce"),
        "can_size": df["CAN_SIZE"].where(pd.notna(df["CAN_SIZE"]), None).astype(object),
        "erp_line": df["LINE"].where(pd.notna(df["LINE"]), None).astype(object),
    })

    out["req_date"] = out["req_date"].where(out["req_date"].notna(), None)
    return out


def dataframes_from_plant(plant: "PlantSchedule") -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_sjs = plant.all()

    schedule_df = pd.DataFrame([{
        "wo": str(sj.job.wo),
        "line": str(sj.line),
        "decorator": str(sj.decorator),
        "seq": int(sj.seq),
        "block": int(sj.block),
        "role_in_block": str(sj.role_in_block),
    } for sj in all_sjs])

    work_orders_df = pd.DataFrame([{
        "wo": str(sj.job.wo),
        "qty": int(sj.job.qty),
        "family": (None if (sj.job.family is None or str(sj.job.family) == "nan") else str(sj.job.family)),
        "primary_color": (None if (sj.job.primary_color is None or str(sj.job.primary_color) == "nan") else str(sj.job.primary_color)),
        "description": (None if (sj.job.description is None or str(sj.job.description) == "nan") else str(sj.job.description)),
        "item_number": (None if (sj.job.item_number is None or str(sj.job.item_number) == "nan") else str(sj.job.item_number)),
        "req_date": sj.job.req_date,
        "can_size": (None if (sj.job.can_size is None or str(sj.job.can_size) == "nan") else str(sj.job.can_size)),
    } for sj in all_sjs])

    if not work_orders_df.empty:
        work_orders_df = work_orders_df.drop_duplicates(subset=["wo"], keep="first").reset_index(drop=True)

    for col in ["family", "primary_color", "description", "item_number", "can_size"]:
        if col in work_orders_df.columns:
            work_orders_df[col] = work_orders_df[col].where(pd.notna(work_orders_df[col]), None)

    if "req_date" in work_orders_df.columns:
        work_orders_df["req_date"] = pd.to_datetime(work_orders_df["req_date"], errors="coerce")
        work_orders_df["req_date"] = work_orders_df["req_date"].where(work_orders_df["req_date"].notna(), None)

    return work_orders_df, schedule_df


def write_input_to_sqlserver(conn_str: str, input_df: pd.DataFrame) -> None:
    cols = [
        "wo", "qty", "color_rank", "family", "primary_color", "description",
        "item_number", "req_decorator", "erp_seq", "req_date", "can_size", "erp_line",
    ]
    rows = df_to_tuples(input_df, cols)

    insert_sql = """
        INSERT INTO dbo.input_work_orders
        (wo, qty, color_rank, family, primary_color, description, item_number, req_decorator, erp_seq, req_date, can_size, erp_line)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    with pyodbc.connect(conn_str) as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM dbo.input_work_orders;")
            cur.fast_executemany = True
            if rows:
                cur.executemany(insert_sql, rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def write_schedule_to_sqlserver(conn_str: str, work_orders_df: pd.DataFrame, schedule_df: pd.DataFrame) -> None:
    wo_cols = ["wo", "qty", "family", "primary_color", "description", "item_number", "req_date", "can_size"]
    sch_cols = ["wo", "line", "decorator", "seq", "block", "role_in_block"]

    wo_rows = df_to_tuples(work_orders_df, wo_cols)
    sch_rows = df_to_tuples(schedule_df, sch_cols)

    insert_work_orders_sql = """
        INSERT INTO dbo.work_orders
        (wo, qty, family, primary_color, description, item_number, req_date, can_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """

    insert_schedule_sql = """
        INSERT INTO dbo.schedule
        (wo, line, decorator, seq, block, role_in_block)
        VALUES (?, ?, ?, ?, ?, ?);
    """

    with pyodbc.connect(conn_str) as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM dbo.schedule;")
            cur.execute("DELETE FROM dbo.work_orders;")

            cur.fast_executemany = True
            if wo_rows:
                cur.executemany(insert_work_orders_sql, wo_rows)
            if sch_rows:
                cur.executemany(insert_schedule_sql, sch_rows)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
