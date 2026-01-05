from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime
import json


def write_excel(
    out_path,
    blocks_df,
    dec_a_data,
    dec_b_data,
    line_view_df,
    fam_assign_df,
    logic_timeline_df,
):
    notes = pd.DataFrame([
        {"Note": "Hard rule: all WOs in the same FAMILY are assigned to the same decorator (A or B)."},
        {"Note": "Families do not need to be adjacent, but should be if they can; this rule only enforces consistent assignment when selecting jobs."},
        {"Note": "Other rules are applied best-effort while honoring the family lock."},
    ])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        blocks_df.to_excel(xw, index=False, sheet_name="Blocks")
        dec_a_data.to_excel(xw, index=False, sheet_name="Decorator A")
        dec_b_data.to_excel(xw, index=False, sheet_name="Decorator B")
        line_view_df.to_excel(xw, index=False, sheet_name="Line View")
        logic_timeline_df.to_excel(xw, sheet_name="Logic Timeline", index=False)
        fam_assign_df.to_excel(xw, index=False, sheet_name="Family Assignment")
        notes.to_excel(xw, index=False, sheet_name="Notes")

def export_timeline_json(
    timeline_df: pd.DataFrame,
    out_json_path: str | Path,
    *,
    start_time: datetime | str,
) -> None:
    if timeline_df is None or timeline_df.empty:
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "start_time": str(start_time),
                "segment_count": 0,
            },
            "groups": [{"id": "A", "label": "Decorator A"}, {"id": "B", "label": "Decorator B"}],
            "items": []
        }
        Path(out_json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    t = timeline_df.copy()

    # Ensure JSON-serializable ISO timestamps
    t["START"] = pd.to_datetime(t["START"])
    t["FINISH"] = pd.to_datetime(t["FINISH"])

    groups = [
        {"id": "A", "label": "Decorator A"},
        {"id": "B", "label": "Decorator B"},
    ]

    items = []
    for i, r in t.reset_index(drop=True).iterrows():
        deco = str(r["DECORATOR"])
        if deco not in ("A", "B"):
            continue

        wo = str(r.get("WO", ""))
        role = str(r.get("ROLE_IN_BLOCK", ""))
        qty_run = int(r.get("QTY_RUN", 0) or 0)

        items.append({
            "id": f"{deco}-{i}",
            "group": deco,
            "start": r["START"].to_pydatetime().isoformat(),
            "end": r["FINISH"].to_pydatetime().isoformat(),
            "label": f"{wo} ({qty_run:,})",
            "data": {
                "wo": wo,
                "role": role,
                "qty_run": qty_run,
                "block": int(r.get("BLOCK", 0) or 0),
                "seq": None,  # timeline_df doesn't carry seq; OK for now
                "family": "",  # timeline_df doesn't carry family; OK for now
                "primary_color": "",
                "description": str(r.get("JOB_DESCRIPTION", "")),
                "req_date": (r.get("REQ_DATE").isoformat() if pd.notna(r.get("REQ_DATE")) and r.get("REQ_DATE") is not None else None),
            }
        })

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "start_time": (start_time.isoformat() if isinstance(start_time, datetime) else str(start_time)),
            "segment_count": len(items),
        },
        "groups": groups,
        "items": items,
    }

    Path(out_json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

