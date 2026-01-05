from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from collections import deque
import pandas as pd
from cleaning_utils import load_and_clean
from exporters import write_excel, export_timeline_json

@dataclass(frozen=True, slots=True)
class Job:
    wo: str
    qty: int
    color_rank: int
    family: str = "UNKNOWN"
    primary_color: str = "UNKNOWN"
    description: str = ""
    item_number: Optional[str] = None
    required_decorator: Optional[str] = None
    erp_seq: int = 0 
    req_date: datetime | None = None

@dataclass(frozen=True, slots=True)
class ScheduledJob:
    block: int
    decorator: str  
    seq: int
    role_in_block: str
    job: Job

class JobPool:
    def __init__(self, jobs: list[Job]):
        self._by_wo = {j.wo: j for j in jobs}

    def remove_wo(self, wo: str) -> None:
        self._by_wo.pop(wo, None)

    def values(self):
        return self._by_wo.values()

    def has_work(self) -> bool:
        return bool(self._by_wo)

class Line:
    def __init__(self, name, daily_capacity, decorator_names=("A", "B"), *, can_run_fn=None):
        self.name = name
        self.daily_capacity = daily_capacity
        self.decorators = {n: Decorator(n) for n in decorator_names}
        if len(self.decorators) != 2:
            raise ValueError("Line must have exactly 2 decorators")
        self._can_run_fn = can_run_fn or (lambda job: True)

    def can_run(self, job):
        return bool(self._can_run_fn(job))

    def decorator(self, name):
        return self.decorators[name]

    def other_decorator(self, name):
        return next(k for k in self.decorators if k != name)

class Decorator:
    def __init__(self, name):
        self.name = name
        self.jobs = []  # ordered

    def add(self, scheduled_job):
        self.jobs.append(scheduled_job)

class Scheduler:
    def __init__(self, jobs, line):
        self.pool = JobPool(jobs)
        self.line = line


        self.family_assignment = {}
        self.last_family = {"A": None, "B": None}
        self.last_color = {"A": None, "B": None}
        self.block_summaries = []

        self.block_number = 0
        self.preferred_anchor_decorator = "A"
        self.forced_next_anchor = None
        self.pinned = {"A": deque(), "B": deque()}
        self.item_assignment = {}          # item_number -> decorator
        self.last_item = {"A": None, "B": None}

    @property
    def decorator_a(self):
        return self.line.decorator("A")

    @property
    def decorator_b(self):
        return self.line.decorator("B")

    @classmethod
    def from_agg(cls, agg, line):
        jobs = []
        for r in agg.itertuples(index=False):
            val = getattr(r, "REQ_DATE", None)
            req_date = None if pd.isna(val) else val.to_pydatetime()
            jobs.append(
                Job(
                    wo=str(r.WO),
                    qty=int(r.QTY),
                    color_rank=int(r.COLOR_RANK),
                    family=str(r.FAMILY) if pd.notna(r.FAMILY) else "UNKNOWN",
                    primary_color=str(r.PRIMARY_COLOR) if pd.notna(r.PRIMARY_COLOR) else "UNKNOWN",
                    description=str(r.DESCRIPTION) if pd.notna(r.DESCRIPTION) else "",
                    item_number=str(r.ITEM_NUMBER) if pd.notna(r.ITEM_NUMBER) else None,
                    required_decorator=(None if pd.isna(getattr(r, "REQ_DECORATOR", None)) else str(r.REQ_DECORATOR)),
                    erp_seq=int(getattr(r, "SEQ", 0) or 0),
                    req_date=req_date,
                )
            )
        return cls(jobs, line)
    
    def is_item_eligible(self, job, decorator_name: str) -> bool:
        it = job.item_number
        if not it:
            return True
        return (it not in self.item_assignment) or (self.item_assignment[it] == decorator_name)

    def lock_item(self, item_number: Optional[str], decorator_name: str) -> None:
        if not item_number:
            return
        if item_number not in self.item_assignment:
            self.item_assignment[item_number] = decorator_name
    
    def pop_pinned(self, decorator_name):
        q = self.pinned.get(decorator_name)
        if q:
            return q.popleft()
        return None
    
    def pin_item_siblings(self, item_number: Optional[str], decorator_name: str) -> None:
        """
        Find remaining jobs with same ITEM_NUMBER and push them to the front of the same
        decorator's pinned queue so they schedule next (keeping items together across WOs).
        """
        if not item_number:
            return

        sibs = [j for j in self.pool.values() if j.item_number == item_number]
        if not sibs:
            return

        # Keep a sensible order; tweak as desired
        sibs.sort(key=lambda j: (-int(j.qty), int(getattr(j, "erp_seq", 0) or 0), j.wo))

        q = self.pinned[decorator_name]
        # push to the FRONT so they are consumed before other pinned jobs
        for j in reversed(sibs):
            q.appendleft(j)
            self.pool.remove_wo(j.wo)  # remove from pool since it's now pinned
    
    def seed_pinned_jobs(self, qty_cap_per_decorator=200_000):
        """
        Collect ERP-sequenced jobs into pinned queues (per decorator), but DO NOT schedule them.
        They will be consumed first as normal anchors/next-anchors in run().
        """
        for dec_name in ("A", "B"):
            sequenced = [
                j for j in self.pool.values()
                if j.required_decorator == dec_name and int(getattr(j, "erp_seq", 0)) > 0
            ]
            sequenced.sort(key=lambda j: (int(j.erp_seq), j.wo))

            cum = 0
            keep_wos = set()
            pinned_jobs = []

            for job in sequenced:
                if cum >= qty_cap_per_decorator:
                    break
                pinned_jobs.append(job)
                keep_wos.add(job.wo)
                cum += int(job.qty)

            # store pinned in order
            self.pinned[dec_name] = deque(pinned_jobs)

            # remove pinned from pool; strip seq constraint from overflow sequenced jobs
            for job in sequenced:
                if job.wo in keep_wos:
                    self.pool.remove_wo(job.wo)
                else:
                    # overflow: keep it but strip ERP requirement
                    self.pool._by_wo[job.wo] = replace(job, erp_seq=0, required_decorator=None)

    def decorator_object(self, decorator_name):
        return self.line.decorator(decorator_name)

    def is_eligible(self, job, decorator_name):
        if job.required_decorator in ("A", "B") and job.required_decorator != decorator_name:
            return False
        
        if not self.is_item_eligible(job, decorator_name):
            return False

        fam = job.family
        if fam == "UNKNOWN":
            return True
        return (fam not in self.family_assignment) or (self.family_assignment[fam] == decorator_name)

    def lock_family(self, family, decorator_name):
        if family == "UNKNOWN":
            return
        if family not in self.family_assignment:
            self.family_assignment[family] = decorator_name

    def remove_job(self, wo):
        self.pool.remove_wo(wo)

    def pick_largest_job(self, decorator_name):
        eligible = [j for j in self.pool.values() if self.is_eligible(j, decorator_name)]
        if not eligible:
            return None

        # Enforce anchor >= 60k when possible
        candidates = [j for j in eligible if int(j.qty) >= 60000]
        if not candidates:
            candidates = eligible

        current_fam = self.last_family.get(decorator_name)
        current_color = self.last_color.get(decorator_name)
        current_item = self.last_item.get(decorator_name)

        def sort_key(j):
            stay_family = int(bool(current_fam) and current_fam != "UNKNOWN" and j.family == current_fam)
            stay_color = int(bool(current_color) and current_color != "UNKNOWN" and j.primary_color == current_color)
            stay_item = int(bool(current_item) and j.item_number and j.item_number == current_item)

            # Sort order:
            # 1) stay_family desc (so matching family first)
            # 2) stay item number
            # 2) qty desc
            # 3) color_rank asc
            # 4) WO asc

            return (-stay_item, -stay_family, -stay_color, -j.qty, j.color_rank, j.wo)

        candidates.sort(key=sort_key)
        return candidates[0]

    def pick_small_jobs(self, decorator_name, count):
        if count <= 0:
            return []

        candidates = [j for j in self.pool.values() if j.qty < 60000 and self.is_eligible(j, decorator_name)]
        if not candidates:
            return []

        current_fam = self.last_family.get(decorator_name)
        current_color = self.last_color.get(decorator_name)
        current_item = self.last_item.get(decorator_name)

        def sort_key(j):
            stay_family = int(bool(current_fam) and current_fam != "UNKNOWN" and j.family == current_fam)
            stay_color = int(bool(current_color) and current_color != "UNKNOWN" and j.primary_color == current_color)
            stay_item = int(bool(current_item) and j.item_number and j.item_number == current_item)

            return (-stay_item, -stay_family, -stay_color, j.color_rank, j.qty, j.wo)


        candidates.sort(key=sort_key)
        return candidates[:count]

    def schedule_job(self, decorator_name, block_no, job, role_in_block):
        dec = self.decorator_object(decorator_name)
        max_seq = max((sj.seq for sj in dec.jobs), default=0)
        seq = max_seq + 1
        dec.add(ScheduledJob(
            block=block_no,
            decorator=decorator_name,
            seq=seq,
            role_in_block=role_in_block,
            job=job,
        ))
        self.last_family[decorator_name] = job.family
        self.last_color[decorator_name] = job.primary_color
        self.lock_item(job.item_number, decorator_name)
        self.last_item[decorator_name] = job.item_number
        self.pin_item_siblings(job.item_number, decorator_name)
    

    def run(self):
        self.seed_pinned_jobs(qty_cap_per_decorator=200_000)

        def has_work():
            return self.pool.has_work() or bool(self.pinned["A"]) or bool(self.pinned["B"])
    
        while has_work():
            self.block_number += 1

            # 1) Choose anchor
            if self.forced_next_anchor is not None:
                anchor_job = self.forced_next_anchor
                self.forced_next_anchor = None

                # Ensure preferred anchor decorator matches this job’s family lock if it exists
                if anchor_job.family != "UNKNOWN" and anchor_job.family in self.family_assignment:
                    self.preferred_anchor_decorator = self.family_assignment[anchor_job.family]
            else:
                # NEW: prefer pinned ERP job as the anchor if available for the preferred decorator
                anchor_job = self.pop_pinned(self.preferred_anchor_decorator)

                if anchor_job is None:
                    anchor_job = self.pick_largest_job(self.preferred_anchor_decorator)

                    if anchor_job is None:
                        # fallback to largest overall from remaining
                        if not self.pool.has_work():
                            # If no remaining jobs exist, but pinned exists on the other decorator,
                            # switch preference and pull from there.
                            other = self.line.other_decorator(self.preferred_anchor_decorator)
                            anchor_job = self.pop_pinned(other)
                            if anchor_job is None:
                                break
                            self.preferred_anchor_decorator = other
                        else:
                            largest_overall = sorted(
                                self.pool.values(),
                                key=lambda j: (-j.qty, j.color_rank, j.wo)
                            )[0]

                            if largest_overall.family != "UNKNOWN" and largest_overall.family in self.family_assignment:
                                self.preferred_anchor_decorator = self.family_assignment[largest_overall.family]
                                anchor_job = self.pick_largest_job(self.preferred_anchor_decorator)

                            if anchor_job is None:
                                anchor_job = largest_overall

            # Anchor is a normal scheduled job (no ERP_LOCKED)
            self.lock_family(anchor_job.family, self.preferred_anchor_decorator)
            self.lock_item(anchor_job.item_number, self.preferred_anchor_decorator)
            self.remove_job(anchor_job.wo)  # safe even if it’s not in remaining (pinned)
            self.schedule_job(self.preferred_anchor_decorator, self.block_number, anchor_job, "ANCHOR")

            other_decorator = self.line.other_decorator(self.preferred_anchor_decorator)

            # 2) Fill other decorator with small jobs (best effort)
            anchor_qty = int(anchor_job.qty)
            target_count_for_other = max(1, int(math.floor(anchor_qty / 60000.0)))
            small_jobs_needed = max(0, target_count_for_other - 1)
            pinned_small = []
            while self.pinned[other_decorator] and int(self.pinned[other_decorator][0].qty) < 60000:
                pinned_small.append(self.pop_pinned(other_decorator))

            for sj in pinned_small:
                self.lock_family(sj.family, other_decorator)
                self.lock_item(sj.item_number, other_decorator)
                self.schedule_job(other_decorator, self.block_number, sj, "SMALL")

            remaining_small_needed = max(0, small_jobs_needed - len(pinned_small))

            small_jobs = self.pick_small_jobs(other_decorator, remaining_small_needed)
            for sj in small_jobs:
                self.lock_family(sj.family, other_decorator)
                self.lock_item(sj.item_number, other_decorator)
                self.remove_job(sj.wo)
                self.schedule_job(other_decorator, self.block_number, sj, "SMALL")

            # 3) Next anchor for other decorator
            # NEW: prefer pinned ERP job as next anchor for the other decorator
            next_anchor_job = None
            if self.pinned[other_decorator] and int(self.pinned[other_decorator][0].qty) >= 60000:
                next_anchor_job = self.pop_pinned(other_decorator)
            else:
                next_anchor_job = self.pick_largest_job(other_decorator) if self.pool.has_work() else None

            if next_anchor_job is not None:
                self.lock_family(next_anchor_job.family, other_decorator)
                self.lock_item(next_anchor_job.item_number, other_decorator)
                self.remove_job(next_anchor_job.wo)
                self.schedule_job(other_decorator, self.block_number, next_anchor_job, "NEXT_ANCHOR")
                self.forced_next_anchor = next_anchor_job
            else:
                self.forced_next_anchor = None

            self.block_summaries.append({
                "BLOCK": self.block_number,
                "ANCHOR_DECORATOR": self.preferred_anchor_decorator,
                "ANCHOR_WO": anchor_job.wo,
                "ANCHOR_QTY": int(anchor_job.qty),
                "ANCHOR_FAMILY": anchor_job.family,
                "SMALL_USED": int(len(pinned_small) + len(small_jobs)),
                "NEXT_ANCHOR_WO": "" if next_anchor_job is None else next_anchor_job.wo,
                "NEXT_ANCHOR_QTY": "" if next_anchor_job is None else int(next_anchor_job.qty),
                "NEXT_ANCHOR_FAMILY": "" if next_anchor_job is None else next_anchor_job.family,
            })

            # Toggle preference
            self.preferred_anchor_decorator = other_decorator

        return self

    def scheduled_jobs_dataframe(self, decorator):
        return pd.DataFrame([{
            "BLOCK": sj.block,
            "DECORATOR": sj.decorator,
            "SEQ": sj.seq,
            "WO": sj.job.wo,
            "QTY": sj.job.qty,
            "PRIMARY_COLOR": sj.job.primary_color,
            "COLOR_RANK": sj.job.color_rank,
            "FAMILY": sj.job.family,
            "ROLE_IN_BLOCK": sj.role_in_block,
            "DESCRIPTION": sj.job.description,
            "ITEM_NUMBER": sj.job.item_number,
            "REQ_DATE": sj.job.req_date,
        } for sj in decorator.jobs])

    def results_to_dataframes(self):
        blocks_df = pd.DataFrame(self.block_summaries)
        decorator_a_df = self.scheduled_jobs_dataframe(self.decorator_a)
        decorator_b_df = self.scheduled_jobs_dataframe(self.decorator_b)
        family_assignment_df = pd.DataFrame(
            [{"FAMILY": fam, "DECORATOR": dec} for fam, dec in sorted(self.family_assignment.items())]
        )
        return blocks_df, decorator_a_df, decorator_b_df, family_assignment_df

def assign_jobs_to_lines(all_jobs, lines):
    buckets = {line.name: [] for line in lines}
    unassigned = []

    for job in all_jobs:
        placed = False
        for line in lines:
            if line.can_run(job):
                buckets[line.name].append(job)
                placed = True
                break
        if not placed:
            unassigned.append(job)

    return buckets, unassigned

@dataclass
class JobState:
    job: Job
    qty_done: int = 0

    @property
    def qty_total(self) -> int:
        return int(self.job.qty)

    @property
    def qty_remaining(self) -> int:
        return max(0, self.qty_total - self.qty_done)

    @property
    def is_done(self) -> bool:
        return self.qty_done >= self.qty_total

def build_timeline_from_objects(
    dec_a_jobs: list[ScheduledJob],
    dec_b_jobs: list[ScheduledJob],
    start_time: "datetime | str",
    *,
    rate_cph: float = 29_000.0,
    setup_equiv_qty: int = 60_000,
    anchor_slice_qty: int = 60_000,
):
    if isinstance(start_time, str):
        cur = pd.to_datetime(start_time).to_pydatetime()
    elif isinstance(start_time, datetime):
        cur = start_time
    else:
        raise TypeError("start_time must be datetime or str")

    if rate_cph <= 0:
        raise ValueError("rate_cph must be > 0")
    if setup_equiv_qty <= 0:
        raise ValueError("setup_equiv_qty must be > 0")
    if anchor_slice_qty <= 0:
        raise ValueError("anchor_slice_qty must be > 0")

    def hours_for_qty(qty: int) -> float:
        return float(qty) / float(rate_cph)

    def dur_for_qty(qty: int) -> timedelta:
        return timedelta(seconds=hours_for_qty(qty) * 3600.0)

    # ---------- Normalize to "block queues" ----------
    # Keep only roles your simulator actually runs in-block
    RUNNABLE_ROLES = {"ANCHOR", "SMALL"}

    a_run = [sj for sj in dec_a_jobs if sj.role_in_block in RUNNABLE_ROLES]
    b_run = [sj for sj in dec_b_jobs if sj.role_in_block in RUNNABLE_ROLES]

    # Sort stable by block then seq
    a_run.sort(key=lambda sj: (int(sj.block), int(sj.seq)))
    b_run.sort(key=lambda sj: (int(sj.block), int(sj.seq)))

    all_blocks = sorted({sj.block for sj in a_run} | {sj.block for sj in b_run})

    # Identify the first "computed" block (first non-zero block)
    nonzero_blocks = [int(b) for b in all_blocks if int(b) != 0]
 

    # Per-WO planned start/finish
    wo_start: Dict[str, datetime] = {}
    wo_finish: Dict[str, datetime] = {}

    timeline_rows: List[Dict[str, Any]] = []

    def append_segment(block_no, deco, role, wo, qty_run, description, req_date):
        nonlocal cur
        if qty_run <= 0:
            return
        seg_start = cur
        seg_end = cur + dur_for_qty(qty_run)

        timeline_rows.append({
            "BLOCK": int(block_no),
            "DECORATOR": str(deco),
            "ROLE_IN_BLOCK": str(role),
            "WO": str(wo),
            "QTY_RUN": int(qty_run),
            "START": seg_start,
            "FINISH": seg_end,
            "DURATION_HOURS": hours_for_qty(qty_run),
            "JOB_DESCRIPTION": description,
            "REQ_DATE": req_date,
        })

        if wo not in wo_start:
            wo_start[wo] = seg_start
        wo_finish[wo] = seg_end
        cur = seg_end

    # Helper: scheduled jobs in a block for a decorator + role
    def block_jobs(jobs: list[ScheduledJob], block_no: int, role: str) -> list["JobState"]:
        blk = [sj for sj in jobs if sj.block == block_no and sj.role_in_block == role]
        blk.sort(key=lambda sj: int(sj.seq))
        return [JobState(job=sj.job) for sj in blk]

    

    # For annotating the schedule “rows” (ScheduledJob -> df row)
    def scheduled_jobs_df(deco_name: str, jobs: list[ScheduledJob]) -> pd.DataFrame:
        return pd.DataFrame([{
            "BLOCK": sj.block,
            "DECORATOR": deco_name,
            "SEQ": sj.seq,
            "WO": sj.job.wo,
            "QTY": sj.job.qty,
            "PRIMARY_COLOR": sj.job.primary_color,
            "COLOR_RANK": sj.job.color_rank,
            "FAMILY": sj.job.family,
            "ROLE_IN_BLOCK": sj.role_in_block,
            "DESCRIPTION": sj.job.description,
            "ITEM_NUMBER": sj.job.item_number,
            "REQ_DATE": sj.job.req_date,
        } for sj in jobs])


    # ---------- Simulate block-by-block ----------
    for block in all_blocks:
        if int(block) == 0:
            # block 0 already simulated above (ERP_LOCKED)
            continue

        # Find the anchor in this block
        a_anchor = [sj for sj in a_run if sj.block == block and sj.role_in_block == "ANCHOR"]
        b_anchor = [sj for sj in b_run if sj.block == block and sj.role_in_block == "ANCHOR"]

        if len(a_anchor) + len(b_anchor) == 0:
            continue

        # If weirdly multiple anchors exist, choose earliest by seq
        if len(a_anchor) + len(b_anchor) == 1:
            anchor_sj = a_anchor[0] if len(a_anchor) == 1 else b_anchor[0]
        else:
            anchors = (a_anchor + b_anchor)
            anchors.sort(key=lambda sj: int(sj.seq))
            anchor_sj = anchors[0]

        anchor_deco = str(anchor_sj.decorator)  # "A" or "B"
        other_deco = "B" if anchor_deco == "A" else "A"

        anchor_job = JobState(job=anchor_sj.job)

        small_queue = (
            block_jobs(a_run, block, "SMALL") if other_deco == "A"
            else block_jobs(b_run, block, "SMALL")
        )

        setup_progress = {"A": setup_equiv_qty, "B": setup_equiv_qty}
        small_idx = 0

        producing = anchor_deco

        def other(d: str) -> str:
            return "B" if d == "A" else "A"

        def next_job_ready(d: str) -> bool:
            if d == anchor_deco:
                return (not anchor_job.is_done) and (setup_progress[d] >= setup_equiv_qty)
            return (small_idx < len(small_queue)) and (setup_progress[d] >= setup_equiv_qty)

        def run_qty_for(d):
            if d == anchor_deco:
                qty = min(anchor_slice_qty, anchor_job.qty_remaining)
                j = anchor_job.job
                return j.wo, "ANCHOR", qty, j.description, j.req_date

            js = small_queue[small_idx]
            j = js.job
            return j.wo, "SMALL", js.qty_remaining, j.description, j.req_date

        def mark_run(d: str, qty: int):
            nonlocal small_idx
            if d == anchor_deco:
                anchor_job.qty_done += qty
            else:
                small_queue[small_idx].qty_done += qty
                if small_queue[small_idx].is_done:
                    small_idx += 1
                    setup_progress[d] = 0  # next small needs setup from scratch
        
        while (not anchor_job.is_done) or (small_idx < len(small_queue)):
            if not next_job_ready(producing):
                if next_job_ready(other(producing)):
                    producing = other(producing)
                else:
                    setup_progress[producing] = setup_equiv_qty

            if not next_job_ready(producing):
                break

            wo, role, qty_plan, desc, req_date = run_qty_for(producing)

            setup_progress[other(producing)] = min(
                setup_equiv_qty,
                setup_progress[other(producing)] + qty_plan
            )

            append_segment(block, producing, role, wo, qty_plan, desc,req_date)
            mark_run(producing, qty_plan)

            if next_job_ready(other(producing)):
                producing = other(producing)

    timeline_df = pd.DataFrame(timeline_rows)

    # ---------- Annotate schedule rows with planned WO start/finish ----------
    def annotate_schedule(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["PLANNED_START"] = out["WO"].map(wo_start)
        out["PLANNED_FINISH"] = out["WO"].map(wo_finish)
        out["PLANNED_HOURS"] = (
            (pd.to_datetime(out["PLANNED_FINISH"]) - pd.to_datetime(out["PLANNED_START"]))
            .dt.total_seconds()
            .fillna(0.0) / 3600.0
        )
        return out

    a_df = scheduled_jobs_df("A", dec_a_jobs)
    b_df = scheduled_jobs_df("B", dec_b_jobs)

    return annotate_schedule(a_df), annotate_schedule(b_df), timeline_df

def make_logic_timeline_sheet(timeline_df: pd.DataFrame) -> pd.DataFrame:
    if timeline_df is None or timeline_df.empty:
        return pd.DataFrame(columns=["TimeStart", "DecoratorRunning", "Description", "Qty"])

    t = timeline_df.sort_values("START", kind="mergesort").reset_index(drop=True)

    return pd.DataFrame({
        "TimeStart": pd.to_datetime(t["START"]),
        "DecoratorRunning": t["DECORATOR"].astype(str),
        "Description": t["JOB_DESCRIPTION"].astype(str),
        "Qty": pd.to_numeric(t["QTY_RUN"], errors="coerce").fillna(0).astype(int),
    })

def build_schedule(data, line):
    sched = Scheduler.from_agg(data, line).run()
    return sched.results_to_dataframes()


def build_line_view_sheet(a_data, b_data):
    cols = [
    "INDEX", "BLOCK",
    "DecoA_WO", "A_ITEM_NUMBER", "A_QTY", "A_FAMILY", "A_PRIMARY_COLOR", "A_DESCRIPTION",
    "DecoB_WO", "B_ITEM_NUMBER", "B_QTY", "B_FAMILY", "B_PRIMARY_COLOR", "B_DESCRIPTION",
    ]

    if a_data.empty and b_data.empty:
        return pd.DataFrame(columns=cols)

    a_sorted = a_data.sort_values(["BLOCK", "SEQ"], kind="mergesort") if not a_data.empty else a_data
    b_sorted = b_data.sort_values(["BLOCK", "SEQ"], kind="mergesort") if not b_data.empty else b_data

    all_blocks = sorted(set(a_sorted["BLOCK"]).union(b_sorted["BLOCK"]))

    rows = []
    row_index = 0

    for block in all_blocks:
        a_rows = a_sorted[a_sorted["BLOCK"] == block][
            ["WO", "ITEM_NUMBER", "QTY", "FAMILY", "PRIMARY_COLOR", "DESCRIPTION"]
        ].to_records(index=False).tolist()

        b_rows = b_sorted[b_sorted["BLOCK"] == block][
            ["WO", "ITEM_NUMBER", "QTY", "FAMILY", "PRIMARY_COLOR", "DESCRIPTION"]
        ].to_records(index=False).tolist()

        row_count = max(len(a_rows), len(b_rows), 1)

        repeat_a_row = (len(a_rows) == 1 and row_count > 1)
        repeat_b_row = (len(b_rows) == 1 and row_count > 1)

        for i in range(row_count):
            row_index += 1

            if not a_rows:
                a_wo = a_item = a_qty = a_family = a_color = a_desc = ""
            elif repeat_a_row:
                a_wo, a_item, a_qty, a_family, a_color, a_desc = a_rows[0]
            elif i < len(a_rows):
                a_wo, a_item, a_qty, a_family, a_color, a_desc = a_rows[i]
            else:
                a_wo = a_item = a_qty = a_family = a_color = a_desc = ""

            # B side
            if not b_rows:
                b_wo = b_item = b_qty = b_family = b_color = b_desc = ""
            elif repeat_b_row:
                b_wo, b_item, b_qty, b_family, b_color, b_desc = b_rows[0]
            elif i < len(b_rows):
                b_wo, b_item, b_qty, b_family, b_color, b_desc = b_rows[i]
            else:
                b_wo = b_item = b_qty = b_family = b_color = b_desc = ""

            rows.append({
                "INDEX": row_index,
                "BLOCK": block,
                "DecoA_WO": a_wo,
                "A_ITEM_NUMBER": a_item,
                "A_QTY": a_qty,
                "A_FAMILY": a_family if a_family != "UNKNOWN" else "",
                "A_PRIMARY_COLOR": a_color,
                "A_DESCRIPTION": a_desc,
                "DecoB_WO": b_wo,
                "B_ITEM_NUMBER": b_item,
                "B_QTY": b_qty,
                "B_FAMILY": b_family if b_family != "UNKNOWN" else "",
                "B_PRIMARY_COLOR": b_color,
                "B_DESCRIPTION": b_desc,
            })

    return pd.DataFrame(rows, columns=cols)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_xlsx", help="Input Excel file path")
    ap.add_argument("-o", "--output", default="Two_Decorator_Schedule_FamilyLocked.xlsx", help="Output Excel filename")
    args = ap.parse_args()

    in_path = Path(args.input_xlsx).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    start_time = "2025-12-30 06:00"
    # Load the data from the input Excel and make sure all columns are present
    data = load_and_clean(in_path)
    #Create line object
    line_4 = Line(name="LINE_4", daily_capacity=0, decorator_names=("A", "B"))
    # Build the schedule
    blocks_data, deco_a_data, deco_b_data, family_data = build_schedule(data, line_4)

    deco_a_data, deco_b_data, timeline_df = build_timeline_from_objects(
    line_4.decorator("A").jobs,
    line_4.decorator("B").jobs,
    start_time=start_time,
)
    export_timeline_json(timeline_df, out_path.with_suffix(".timeline.json"), start_time=start_time)
    logic_timeline_df = make_logic_timeline_sheet(timeline_df)
    # Build the line view sheet
    line_view_data = build_line_view_sheet(deco_a_data, deco_b_data)

    write_excel(out_path, blocks_data, deco_a_data, deco_b_data, line_view_data, family_data, logic_timeline_df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()