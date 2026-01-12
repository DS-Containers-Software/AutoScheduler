from __future__ import annotations

from datetime import datetime, timedelta
import json
import argparse
import math
from pathlib import Path
from collections import deque, defaultdict
import pandas as pd
from cleaning_utils import load_and_clean
from exporters import export_timeline_json
import re
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple, Callable, Set, Any
import pyodbc


ResourceKey = Tuple[str, str]  # (line, decorator)
class ScheduleError(Exception):pass
class DuplicateWOError(ScheduleError):pass
class LockedJobError(ScheduleError):pass
class ValidationError(ScheduleError):pass


@dataclass(frozen=True, slots=True)
class JobLock:
    """
    Lock rules for a WO.

    - required_resource: WO must stay on that (line, decorator)
    - fixed_seq: WO must stay at that seq (within its decorator)
    - frozen: disallow remove/move unless explicitly overridden/unlocked
    """
    required_resource: Optional[ResourceKey] = None
    fixed_seq: Optional[int] = None
    frozen: bool = True
    reason: str = ""


@dataclass
class PlantSchedule:
    """
    Authoritative container for ScheduledJob across the whole plant.

    Key invariant (no WO splitting):
      - each WO appears at most once across the entire schedule
    """
    _by_wo: Dict[str, "ScheduledJob"] = field(default_factory=dict)
    _locks: Dict[str, JobLock] = field(default_factory=dict)

    # -------------------------
    # Locks
    # -------------------------

    def lock(self, wo: str, lock: JobLock) -> None:
        self._locks[str(wo)] = lock

    def unlock(self, wo: str) -> None:
        self._locks.pop(str(wo), None)

    def lock_for(self, wo: str) -> Optional[JobLock]:
        return self._locks.get(str(wo))

    def is_frozen(self, wo: str) -> bool:
        lk = self._locks.get(str(wo))
        return bool(lk and lk.frozen)

    # -------------------------
    # Core CRUD
    # -------------------------

    def add(self, sj: "ScheduledJob") -> None:
        wo = str(sj.job.wo)
        if wo in self._by_wo:
            existing = self._by_wo[wo]
            raise DuplicateWOError(
                f"WO {wo} already scheduled on {existing.line}/{existing.decorator} "
                f"(trying {sj.line}/{sj.decorator})."
            )
        self._enforce_lock(sj)
        self._by_wo[wo] = sj

    def upsert(self, sj: "ScheduledJob", *, force: bool = False) -> None:
        """
        Replace or insert. Use this for manual edits.
        If the WO is frozen, you must pass force=True (or unlock first).
        """
        wo = str(sj.job.wo)
        if self.is_frozen(wo) and not force:
            raise LockedJobError(f"WO {wo} is frozen; unlock or use force=True.")
        self._enforce_lock(sj)
        self._by_wo[wo] = sj

    def remove(self, wo: str, *, force: bool = False) -> None:
        wo = str(wo)
        if self.is_frozen(wo) and not force:
            raise LockedJobError(f"WO {wo} is frozen; unlock or use force=True.")
        self._by_wo.pop(wo, None)

    def get(self, wo: str) -> Optional["ScheduledJob"]:
        return self._by_wo.get(str(wo))

    def all(self) -> List["ScheduledJob"]:
        return list(self._by_wo.values())

    # -------------------------
    # Views
    # -------------------------

    def resources(self) -> Set[ResourceKey]:
        return {(sj.line, sj.decorator) for sj in self._by_wo.values()}

    def by_line(self, line: str) -> List["ScheduledJob"]:
        out = [sj for sj in self._by_wo.values() if sj.line == line]
        out.sort(key=lambda sj: (sj.decorator, int(sj.seq)))
        return out

    def by_resource(self, line: str, decorator: str) -> List["ScheduledJob"]:
        out = [sj for sj in self._by_wo.values() if sj.line == line and sj.decorator == decorator]
        out.sort(key=lambda sj: int(sj.seq))
        return out

    def validate(
        self,
        *,
        can_run: Optional[Callable[["ScheduledJob"], bool]] = None,
        require_unique_seq: bool = True,
    ) -> None:
        # Validate lock compliance for all scheduled jobs
        for sj in self._by_wo.values():
            self._enforce_lock(sj)

        # Feasibility hook
        if can_run is not None:
            bad = [sj for sj in self._by_wo.values() if not can_run(sj)]
            if bad:
                sample = ", ".join(f"{x.job.wo}@{x.line}/{x.decorator}" for x in bad[:10])
                raise ValidationError(f"Infeasible jobs in schedule (sample): {sample}")

        # Seq uniqueness per resource
        if require_unique_seq:
            seen: Dict[ResourceKey, Set[int]] = {}
            for sj in self._by_wo.values():
                rk = (sj.line, sj.decorator)
                seen.setdefault(rk, set())
                s = int(sj.seq)
                if s in seen[rk]:
                    raise ValidationError(f"Duplicate seq {s} on {rk} (WO {sj.job.wo}).")
                seen[rk].add(s)

    def normalize_sequences(self) -> None:
        """
        Renumber seq per (line, decorator) to 1..N in current seq order.

        Respects fixed_seq locks:
          - jobs with fixed_seq keep that number
          - other jobs are packed around them
          - raises if fixed_seq collisions occur
        """
        grouped: Dict[ResourceKey, List["ScheduledJob"]] = {}
        for sj in self._by_wo.values():
            grouped.setdefault((sj.line, sj.decorator), []).append(sj)

        new_by_wo = dict(self._by_wo)

        for rk, lst in grouped.items():
            lst.sort(key=lambda sj: int(sj.seq))
            fixed_map: Dict[int, "ScheduledJob"] = {}
            movable: List["ScheduledJob"] = []

            for sj in lst:
                lk = self._locks.get(str(sj.job.wo))
                if lk and lk.fixed_seq is not None:
                    fs = int(lk.fixed_seq)
                    if fs in fixed_map and str(fixed_map[fs].job.wo) != str(sj.job.wo):
                        raise LockedJobError(f"fixed_seq collision on {rk} at seq={fs}.")
                    fixed_map[fs] = sj
                else:
                    movable.append(sj)

            # Assign sequences 1..N, preserving fixed slots
            i = 1
            m_idx = 0
            total = len(lst)
            while i <= total:
                if i in fixed_map:
                    sj = fixed_map[i]
                    if int(sj.seq) != i:
                        new_by_wo[str(sj.job.wo)] = replace(sj, seq=i)
                else:
                    sj = movable[m_idx]
                    m_idx += 1
                    if int(sj.seq) != i:
                        new_by_wo[str(sj.job.wo)] = replace(sj, seq=i)
                i += 1

        self._by_wo = new_by_wo

    # -------------------------
    # Internal
    # -------------------------

    def _enforce_lock(self, sj: "ScheduledJob") -> None:
        wo = str(sj.job.wo)
        lk = self._locks.get(wo)
        if not lk:
            return

        if lk.required_resource is not None:
            req_line, req_dec = lk.required_resource
            if (sj.line, sj.decorator) != (req_line, req_dec):
                raise LockedJobError(
                    f"WO {wo} locked to {req_line}/{req_dec}, got {sj.line}/{sj.decorator}."
                )

        if lk.fixed_seq is not None:
            if int(sj.seq) != int(lk.fixed_seq):
                raise LockedJobError(
                    f"WO {wo} locked to seq {lk.fixed_seq}, got seq {sj.seq}."
                )

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
    can_size: str = ""
    erp_line: Optional[str] = None

@dataclass(frozen=True, slots=True)
class ScheduledJob:
    line: str
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
    def __init__(self, jobs, line, plant: PlantSchedule):
        self.pool = JobPool(jobs)
        self.line = line
        self.plant = plant


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
    def from_agg(cls, agg, line,plant: PlantSchedule):
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
                    can_size=str(getattr(r, "CAN_SIZE", "") or ""),
                    erp_line=(None if pd.isna(getattr(r, "LINE", None)) else str(getattr(r, "LINE"))),
                )
            )
        return cls(jobs, line, plant)
    
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
            target = norm_line(self.line.name)
            bad = [
            (j.wo, j.erp_line)
            for j in self.pool.values()
            if j.required_decorator == dec_name and int(getattr(j, "erp_seq", 0)) > 0
            and norm_line(j.erp_line) != target
                ]
            good = [j.wo for j in self.pool.values()
                if j.required_decorator == dec_name and int(getattr(j, "erp_seq", 0)) > 0
                and norm_line(j.erp_line) == target
                ]
            print("Pinned mismatch sample:", bad[:20])
            print(f"Pinned good sample for decorator {dec_name}:", good[:20])
            sequenced = [
                    j for j in self.pool.values()
                    if j.required_decorator == dec_name
                    and int(getattr(j, "erp_seq", 0)) > 0
                    and norm_line(j.erp_line) == norm_line(self.line.name)
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
        eligible = [
        j for j in self.pool.values() if self.line.can_run(j) and self.is_eligible(j, decorator_name)
    ]
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

        candidates = [j for j in self.pool.values() if j.qty < 60000 and self.line.can_run(j) and self.is_eligible(j, decorator_name)]
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
        existing = self.plant.by_resource(self.line.name, decorator_name)
        max_seq = max((int(sj.seq) for sj in existing), default=0)
        seq = max_seq + 1
        sj = ScheduledJob(
            line=self.line.name,
            block=block_no,
            decorator=decorator_name,
            seq=seq,
            role_in_block=role_in_block,
            job=job,
        )
        self.plant.add(sj)
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
                # Reserve it for next loop iteration, but DON'T schedule it yet
                self.lock_family(next_anchor_job.family, other_decorator)
                self.lock_item(next_anchor_job.item_number, other_decorator)
                self.remove_job(next_anchor_job.wo)   # safe even if it came from pinned
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

    def scheduled_jobs_dataframe(self, decorator_name: str):
        jobs = self.plant.by_resource(self.line.name, decorator_name)
        return pd.DataFrame([{
            "LINE": sj.line,
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
            "CAN_SIZE": sj.job.can_size,
        } for sj in jobs])

    def results_to_dataframes(self):
        blocks_df = pd.DataFrame(self.block_summaries)
        decorator_a_df = self.scheduled_jobs_dataframe("A")
        decorator_b_df = self.scheduled_jobs_dataframe("B")
        family_assignment_df = pd.DataFrame(
            [{"FAMILY": fam, "DECORATOR": dec} for fam, dec in sorted(self.family_assignment.items())]
        )
        return blocks_df, decorator_a_df, decorator_b_df, family_assignment_df
    
def norm_line(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    # keep only letters+digits so "LINE 4" == "LINE4"
    return re.sub(r"[^A-Z0-9]+", "", s)

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
        if out.empty or "WO" not in out.columns:
            out["PLANNED_START"] = pd.NaT
            out["PLANNED_FINISH"] = pd.NaT
            out["PLANNED_HOURS"] = 0.0
            return out
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


def build_schedule(data, line, plant):
    sched = Scheduler.from_agg(data, line, plant).run()
    return sched.results_to_dataframes()


def assign_jobs_to_lines_balanced(all_jobs, lines):
    buckets = {line.name: [] for line in lines}
    unassigned = []

    # simple “load” metric: total qty assigned so far
    load = defaultdict(int)

    for job in all_jobs:
        eligible_lines = [ln for ln in lines if ln.can_run(job)]
        if not eligible_lines:
            unassigned.append(job)
            continue

        # pick the line with the smallest current load
        best = min(eligible_lines, key=lambda ln: load[ln.name])

        buckets[best.name].append(job)
        load[best.name] += int(job.qty)

    return buckets, unassigned

def jobs_from_agg(agg) -> list[Job]:
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
                can_size=str(getattr(r, "CAN_SIZE", "") or ""),
                erp_line=(None if pd.isna(getattr(r, "LINE", None)) else str(getattr(r, "LINE"))),
            )
        )
    return jobs

def df_to_tuples(df, cols):
    # Convert NaN/NaT to None so pyodbc sends NULL
    out = df[cols].copy()
    out = out.where(pd.notna(out), None)
    return list(map(tuple, out.to_numpy()))

def write_schedule_to_sqlserver(conn_str: str, work_orders_df: pd.DataFrame, schedule_df: pd.DataFrame):
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
            # Full refresh (delete children first)
            cur.execute("DELETE FROM dbo.schedule;")
            cur.execute("DELETE FROM dbo.work_orders;")

            # Fast bulk insert
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_xlsx", help="Input Excel file path")
    args = ap.parse_args()

    in_path = Path(args.input_xlsx).expanduser().resolve()
    start_time = "2025-12-30 06:00"

    data = load_and_clean(in_path)
    all_jobs = jobs_from_agg(data) 

    plant = PlantSchedule()

    line_4 = Line(name="LINE4", daily_capacity=0, decorator_names=("A", "B"),can_run_fn=lambda job: job.can_size == "710")
    lines = [line_4] 
    buckets, unassigned = assign_jobs_to_lines_balanced(all_jobs, lines) 
    sched4 = Scheduler(buckets["LINE4"], line_4, plant).run()
    blocks_data, deco_a_data, deco_b_data, family_data = sched4.results_to_dataframes()
    schedule_df = pd.concat([deco_a_data, deco_b_data], ignore_index=True)

    # match dbo.schedule columns exactly
    schedule_df = schedule_df.rename(columns={
        "LINE": "line",
        "DECORATOR": "decorator",
        "SEQ": "seq",
        "BLOCK": "block",
        "ROLE_IN_BLOCK": "role_in_block",
        "WO": "wo",
    })[["wo", "line", "decorator", "seq", "block", "role_in_block"]]

    # enforce types
    schedule_df["wo"] = schedule_df["wo"].astype(str)
    schedule_df["line"] = schedule_df["line"].astype(str)
    schedule_df["decorator"] = schedule_df["decorator"].astype(str)
    schedule_df["seq"] = schedule_df["seq"].astype(int)
    schedule_df["block"] = schedule_df["block"].astype(int)
    schedule_df["role_in_block"] = schedule_df["role_in_block"].astype(str)

    work_orders_df = pd.concat([deco_a_data, deco_b_data], ignore_index=True)

    work_orders_df = (
        work_orders_df.rename(columns={
            "WO": "wo",
            "QTY": "qty",
            "FAMILY": "family",
            "PRIMARY_COLOR": "primary_color",
            "DESCRIPTION": "description",
            "ITEM_NUMBER": "item_number",
            "REQ_DATE": "req_date",
            "CAN_SIZE": "can_size",
        })[
            ["wo", "qty", "family", "primary_color", "description", "item_number", "req_date", "can_size"]
        ]
    )

    # Enforce types + normalize nulls
    work_orders_df["wo"] = work_orders_df["wo"].astype(str)
    work_orders_df["qty"] = pd.to_numeric(work_orders_df["qty"], errors="coerce").fillna(0).astype(int)

    for col in ["family", "primary_color", "description", "item_number", "can_size"]:
        work_orders_df[col] = work_orders_df[col].where(pd.notna(work_orders_df[col]), None)

    # datetime null handling (NaT -> None)
    work_orders_df["req_date"] = pd.to_datetime(work_orders_df["req_date"], errors="coerce")
    work_orders_df["req_date"] = work_orders_df["req_date"].where(work_orders_df["req_date"].notna(), None)

    # Deduplicate by WO (keep first)
    work_orders_df = work_orders_df.drop_duplicates(subset=["wo"], keep="first").reset_index(drop=True)
    conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=dscsqc;"
    "DATABASE=SchedulePlanning;"
    "Trusted_Connection=yes;"
    )
    write_schedule_to_sqlserver(conn_str, work_orders_df, schedule_df)

    # timeline uses the ScheduledJob objects; fetch them from plant
    deco_a_objs = plant.by_resource("LINE4", "A")
    deco_b_objs = plant.by_resource("LINE4", "B")

    deco_a_data, deco_b_data, timeline_df = build_timeline_from_objects(
        deco_a_objs,
        deco_b_objs,
        start_time=start_time,
    )

    export_timeline_json(
    timeline_df,
    Path("output.timeline.json"),
    start_time=start_time,
    deco_a_df=deco_a_data,
    deco_b_df=deco_b_data,
)



if __name__ == "__main__":
    main()