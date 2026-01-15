from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    Tuple,
)
import math
import re

import pandas as pd


# ----------------------------
# Types / Exceptions
# ----------------------------

ResourceKey = Tuple[str, str]  # (line, decorator)


class ScheduleError(Exception):
    pass


class DuplicateWOError(ScheduleError):
    pass


class LockedJobError(ScheduleError):
    pass


class ValidationError(ScheduleError):
    pass


# ----------------------------
# Locking model
# ----------------------------

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


# ----------------------------
# Production clocks
# ----------------------------

class ProductionClock(Protocol):
    """Advance time by producing qty, returning (new_cur, pieces)."""
    def consume(self, cur: datetime, qty: int) -> tuple[datetime, list[dict[str, Any]]]:
        ...


@dataclass
class ConstantRateClock:
    """No downtime calendar: time = qty / rate."""
    rate_cph: float

    def consume(self, cur: datetime, qty: int) -> tuple[datetime, list[dict[str, Any]]]:
        if qty <= 0:
            return cur, []
        if self.rate_cph <= 0:
            raise ValueError("rate_cph must be > 0")

        hours = float(qty) / float(self.rate_cph)
        end = cur + timedelta(seconds=hours * 3600.0)
        return end, [{"TYPE": "RUN", "START": cur, "FINISH": end, "QTY_RUN": int(qty)}]


@dataclass
class CalendarCursor:
    """
    Pointer into an hourly calendar.

    calendar_df must be sorted by HourStart ascending and include:
      - HourStart (datetime-like)
      - IsDown (0/1)
    """
    calendar_df: pd.DataFrame
    idx: int = 0


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def consume_qty_with_hourly_calendar(
    cur: datetime,
    qty: int,
    *,
    rate_cph: float,
    cal: CalendarCursor,
) -> tuple[datetime, list[dict[str, Any]]]:
    """
    Produce `qty` at `rate_cph` while respecting hourly downtime.
    Returns (new_cur, pieces).

    pieces: list of {"TYPE": "RUN"|"DOWN", "START": dt, "FINISH": dt, "QTY_RUN": int}
    """
    if qty <= 0:
        return cur, []
    if rate_cph <= 0:
        raise ValueError("rate_cph must be > 0")

    df = cal.calendar_df
    if df.empty:
        raise ValueError("calendar_df is empty")

    remaining = float(qty)
    pieces: list[dict[str, Any]] = []

    # Align cursor to the hour containing cur (or the next hour in the df)
    cur_hour = _floor_to_hour(cur)
    while cal.idx < len(df) and pd.to_datetime(df.loc[cal.idx, "HourStart"]).to_pydatetime() < cur_hour:
        cal.idx += 1

    while remaining > 1e-9:
        if cal.idx >= len(df):
            raise RuntimeError("Ran out of calendar hours; extend your horizon.")

        hour_start = pd.to_datetime(df.loc[cal.idx, "HourStart"]).to_pydatetime()
        hour_end = hour_start + timedelta(hours=1)
        is_down = int(df.loc[cal.idx, "IsDown"])

        # Jump forward if we're before this hour
        if cur < hour_start:
            cur = hour_start

        # If we're already past this hour, advance cursor
        if cur >= hour_end:
            cal.idx += 1
            continue

        # If down, skip whole hour
        if is_down == 1:
            pieces.append({
                "TYPE": "DOWN",
                "START": cur,
                "FINISH": hour_end,
                "QTY_RUN": 0,
            })
            cur = hour_end
            cal.idx += 1
            continue

        # Available run window in this hour
        available_hours = (hour_end - cur).total_seconds() / 3600.0
        available_qty = available_hours * float(rate_cph)

        run_qty = min(remaining, available_qty)
        run_hours = run_qty / float(rate_cph)
        run_end = cur + timedelta(seconds=run_hours * 3600.0)

        pieces.append({
            "TYPE": "RUN",
            "START": cur,
            "FINISH": run_end,
            "QTY_RUN": int(round(run_qty)),
        })

        remaining -= run_qty
        cur = run_end

        # If we hit end of hour, move to next row
        if cur >= hour_end - timedelta(microseconds=1):
            cur = hour_end
            cal.idx += 1

    return cur, pieces


@dataclass
class HourlyCalendarClock:
    """Downtime-aware: advances time through CalendarCursor."""
    rate_cph: float
    cal: CalendarCursor

    def consume(self, cur: datetime, qty: int) -> tuple[datetime, list[dict[str, Any]]]:
        return consume_qty_with_hourly_calendar(cur, qty, rate_cph=self.rate_cph, cal=self.cal)


def clone_clock(clock: ProductionClock) -> ProductionClock:
    """
    Return a copy of the clock that can be mutated in 'what-if' simulations
    without affecting the real schedule clock.
    """
    if isinstance(clock, HourlyCalendarClock):
        return HourlyCalendarClock(
            rate_cph=float(clock.rate_cph),
            cal=CalendarCursor(calendar_df=clock.cal.calendar_df, idx=int(clock.cal.idx)),
        )
    if isinstance(clock, ConstantRateClock):
        return ConstantRateClock(rate_cph=float(clock.rate_cph))
    raise TypeError(f"Unsupported clock type: {type(clock)}")


# ----------------------------
# Slack / due-date helpers
# ----------------------------

def is_late(finish: datetime, req_date: Optional[datetime]) -> bool:
    if req_date is None:
        return False
    return finish > req_date


def slack_hours(finish: datetime, req_date: Optional[datetime]) -> float:
    if req_date is None:
        return float("inf")
    return (req_date - finish).total_seconds() / 3600.0


# ----------------------------
# Core simulation primitive
# ----------------------------

class BlockSimResult(NamedTuple):
    end_time: datetime
    wo_finish: dict[str, datetime]   # finish time for each WO in block
    min_slack_hrs: float             # min slack across block jobs (req_date only)


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


def simulate_block(
    *,
    start_time: datetime,
    clock: ProductionClock,
    anchor_deco: str,
    anchor_job: Job,
    small_jobs: list[Job],
    setup_equiv_qty: int = 60_000,
    anchor_slice_qty: int = 60_000,
) -> BlockSimResult:
    """
    Simulate ONE block using the same interleaving rules as build_timeline_from_objects,
    but limited to the anchor and the chosen small jobs.
    Returns predicted end_time and finish times for WOs in this block.
    """
    if setup_equiv_qty <= 0 or anchor_slice_qty <= 0:
        raise ValueError("setup_equiv_qty and anchor_slice_qty must be > 0")

    cur = start_time
    producing = anchor_deco

    setup_progress = {"A": setup_equiv_qty, "B": setup_equiv_qty}

    anchor_done = 0
    small_states = [{"job": j, "done": 0} for j in small_jobs]
    small_idx = 0

    wo_finish: dict[str, datetime] = {}

    def other(d: str) -> str:
        return "B" if d == "A" else "A"

    def anchor_remaining() -> int:
        return max(0, int(anchor_job.qty) - int(anchor_done))

    def small_remaining(idx: int) -> int:
        st = small_states[idx]
        return max(0, int(st["job"].qty) - int(st["done"]))

    def next_job_ready(d: str) -> bool:
        if d == anchor_deco:
            return anchor_remaining() > 0 and setup_progress[d] >= setup_equiv_qty
        return small_idx < len(small_states) and setup_progress[d] >= setup_equiv_qty

    def run_qty_for(d: str) -> tuple[str, Optional[datetime], int]:
        if d == anchor_deco:
            qty = min(anchor_slice_qty, anchor_remaining())
            return anchor_job.wo, anchor_job.req_date, int(qty)
        st = small_states[small_idx]
        qty = small_remaining(small_idx)
        return st["job"].wo, st["job"].req_date, int(qty)

    def mark_run(d: str, qty: int, finish_time: datetime) -> None:
        nonlocal anchor_done, small_idx
        if d == anchor_deco:
            anchor_done += qty
            if anchor_remaining() == 0:
                wo_finish[anchor_job.wo] = finish_time
        else:
            st = small_states[small_idx]
            st["done"] += qty
            if small_remaining(small_idx) == 0:
                wo_finish[st["job"].wo] = finish_time
                small_idx += 1
                setup_progress[d] = 0

    while anchor_remaining() > 0 or small_idx < len(small_states):
        if not next_job_ready(producing):
            if next_job_ready(other(producing)):
                producing = other(producing)
            else:
                setup_progress[producing] = setup_equiv_qty

        if not next_job_ready(producing):
            break

        wo, _req, qty_plan = run_qty_for(producing)
        if qty_plan <= 0:
            break

        setup_progress[other(producing)] = min(
            setup_equiv_qty,
            setup_progress[other(producing)] + int(qty_plan),
        )

        cur, _pieces = clock.consume(cur, int(qty_plan))
        mark_run(producing, int(qty_plan), finish_time=cur)

        if next_job_ready(other(producing)):
            producing = other(producing)

    min_slack = float("inf")
    for j in [anchor_job] + [x["job"] for x in small_states]:
        fin = wo_finish.get(j.wo)
        if fin is None:
            min_slack = -float("inf")
            continue
        min_slack = min(min_slack, slack_hours(fin, j.req_date))

    return BlockSimResult(end_time=cur, wo_finish=wo_finish, min_slack_hrs=min_slack)


# ----------------------------
# Plant schedule container
# ----------------------------

@dataclass(frozen=True, slots=True)
class ScheduledJob:
    line: str
    block: int
    decorator: str
    seq: int
    role_in_block: str
    job: Job


@dataclass
class PlantSchedule:
    """
    Authoritative container for ScheduledJob across the whole plant.

    Key invariant (no WO splitting):
      - each WO appears at most once across the entire schedule
    """
    _by_wo: Dict[str, ScheduledJob] = field(default_factory=dict)
    _locks: Dict[str, JobLock] = field(default_factory=dict)

    def lock(self, wo: str, lock: JobLock) -> None:
        self._locks[str(wo)] = lock

    def unlock(self, wo: str) -> None:
        self._locks.pop(str(wo), None)

    def lock_for(self, wo: str) -> Optional[JobLock]:
        return self._locks.get(str(wo))

    def is_frozen(self, wo: str) -> bool:
        lk = self._locks.get(str(wo))
        return bool(lk and lk.frozen)

    def add(self, sj: ScheduledJob) -> None:
        wo = str(sj.job.wo)
        if wo in self._by_wo:
            existing = self._by_wo[wo]
            raise DuplicateWOError(
                f"WO {wo} already scheduled on {existing.line}/{existing.decorator} "
                f"(trying {sj.line}/{sj.decorator})."
            )
        self._enforce_lock(sj)
        self._by_wo[wo] = sj

    def upsert(self, sj: ScheduledJob, *, force: bool = False) -> None:
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

    def get(self, wo: str) -> Optional[ScheduledJob]:
        return self._by_wo.get(str(wo))

    def all(self) -> List[ScheduledJob]:
        return list(self._by_wo.values())

    def resources(self) -> Set[ResourceKey]:
        return {(sj.line, sj.decorator) for sj in self._by_wo.values()}

    def by_line(self, line: str) -> List[ScheduledJob]:
        out = [sj for sj in self._by_wo.values() if sj.line == line]
        out.sort(key=lambda sj: (sj.decorator, int(sj.seq)))
        return out

    def by_resource(self, line: str, decorator: str) -> List[ScheduledJob]:
        out = [sj for sj in self._by_wo.values() if sj.line == line and sj.decorator == decorator]
        out.sort(key=lambda sj: int(sj.seq))
        return out

    def validate(
        self,
        *,
        can_run: Optional[Callable[[ScheduledJob], bool]] = None,
        require_unique_seq: bool = True,
    ) -> None:
        for sj in self._by_wo.values():
            self._enforce_lock(sj)

        if can_run is not None:
            bad = [sj for sj in self._by_wo.values() if not can_run(sj)]
            if bad:
                sample = ", ".join(f"{x.job.wo}@{x.line}/{x.decorator}" for x in bad[:10])
                raise ValidationError(f"Infeasible jobs in schedule (sample): {sample}")

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
        grouped: Dict[ResourceKey, List[ScheduledJob]] = {}
        for sj in self._by_wo.values():
            grouped.setdefault((sj.line, sj.decorator), []).append(sj)

        new_by_wo = dict(self._by_wo)

        for rk, lst in grouped.items():
            lst.sort(key=lambda sj: int(sj.seq))
            fixed_map: Dict[int, ScheduledJob] = {}
            movable: List[ScheduledJob] = []

            for sj in lst:
                lk = self._locks.get(str(sj.job.wo))
                if lk and lk.fixed_seq is not None:
                    fs = int(lk.fixed_seq)
                    if fs in fixed_map and str(fixed_map[fs].job.wo) != str(sj.job.wo):
                        raise LockedJobError(f"fixed_seq collision on {rk} at seq={fs}.")
                    fixed_map[fs] = sj
                else:
                    movable.append(sj)

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

    def _enforce_lock(self, sj: ScheduledJob) -> None:
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


# ----------------------------
# Line / eligibility model
# ----------------------------

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

    def can_run(self, job: Job):
        return bool(self._can_run_fn(job))

    def decorator(self, name: str):
        return self.decorators[name]

    def other_decorator(self, name: str):
        return next(k for k in self.decorators if k != name)


class Decorator:
    def __init__(self, name):
        self.name = name
        self.jobs = []

    def add(self, scheduled_job: ScheduledJob):
        self.jobs.append(scheduled_job)


# ----------------------------
# Scheduler (sorting algorithm)
# ----------------------------

class Scheduler:
    """
    Scheduler with deadline-aware "promotion".
    Keeps your object models & algorithm together for easy LLM handoff.
    """

    def __init__(self, jobs: list[Job], line: Line, plant: PlantSchedule, *, clock: ProductionClock, start_time: datetime):
        self.pool = JobPool(jobs)
        self.line = line
        self.plant = plant
        self.clock = clock
        self.cur_time = start_time

        self.family_assignment: dict[str, str] = {}
        self.last_family = {"A": None, "B": None}
        self.last_color = {"A": None, "B": None}
        self.block_summaries: list[dict[str, Any]] = []

        self.block_number = 0
        self.preferred_anchor_decorator = "A"
        self.pinned = {"A": deque(), "B": deque()}
        self.item_assignment: dict[str, str] = {}
        self.last_item = {"A": None, "B": None}

        self._anchor_slice_qty = 60_000
        self._setup_equiv_qty = 60_000

        # populated by seed_pinned_jobs
        self.erp_locked: dict[str, set[str]] = {"A": set(), "B": set()}

    def is_item_eligible(self, job: Job, decorator_name: str) -> bool:
        it = job.item_number
        if not it:
            return True
        return (it not in self.item_assignment) or (self.item_assignment[it] == decorator_name)

    def lock_item(self, item_number: Optional[str], decorator_name: str) -> None:
        if not item_number:
            return
        if item_number not in self.item_assignment:
            self.item_assignment[item_number] = decorator_name

    def remove_job(self, wo: str) -> None:
        self.pool.remove_wo(wo)

    def remove_pinned_wo(self, decorator_name: str, wo: str) -> bool:
        q = self.pinned.get(decorator_name)
        if not q:
            return False
        for j in list(q):
            if j.wo == wo:
                q.remove(j)
                return True
        return False

    def _remove_job_everywhere(self, decorator_name: str, wo: str) -> bool:
        if self.remove_pinned_wo(decorator_name, wo):
            return True
        self.remove_job(wo)
        return False

    def is_eligible(self, job: Job, decorator_name: str) -> bool:
        if job.required_decorator in ("A", "B") and job.required_decorator != decorator_name:
            return False

        if not self.is_item_eligible(job, decorator_name):
            return False

        fam = job.family
        if fam == "UNKNOWN":
            return True
        return (fam not in self.family_assignment) or (self.family_assignment[fam] == decorator_name)

    def lock_family(self, family: str, decorator_name: str) -> None:
        if family == "UNKNOWN":
            return
        if family not in self.family_assignment:
            self.family_assignment[family] = decorator_name

    def pin_item_siblings(self, item_number: Optional[str], decorator_name: str) -> None:
        if not item_number:
            return

        sibs = [j for j in self.pool.values() if j.item_number == item_number]
        if not sibs:
            return

        sibs.sort(key=lambda j: (-int(j.qty), int(getattr(j, "erp_seq", 0) or 0), j.wo))

        q = self.pinned[decorator_name]
        erp_mode = bool(self.erp_locked.get(decorator_name))
        for j in sibs:
            if erp_mode:
                q.append(j)
            else:
                q.appendleft(j)
            self.pool.remove_wo(j.wo)

    def schedule_job(self, decorator_name: str, block_no: int, job: Job, role_in_block: str) -> None:
        existing = self.plant.by_resource(self.line.name, decorator_name)
        max_seq = max((int(sj.seq) for sj in existing), default=0)

        lk = self.plant.lock_for(job.wo)
        if lk and lk.fixed_seq is not None:
            seq = int(lk.fixed_seq)
        else:
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

        self.erp_locked.get(decorator_name, set()).discard(job.wo)

        self.last_family[decorator_name] = job.family
        self.last_color[decorator_name] = job.primary_color
        self.lock_item(job.item_number, decorator_name)
        self.last_item[decorator_name] = job.item_number
        self.pin_item_siblings(job.item_number, decorator_name)

    def seed_pinned_jobs(self, qty_cap_per_decorator: int = 200_000) -> None:
        self.erp_locked = {"A": set(), "B": set()}
        target = norm_line(self.line.name)

        for dec_name in ("A", "B"):
            sequenced = [
                j for j in self.pool.values()
                if j.required_decorator == dec_name
                and int(getattr(j, "erp_seq", 0) or 0) > 0
                and norm_line(j.erp_line) == target
            ]
            sequenced.sort(key=lambda j: (int(j.erp_seq), j.wo))

            wo_to_rank: dict[str, int] = {}
            for rank, j in enumerate(sequenced, start=1):
                wo_to_rank[j.wo] = rank

            cum = 0
            pinned_jobs: list[Job] = []
            protected_wos: set[str] = set()

            for j in sequenced:
                if cum >= qty_cap_per_decorator:
                    break
                pinned_jobs.append(j)
                protected_wos.add(j.wo)
                cum += int(j.qty)

            self.pinned[dec_name] = deque(pinned_jobs)

            for j in pinned_jobs:
                rank = wo_to_rank[j.wo]
                self.erp_locked[dec_name].add(j.wo)
                self.plant.lock(
                    j.wo,
                    JobLock(
                        required_resource=(self.line.name, dec_name),
                        fixed_seq=rank,
                        frozen=True,
                        reason="ERP first 200k (reindexed)",
                    ),
                )

            for j in sequenced:
                if j.wo in protected_wos:
                    self.pool.remove_wo(j.wo)
                else:
                    self.pool._by_wo[j.wo] = replace(j, erp_seq=0, required_decorator=None)

    def pick_small_jobs_due(self, decorator_name: str, count: int) -> list[Job]:
        if count <= 0:
            return []

        candidates = [
            j for j in self.pool.values()
            if int(j.qty) < 60000
            and self.line.can_run(j)
            and self.is_eligible(j, decorator_name)
        ]
        if not candidates:
            return []

        current_item = self.last_item.get(decorator_name)
        current_fam = self.last_family.get(decorator_name)
        current_color = self.last_color.get(decorator_name)

        def key(j: Job):
            due = j.req_date or datetime.max
            stay_item = int(bool(current_item) and j.item_number and j.item_number == current_item)
            stay_family = int(bool(current_fam) and current_fam != "UNKNOWN" and j.family == current_fam)
            stay_color = int(bool(current_color) and current_color != "UNKNOWN" and j.primary_color == current_color)
            return (due, -stay_item, -stay_family, -stay_color, j.color_rank, j.qty, j.wo)

        candidates.sort(key=key)
        return candidates[:count]

    def _iter_unscheduled_jobs(self, decorator_name: str) -> Iterable[Job]:
        for j in self.pinned.get(decorator_name, deque()):
            yield j
        for j in self.pool.values():
            yield j

    def _next_due_job(self, decorator_name: str) -> Optional[Job]:
        cands: list[Job] = []
        for j in self._iter_unscheduled_jobs(decorator_name):
            if j.req_date is None:
                continue
            if not self.line.can_run(j):
                continue
            if not self.is_eligible(j, decorator_name):
                continue
            cands.append(j)

        if not cands:
            return None

        cands.sort(key=lambda j: (j.req_date or datetime.max, -int(j.qty), j.color_rank, j.wo))
        return cands[0]

    def _build_other_side_jobs_for_anchor(self, other_deco: str, anchor_job: Job) -> list[Job]:
        anchor_qty = int(anchor_job.qty)
        target_count_for_other = max(1, int(math.floor(anchor_qty / 60000.0)))
        small_jobs_needed = max(0, target_count_for_other - 1)

        chosen: list[Job] = []
        erp_protect_other = bool(self.erp_locked.get(other_deco))
        pinned_other_preview = list(self.pinned[other_deco])[:50]

        if erp_protect_other:
            chosen.extend(pinned_other_preview[:small_jobs_needed])
        else:
            small_only = [j for j in pinned_other_preview if int(j.qty) < 60000]
            chosen.extend(small_only[:small_jobs_needed])

        remaining_needed = max(0, small_jobs_needed - len(chosen))
        if remaining_needed > 0:
            chosen.extend(self.pick_small_jobs_due(other_deco, remaining_needed))

        return chosen

    def _finish_if_anchored_now(self, decorator_name: str, job: Job) -> Optional[datetime]:
        other = self.line.other_decorator(decorator_name)
        other_jobs = self._build_other_side_jobs_for_anchor(other, job)

        sim_clock = clone_clock(self.clock)
        sim = simulate_block(
            start_time=self.cur_time,
            clock=sim_clock,
            anchor_deco=decorator_name,
            anchor_job=job,
            small_jobs=other_jobs,
            setup_equiv_qty=self._setup_equiv_qty,
            anchor_slice_qty=self._anchor_slice_qty,
        )
        return sim.wo_finish.get(job.wo)

    def _find_promotable_critical_anchor(self) -> Optional[tuple[str, Job]]:
        buffer_hours = 0.0
        candidates: list[tuple[float, str, Job, datetime]] = []

        for dec in ("A", "B"):
            j = self._next_due_job(dec)
            if j is None:
                continue

            fin = self._finish_if_anchored_now(dec, j)
            if fin is None:
                continue

            if j.req_date is not None and fin > j.req_date:
                raise ValidationError(
                    f"WO {j.wo} cannot meet REQ_DATE even if anchored immediately on {self.line.name}/{dec}. "
                    f"Finish={fin}, REQ_DATE={j.req_date}."
                )

            s = slack_hours(fin, j.req_date)
            candidates.append((s, dec, j, fin))

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t[0], (t[2].req_date or datetime.max).timestamp(), -int(t[2].qty), t[2].wo))
        best_slack, best_dec, best_job, _fin = candidates[0]

        if best_slack <= buffer_hours:
            return (best_dec, best_job)

        return None

    def run(self):
        self.seed_pinned_jobs(qty_cap_per_decorator=200_000)

        def has_work() -> bool:
            return self.pool.has_work() or bool(self.pinned["A"]) or bool(self.pinned["B"])

        while has_work():
            promoted = self._find_promotable_critical_anchor()

            if promoted is not None:
                pref, forced_job = promoted
                other = self.line.other_decorator(pref)

                if self.pinned[pref]:
                    forced = None
                    for j in self.pinned[pref]:
                        if self.line.can_run(j) and self.is_eligible(j, pref):
                            forced = j
                            break
                    if forced is None:
                        raise ValidationError(f"Pinned ERP jobs exist on {self.line.name}/{pref} but none are eligible.")
                    anchor_shortlist = [forced]
                else:
                    anchor_shortlist = [forced_job]
            else:
                pref = self.preferred_anchor_decorator
                other = self.line.other_decorator(pref)

                if self.pinned[pref]:
                    forced = None
                    for j in self.pinned[pref]:
                        if self.line.can_run(j) and self.is_eligible(j, pref):
                            forced = j
                            break
                    if forced is None:
                        raise ValidationError(f"Pinned ERP jobs exist on {self.line.name}/{pref} but none are eligible.")
                    anchor_shortlist = [forced]
                else:
                    pool_candidates = [
                        j for j in self.pool.values()
                        if self.line.can_run(j) and self.is_eligible(j, pref) and int(j.qty) >= 60000
                    ]
                    pool_candidates.sort(key=lambda j: (j.req_date or datetime.max, -int(j.qty), j.color_rank, j.wo))

                    if not pool_candidates:
                        pool_candidates = [
                            j for j in self.pool.values()
                            if self.line.can_run(j) and self.is_eligible(j, pref)
                        ]
                        pool_candidates.sort(key=lambda j: (j.req_date or datetime.max, -int(j.qty), j.color_rank, j.wo))

                    anchor_shortlist = pool_candidates[:20]

            if not anchor_shortlist:
                pref = other
                other = self.line.other_decorator(pref)
                self.preferred_anchor_decorator = pref
                continue

            best_choice = None  # (score_tuple, anchor_job, chosen_other_jobs)

            def would_be_late(sim: BlockSimResult, jobs: list[Job]) -> bool:
                for j in jobs:
                    if j.req_date is None:
                        continue
                    fin = sim.wo_finish.get(j.wo)
                    if fin is None or is_late(fin, j.req_date):
                        return True
                return False

            for candidate_anchor in anchor_shortlist:
                anchor_deco = pref
                other_deco = other

                chosen_other_jobs = self._build_other_side_jobs_for_anchor(other_deco, candidate_anchor)

                sim_clock = clone_clock(self.clock)
                sim = simulate_block(
                    start_time=self.cur_time,
                    clock=sim_clock,
                    anchor_deco=anchor_deco,
                    anchor_job=candidate_anchor,
                    small_jobs=chosen_other_jobs,
                    setup_equiv_qty=self._setup_equiv_qty,
                    anchor_slice_qty=self._anchor_slice_qty,
                )

                block_jobs = [candidate_anchor] + chosen_other_jobs
                if would_be_late(sim, block_jobs):
                    continue

                score = (
                    -sim.min_slack_hrs,
                    (candidate_anchor.req_date or datetime.max).timestamp(),
                    -int(candidate_anchor.qty),
                    int(candidate_anchor.color_rank),
                )

                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, candidate_anchor, chosen_other_jobs)

            if best_choice is None:
                raise ValidationError(
                    f"No feasible anchor/other selection found at time {self.cur_time} on {self.line.name}/{pref}. "
                    "This implies infeasible due-dates under current constraints/calendar."
                )

            _score, anchor_job, other_jobs = best_choice

            self._remove_job_everywhere(pref, anchor_job.wo)

            self.block_number += 1
            self.lock_family(anchor_job.family, pref)
            self.lock_item(anchor_job.item_number, pref)
            self.schedule_job(pref, self.block_number, anchor_job, "ANCHOR")

            other_decorator = self.line.other_decorator(pref)
            for j in other_jobs:
                self._remove_job_everywhere(other_decorator, j.wo)
                self.lock_family(j.family, other_decorator)
                self.lock_item(j.item_number, other_decorator)
                self.schedule_job(other_decorator, self.block_number, j, "SMALL")

            committed = simulate_block(
                start_time=self.cur_time,
                clock=self.clock,
                anchor_deco=pref,
                anchor_job=anchor_job,
                small_jobs=other_jobs,
                setup_equiv_qty=self._setup_equiv_qty,
                anchor_slice_qty=self._anchor_slice_qty,
            )
            self.cur_time = committed.end_time

            self.preferred_anchor_decorator = other_decorator

            self.block_summaries.append({
                "BLOCK": self.block_number,
                "ANCHOR_DECORATOR": pref,
                "ANCHOR_WO": anchor_job.wo,
                "ANCHOR_QTY": int(anchor_job.qty),
                "ANCHOR_FAMILY": anchor_job.family,
                "SMALL_USED": int(len(other_jobs)),
                "BLOCK_END": self.cur_time,
                "MIN_SLACK_HRS": committed.min_slack_hrs,
            })

        return self

    def results_to_dataframes(self):
        blocks_df = pd.DataFrame(self.block_summaries)
        decorator_a_df = scheduled_jobs_to_df(self.plant.by_resource(self.line.name, "A"))
        decorator_b_df = scheduled_jobs_to_df(self.plant.by_resource(self.line.name, "B"))
        family_assignment_df = pd.DataFrame(
            [{"FAMILY": fam, "DECORATOR": dec} for fam, dec in sorted(self.family_assignment.items())]
        )
        return blocks_df, decorator_a_df, decorator_b_df, family_assignment_df


# ----------------------------
# DataFrame helpers + timeline
# ----------------------------

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


def scheduled_jobs_to_df(jobs: list[ScheduledJob]) -> pd.DataFrame:
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


def build_timeline_from_objects(
    dec_a_jobs: list[ScheduledJob],
    dec_b_jobs: list[ScheduledJob],
    start_time: datetime | str,
    *,
    clock: ProductionClock,
    setup_equiv_qty: int = 60_000,
    anchor_slice_qty: int = 60_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if isinstance(start_time, str):
        cur = pd.to_datetime(start_time).to_pydatetime()
    elif isinstance(start_time, datetime):
        cur = start_time
    else:
        raise TypeError("start_time must be datetime or str")

    if setup_equiv_qty <= 0:
        raise ValueError("setup_equiv_qty must be > 0")
    if anchor_slice_qty <= 0:
        raise ValueError("anchor_slice_qty must be > 0")

    RUNNABLE_ROLES = {"ANCHOR", "SMALL"}

    a_run = [sj for sj in dec_a_jobs if sj.role_in_block in RUNNABLE_ROLES]
    b_run = [sj for sj in dec_b_jobs if sj.role_in_block in RUNNABLE_ROLES]
    a_run.sort(key=lambda sj: (int(sj.block), int(sj.seq)))
    b_run.sort(key=lambda sj: (int(sj.block), int(sj.seq)))

    all_blocks = sorted({sj.block for sj in a_run} | {sj.block for sj in b_run})

    wo_start: Dict[str, datetime] = {}
    wo_finish: Dict[str, datetime] = {}
    timeline_rows: List[Dict[str, Any]] = []

    def append_segment(block_no, producing_deco, role, wo, qty_run, description, req_date):
        nonlocal cur
        if qty_run <= 0:
            return

        seg_start = cur
        cur, pieces = clock.consume(cur, int(qty_run))
        seg_end = cur

        timeline_rows.append({
            "BLOCK": int(block_no),
            "DECORATOR": str(producing_deco),
            "ROLE_IN_BLOCK": str(role),
            "WO": str(wo),
            "QTY_RUN": int(qty_run),
            "START": seg_start,
            "FINISH": seg_end,
            "DURATION_HOURS": (seg_end - seg_start).total_seconds() / 3600.0,
            "JOB_DESCRIPTION": description,
            "REQ_DATE": req_date,
            "TYPE": "RUN",
        })

        for p in pieces:
            if p.get("TYPE") == "DOWN":
                timeline_rows.append({
                    "BLOCK": int(block_no),
                    "DECORATOR": str(producing_deco),
                    "ROLE_IN_BLOCK": "DOWN",
                    "WO": "",
                    "QTY_RUN": 0,
                    "START": p["START"],
                    "FINISH": p["FINISH"],
                    "DURATION_HOURS": (p["FINISH"] - p["START"]).total_seconds() / 3600.0,
                    "JOB_DESCRIPTION": "LINE DOWN",
                    "REQ_DATE": None,
                    "TYPE": "DOWN",
                })

    def block_jobs(jobs: list[ScheduledJob], block_no: int, role: str) -> list[JobState]:
        blk = [sj for sj in jobs if sj.block == block_no and sj.role_in_block == role]
        blk.sort(key=lambda sj: int(sj.seq))
        return [JobState(job=sj.job) for sj in blk]

    for block in all_blocks:
        a_anchor = [sj for sj in a_run if sj.block == block and sj.role_in_block == "ANCHOR"]
        b_anchor = [sj for sj in b_run if sj.block == block and sj.role_in_block == "ANCHOR"]
        if len(a_anchor) + len(b_anchor) == 0:
            continue

        if len(a_anchor) + len(b_anchor) == 1:
            anchor_sj = a_anchor[0] if len(a_anchor) == 1 else b_anchor[0]
        else:
            anchors = (a_anchor + b_anchor)
            anchors.sort(key=lambda sj: int(sj.seq))
            anchor_sj = anchors[0]

        anchor_deco = str(anchor_sj.decorator)
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

        def run_qty_for(d: str):
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
                    setup_progress[d] = 0

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
                setup_progress[other(producing)] + int(qty_plan)
            )

            append_segment(block, producing, role, wo, int(qty_plan), desc, req_date)
            mark_run(producing, int(qty_plan))

            if next_job_ready(other(producing)):
                producing = other(producing)

    timeline_df = pd.DataFrame(timeline_rows)
    if not timeline_df.empty:
        run_only = timeline_df[(timeline_df["TYPE"] == "RUN") & (timeline_df["WO"].astype(str) != "")]
        if not run_only.empty:
            wo_start = run_only.groupby("WO")["START"].min().to_dict()
            wo_finish = run_only.groupby("WO")["FINISH"].max().to_dict()

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

    a_df = scheduled_jobs_to_df(dec_a_jobs)
    b_df = scheduled_jobs_to_df(dec_b_jobs)
    return annotate_schedule(a_df), annotate_schedule(b_df), timeline_df


# ----------------------------
# Assignment + parsing helpers
# ----------------------------

def norm_line(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "", s)


def item_prefix3(item_number: Optional[str]) -> str:
    if item_number is None:
        return ""
    digits = re.sub(r"\D+", "", str(item_number))
    return digits[:3]


def assign_jobs_to_lines_balanced(all_jobs: list[Job], lines: list[Line]):
    buckets = {line.name: [] for line in lines}
    unassigned: list[Job] = []
    load = defaultdict(int)

    for job in all_jobs:
        eligible_lines = [ln for ln in lines if ln.can_run(job)]
        if not eligible_lines:
            unassigned.append(job)
            continue

        best = min(eligible_lines, key=lambda ln: load[ln.name])
        buckets[best.name].append(job)
        load[best.name] += int(job.qty)

    return buckets, unassigned


def jobs_from_rows(rows) -> list[Job]:
    jobs: list[Job] = []
    for r in rows:
        val = getattr(r, "REQ_DATE", None)
        req_date = None if pd.isna(val) else val.to_pydatetime()

        jobs.append(Job(
            wo=str(r.WO),
            qty=int(r.QTY),
            color_rank=int(r.COLOR_RANK),
            family=str(r.FAMILY) if pd.notna(r.FAMILY) else "UNKNOWN",
            primary_color=str(r.PRIMARY_COLOR) if pd.notna(r.PRIMARY_COLOR) else "UNKNOWN",
            description=str(r.DESCRIPTION) if pd.notna(r.DESCRIPTION) else "",
            item_number=str(r.ITEM_NUMBER) if pd.notna(r.ITEM_NUMBER) else None,
            required_decorator=(
                None if pd.isna(getattr(r, "REQ_DECORATOR", None))
                else str(r.REQ_DECORATOR)
            ),
            erp_seq=int(getattr(r, "SEQ", 0) or 0),
            req_date=req_date,
            can_size=str(getattr(r, "CAN_SIZE", "") or ""),
            erp_line=(
                None if pd.isna(getattr(r, "LINE", None))
                else str(getattr(r, "LINE"))
            ),
        ))
    return jobs
