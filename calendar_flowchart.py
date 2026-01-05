from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class TimelineEvent:
    label: str
    start: datetime
    end: datetime
    row: int = 0  # 0 = top row, 1 = second row, etc.


def dt_range_hours(start: datetime, hours: int) -> Tuple[datetime, datetime]:
    return start, start + timedelta(hours=hours)


def is_holiday(d: date, holidays: Iterable[date]) -> bool:
    return d in set(holidays)


def is_working_time(
    t: datetime,
    working_weekdays: Iterable[int],
    shift_start_hour: int,
    shift_end_hour: int,
    holidays: Iterable[date],
) -> bool:
    if is_holiday(t.date(), holidays):
        return False
    if t.weekday() not in set(working_weekdays):
        return False
    # Shift window: [start, end)
    return shift_start_hour <= t.hour < shift_end_hour


def draw_hourly_timeline(
    start: datetime,
    hours: int,
    out_path: Path,
    *,
    title: str = "Hourly Calendar Timeline",
    working_weekdays: Iterable[int] = (0, 1, 2, 3, 4),  # Mon-Fri
    shift_start_hour: int = 7,
    shift_end_hour: int = 15,  # 7:00-15:00
    holidays: Optional[List[date]] = None,
    events: Optional[List[TimelineEvent]] = None,
    rows: int = 2,
) -> None:
    holidays = holidays or []
    events = events or []

    end = start + timedelta(hours=hours)

    # Figure sizing: scale with hours a bit, but cap to something reasonable
    fig_w = max(12, min(24, hours * 0.25))
    fig_h = 2.5 + rows * 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xlim(0, hours)
    ax.set_yticks(range(rows))
    ax.set_yticklabels([f"Lane {i}" for i in range(rows)])

    # Background: shade non-working hours per hour block
    for i in range(hours):
        t0 = start + timedelta(hours=i)
        working = is_working_time(
            t0,
            working_weekdays=working_weekdays,
            shift_start_hour=shift_start_hour,
            shift_end_hour=shift_end_hour,
            holidays=holidays,
        )
        if not working:
            # Shade full vertical band for this hour
            ax.add_patch(Rectangle((i, -0.5), 1, rows, alpha=0.12, linewidth=0))

    # Vertical hour gridlines + labels
    ax.set_xticks(range(0, hours + 1, 1))
    ax.grid(axis="x", linewidth=0.5, alpha=0.3)

    # Make major labels every 4 hours (so it stays readable)
    major_every = 4
    major_ticks = list(range(0, hours + 1, major_every))
    major_labels = []
    for h in major_ticks:
        t = start + timedelta(hours=h)
        # Show day change and hour
        major_labels.append(t.strftime("%a %m/%d\n%H:00"))
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels, fontsize=8)

    # Draw events as blocks
    for ev in events:
        # Clamp to window
        ev_start = max(ev.start, start)
        ev_end = min(ev.end, end)
        if ev_end <= start or ev_start >= end:
            continue

        x0 = (ev_start - start).total_seconds() / 3600.0
        x1 = (ev_end - start).total_seconds() / 3600.0
        w = max(0.05, x1 - x0)

        y = ev.row - 0.35
        h = 0.7
        ax.add_patch(Rectangle((x0, y), w, h, linewidth=1.0, fill=False))
        ax.text(
            x0 + w / 2,
            ev.row,
            ev.label,
            ha="center",
            va="center",
            fontsize=9,
            clip_on=True,
        )

    # Cleanup
    ax.set_xlabel("Time (hours from start)")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # ---- Configure your timeline here ----
    # Pick a start date/time
    start_dt = datetime(2025, 12, 30, 6, 0)  # Dec 30, 2025 06:00
    horizon_hours = 48  # two days

    # Example holidays (optional)
    holidays = [
        # date(2026, 1, 1),
    ]

    # Example blocks (optional): think of these as "run" and "changeover"
    events = [
        TimelineEvent("Run Job A", start_dt + timedelta(hours=1), start_dt + timedelta(hours=7), row=0),
        TimelineEvent("Changeover", start_dt + timedelta(hours=7), start_dt + timedelta(hours=9), row=0),
        TimelineEvent("Run Job B", start_dt + timedelta(hours=9), start_dt + timedelta(hours=15), row=0),

        TimelineEvent("Decorator Setup", start_dt + timedelta(hours=6), start_dt + timedelta(hours=9), row=1),
        TimelineEvent("QA / Ink", start_dt + timedelta(hours=10), start_dt + timedelta(hours=12), row=1),
    ]

    draw_hourly_timeline(
        start=start_dt,
        hours=horizon_hours,
        out_path=Path("calendar_timeline.png"),
        title="Line 1 Calendar (Hourly)",
        working_weekdays=(0, 1, 2, 3, 4),  # Mon-Fri
        shift_start_hour=7,
        shift_end_hour=15,
        holidays=holidays,
        events=events,
        rows=2,
    )

    print("Wrote: calendar_timeline.png")
