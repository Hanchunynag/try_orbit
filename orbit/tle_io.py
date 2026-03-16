"""TLE file parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from sgp4.api import Satrec


@dataclass
class TLESatellite:
    """Parsed TLE satellite record."""

    name: str
    line1: str
    line2: str
    satrec: Satrec
    norad_id: str
    epoch_utc: datetime


def jd_to_datetime_utc(jd: float, fr: float) -> datetime:
    """Convert Julian date + fraction into a timezone-aware UTC datetime."""
    full_jd = jd + fr
    unix_seconds = (full_jd - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc)


def _read_tle_blocks(tle_file: str) -> List[tuple[str, str, str]]:
    """Read a standard 2-line or 3-line TLE file into named blocks."""
    path = Path(tle_file)
    if not path.exists():
        raise FileNotFoundError(f"TLE file not found: {tle_file}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("TLE file is empty.")

    blocks: List[tuple[str, str, str]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("1 ") and idx + 1 < len(lines) and lines[idx + 1].startswith("2 "):
            sat_name = f"SAT_{len(blocks):03d}"
            blocks.append((sat_name, lines[idx], lines[idx + 1]))
            idx += 2
            continue
        if idx + 2 < len(lines) and lines[idx + 1].startswith("1 ") and lines[idx + 2].startswith("2 "):
            blocks.append((lines[idx], lines[idx + 1], lines[idx + 2]))
            idx += 3
            continue
        raise ValueError(f"Malformed TLE content near line {idx + 1}: {line}")
    return blocks


def load_tle_satellite(tle_file: str, sat_name: Optional[str] = None) -> TLESatellite:
    """Load a single satellite from a TLE file."""
    satellites: List[TLESatellite] = []
    for name, line1, line2 in _read_tle_blocks(tle_file):
        satrec = Satrec.twoline2rv(line1, line2)
        satellites.append(
            TLESatellite(
                name=name,
                line1=line1,
                line2=line2,
                satrec=satrec,
                norad_id=line1[2:7].strip(),
                epoch_utc=jd_to_datetime_utc(satrec.jdsatepoch, satrec.jdsatepochF),
            )
        )

    if sat_name is None:
        if len(satellites) != 1:
            available = ", ".join(s.name for s in satellites[:10])
            raise ValueError(
                f"Multiple satellites found in TLE. Please provide --sat_name. "
                f"First entries: {available}"
            )
        return satellites[0]

    target = sat_name.strip().lower()
    for sat in satellites:
        if sat.name.strip().lower() == target:
            return sat
    available = ", ".join(s.name for s in satellites[:10])
    raise ValueError(f"Satellite '{sat_name}' not found. Available examples: {available}")
