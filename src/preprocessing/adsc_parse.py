from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


ICAO_RE = re.compile(r"ICAO ID:\s*([0-9A-Fa-f]{6})")
TAG_HEADER_RE = re.compile(r"^\s*Tag\s+(\d+)")
LAT_RE = re.compile(r"^\s*Latitude:\s*([-0-9.]+)")
LON_RE = re.compile(r"^\s*Longitude:\s*([-0-9.]+)")
ALT_RE = re.compile(r"^\s*Altitude:\s*([-0-9.]+)\s*ft", re.IGNORECASE)
TS_RE = re.compile(r"^\s*Timestamp:\s*([0-9:-]+\s+[0-9:.]+)")
ACC_RE = re.compile(r"^\s*Position accuracy:\s*(.+)$", re.IGNORECASE)
ACC_NUM_RE = re.compile(r"(?P<op><=|>=|<|>)?\s*(?P<val>[0-9.]+)\s*nm", re.IGNORECASE)


@dataclass
class ADSCRawParser:
    """Parse ADS-C decoded text and keep MVP-required fields."""

    flight_id_col: str = "flight_id"

    @staticmethod
    def _parse_ts(raw: str) -> datetime | None:
        raw = raw.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_acc_nm(text: str | None) -> float | None:
        if not text:
            return None
        match = ACC_NUM_RE.search(text)
        if not match:
            return None
        try:
            return float(match.group("val"))
        except ValueError:
            return None

    def parse_file(self, path: str) -> pd.DataFrame:
        rows: list[dict] = []
        current_icao: str | None = None
        current_message: dict | None = None
        current_tag: int | None = None

        def flush() -> None:
            nonlocal current_message
            if not current_message:
                return
            ts = current_message.get("timestamp")
            lat = current_message.get("lat")
            lon = current_message.get("lon")
            alt = current_message.get("alt")
            if ts is not None and lat is not None and lon is not None and alt is not None:
                rows.append(current_message)
            current_message = None

        with open(path, "r", encoding="utf-8", errors="ignore") as fin:
            for raw_line in fin:
                line = raw_line.rstrip("\n")

                match_icao = ICAO_RE.search(line)
                if match_icao:
                    current_icao = match_icao.group(1).lower()

                match_tag = TAG_HEADER_RE.match(line)
                if match_tag:
                    tag_no = int(match_tag.group(1))
                    current_tag = tag_no
                    if tag_no == 7:
                        flush()
                        current_message = {
                            self.flight_id_col: current_icao,
                            "timestamp": None,
                            "lat": None,
                            "lon": None,
                            "alt": None,
                            "position_accuracy": None,
                            "tag13_exists": 0,
                            "tag14_exists": 0,
                            "tag15_exists": 0,
                            "tag16_exists": 0,
                            "tag07_exists": 1,
                        }
                    elif current_message and tag_no in (13, 14, 15, 16):
                        current_message[f"tag{tag_no}_exists"] = 1
                    continue

                if not current_message:
                    continue

                # Strict anchoring rule:
                # only parse anchor timestamp/position/altitude from Tag 7 body.
                # Tag 13 ETA (and other tags) must never overwrite anchor timestamp.
                if current_tag != 7:
                    if not line.strip():
                        flush()
                        current_tag = None
                    continue

                if (m := LAT_RE.match(line)):
                    current_message["lat"] = float(m.group(1))
                    continue
                if (m := LON_RE.match(line)):
                    current_message["lon"] = float(m.group(1))
                    continue
                if (m := ALT_RE.match(line)):
                    current_message["alt"] = float(m.group(1)) * 0.3048
                    continue
                if (m := TS_RE.match(line)):
                    current_message["timestamp"] = self._parse_ts(m.group(1))
                    continue
                if (m := ACC_RE.match(line)):
                    current_message["position_accuracy"] = self._parse_acc_nm(m.group(1))
                    continue

                if not line.strip():
                    flush()
                    current_tag = None

        flush()

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.dropna(subset=[self.flight_id_col, "timestamp", "lat", "lon", "alt"]).copy()
        out = out.sort_values([self.flight_id_col, "timestamp"]).reset_index(drop=True)
        return out
