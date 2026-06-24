from __future__ import annotations

import math


_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2 - _F)


def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = _A / math.sqrt(1 - _E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1 - _E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_ecef: tuple[float, float, float],
) -> tuple[float, float, float]:
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    dx = x - ref_ecef[0]
    dy = y - ref_ecef[1]
    dz = z - ref_ecef[2]

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def enu_to_ecef(
    east: float,
    north: float,
    up: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_ecef: tuple[float, float, float],
) -> tuple[float, float, float]:
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    x = ref_ecef[0] + dx
    y = ref_ecef[1] + dy
    z = ref_ecef[2] + dz
    return x, y, z


def ecef_to_wgs84(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)

    lat = math.atan2(z, p * (1 - _E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = _A / math.sqrt(1 - _E2 * sin_lat * sin_lat)
        alt = p / max(math.cos(lat), 1e-12) - n
        lat_new = math.atan2(z, p * (1 - _E2 * n / max(n + alt, 1e-12)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    n = _A / math.sqrt(1 - _E2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), 1e-12) - n
    return math.degrees(lat), math.degrees(lon), alt
