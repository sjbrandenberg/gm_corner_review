"""Source-to-site distances (Rrup, Rjb, Rx, Ry0) from NGA finite-fault segments.

A segment is a rectangle given by its upper-left corner (ULC_latitude,
ULC_longitude, ULC_depth km), seg_length and seg_width (km), seg_strike and
seg_dip (deg). The top edge runs from the ULC along strike; the bottom edge is
offset down-dip (strike + 90 deg).

Geometry is evaluated in a site-centred azimuthal-equidistant frame: each fault corner is mapped to (x, y) = d * (sin az, cos az)
using the geodesic distance d and azimuth az from the site, with depth kept
vertical. Distances from the site to fault corners are therefore true
geodesic (surface) distances, "down" is the same direction at the fault and
the site, and Rrup >= Rjb holds by construction. (Earth-centred Cartesian
coordinates instead give chord distances and a tilted vertical, which made
Rrup < Rjb beyond ~100 km.) The vectorized point-to-triangle and Rx/Ry0
routines come from ucla_plha.

Earth model: a sphere of radius 6371 km by default, which reproduces the
nga-gm.org flatfile Rjb/Rrup to a median of 2-18 m (max 0.17 km, Ridgecrest)
for 1,987 stations; WGS84 geodesics differ from the database by up to ~1 km
at 300-500 km. Pass earth=ELLIPSOID for true ellipsoidal distances.
"""

import numpy as np
from pyproj import Geod
from ucla_plha.geometry import geometry as geo

SPHERE = Geod(a=6371000.0, b=6371000.0)
ELLIPSOID = Geod(ellps="WGS84")


def segment_corners(seg, earth=SPHERE):
    """Return 4 x 3 array of [lat, lon, depth_km]: top-start, top-end, bottom-start, bottom-end."""
    lat0, lon0, z0 = seg["ULC_latitude"], seg["ULC_longitude"], seg["ULC_depth"]
    L, W = seg["seg_length"], seg["seg_width"]
    stk, dip = seg["seg_strike"], np.radians(seg["seg_dip"])
    h = W * np.cos(dip)  # horizontal down-dip offset
    lon2, lat2, _ = earth.fwd(lon0, lat0, stk, L * 1000.0)
    lon3, lat3, _ = earth.fwd(lon0, lat0, stk + 90.0, h * 1000.0)
    lon4, lat4, _ = earth.fwd(lon2, lat2, stk + 90.0, h * 1000.0)
    zb = z0 + W * np.sin(dip)
    return np.array([[lat0, lon0, z0], [lat2, lon2, z0], [lat3, lon3, zb], [lat4, lon4, zb]])


def fault_geometry(segments, earth=SPHERE):
    """Corner coordinates for a list of segment dicts (projection happens per site)."""
    corners = np.asarray([segment_corners(s, earth) for s in segments])  # n_seg x 4 x 3
    return {"corners": corners, "n_seg": len(segments), "earth": earth}


def _project(corners, lat, lon, earth):
    """Site-centred azimuthal-equidistant x, y (km) and depth (km) of corners."""
    flat = corners.reshape(-1, 3)
    az, _, d = earth.inv(np.full(len(flat), lon), np.full(len(flat), lat), flat[:, 1], flat[:, 0])
    az = np.radians(az)
    d = d / 1000.0
    xyz = np.column_stack([d * np.sin(az), d * np.cos(az), -flat[:, 2]])
    return xyz.reshape(corners.shape)


def event_distances(segments, lats, lons, earth=SPHERE):
    """Rrup, Rjb (site-centred frames) and GC2 Rx, Ry0, U (fault-centred frame)
    for arrays of site coordinates. Returns a dict of numpy arrays."""
    from . import gc2

    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    n = len(lats)
    out = {k: np.full(n, np.nan) for k in ("rrup", "rjb", "rx", "ry0", "gc2_u")}
    out["sub_rupture"] = np.full(n, -1)
    ok = np.flatnonzero(np.isfinite(lats) & np.isfinite(lons))
    if not len(ok):
        return out

    corners = fault_geometry(segments, earth)["corners"]
    clat, clon = corners[:, :2, 0].mean(), corners[:, :2, 1].mean()
    starts = _project(corners[:, [0], :], clat, clon, earth)[:, 0, :2]
    ends = _project(corners[:, [1], :], clat, clon, earth)[:, 0, :2]
    pts = np.column_stack([lats[ok], lons[ok], np.zeros(len(ok))])[:, None, :]
    xy = _project(pts, clat, clon, earth)[:, 0, :2]

    sub = np.array([s.get("seg_sub_rupture_number") or 1 for s in segments])
    best = np.full(len(ok), np.inf)
    for k in np.unique(sub):
        m = sub == k
        g = fault_geometry([s for s, keep in zip(segments, m) if keep], earth)
        d = [site_distances(g, lats[i], lons[i]) for i in ok]
        rrup = np.array([x["rrup"] for x in d])
        rjb = np.array([x["rjb"] for x in d])
        rx, ry0, u = gc2.rx_ry0(gc2.setup(starts[m], ends[m]), xy)
        use = rrup < best                      # this sub-rupture is the closest so far
        best[use] = rrup[use]
        idx = ok[use]
        out["rx"][idx], out["ry0"][idx], out["gc2_u"][idx] = rx[use], ry0[use], u[use]
        out["sub_rupture"][idx] = k
        out["rrup"][ok] = np.fmin(out["rrup"][ok], rrup)
        out["rjb"][ok] = np.fmin(out["rjb"][ok], rjb)
    return out


def site_distances(geom, lat, lon, elev_km=0.0):
    """Rrup, Rjb, Rx, Ry0 (km) for one site. Rx/Ry0 here are relative to the
    closest segment; use event_distances() for GC2 Rx/Ry0."""
    P = _project(geom["corners"], lat, lon, geom["earth"])            # n_seg x 4 x 3
    S = P.copy()
    S[:, :, 2] = 0.0                                   # surface projection
    tri_idx = ([0, 1, 2], [1, 3, 2])
    tri_rup = np.concatenate([P[:, list(t)] for t in tri_idx])
    tri_jb = np.concatenate([S[:, list(t)] for t in tri_idx])
    tri_id = np.tile(np.arange(geom["n_seg"]), 2)
    site = np.array([0.0, 0.0, elev_km])
    rrup_s = geo.point_triangle_distance(tri_rup, site, tri_id)
    rjb_s = geo.point_triangle_distance(tri_jb, np.zeros(3), tri_id)
    _, _, ry0_s = geo.get_Rx_Rx1_Ry0(S, np.zeros(3), np.arange(geom["n_seg"]))
    k = int(np.argmin(rrup_s))
    # Rx: signed perpendicular distance from the top-edge line, positive to the
    # right of strike (down-dip / hanging-wall side), as in NGA/GC2. ucla_plha's
    # |Rx| < |Rx1| sign test is ambiguous for sites off the fault ends.
    p1, p2 = S[k, 0, :2], S[k, 1, :2]
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    rx = float(np.dot(-p1, [u[1], -u[0]]))
    return {"rrup": float(rrup_s.min()), "rjb": float(rjb_s.min()),
            "rx": rx, "ry0": float(ry0_s[k])}
