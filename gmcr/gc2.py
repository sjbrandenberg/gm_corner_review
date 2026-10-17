"""Generalized Coordinate System version 2 (GC2) of Spudich & Chiou (2015),
USGS Open-File Report 2015-1028, for Rx (= T) and Ry0 (from U).

For one trace made of straight pieces (segment top edges projected to the
surface):

1. a_hat: unit vector between the two most distant piece endpoints, oriented
   eastward.
2. Discordance: a piece whose vector projects negatively on a_hat (relative to
   the sign of the total) is reversed.
3. b_hat: unit vector of the sum of the re-oriented piece vectors (nominal
   strike); origin p0 = the a_hat end point that comes first along b_hat.
4. Pieces are chained in order along b_hat; piece i's offset is
   s_i = (start_0 - p0) . b_hat + sum of preceding piece lengths.
   (NGA segment numbers don't follow along-strike order, hence the sort.)
5. For a site, each piece gives local (t_i, u_i), t positive to the right of
   the piece direction, and weight
       w_i = (1/t_i)[atan((l_i - u_i)/t_i) - atan(-u_i/t_i)]
   (limit forms for t_i = 0). T = sum(w t)/sum(w), U = sum(w (u + s))/sum(w).
6. Ry0 = -U if U < 0, U - U_max if U > U_max, else 0, where U_max is the
   largest U of any piece endpoint.

Ruptures with several sub-ruptures (NGA seg_sub_rupture_number) are handled
by `gmcr.distances.event_distances`: each sub-rupture gets its own GC2 frame
and a site uses the sub-rupture with the smallest Rrup. That reproduces the
nga-gm.org flatfile T/U (median |error| 0.29 km in T, 0.17 km in U over all 21
multi-segment faults), whereas a single multi-trace frame for the whole
rupture does not.

All coordinates are planar km in a common fault-centred projection.
"""

import numpy as np

_EPS = 1e-9


def setup(starts, ends):
    """starts, ends: (n, 2) arrays of piece start/end points (km). Returns config dict."""
    starts = np.asarray(starts, float).copy()
    ends = np.asarray(ends, float).copy()
    pts = np.vstack([starts, ends])
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    beg, end = (pts[i], pts[j]) if pts[i, 0] <= pts[j, 0] else (pts[j], pts[i])
    a_vec = end - beg
    a_hat = a_vec / max(np.linalg.norm(a_vec), _EPS)

    e = (ends - starts) @ a_hat
    e_tot_sign = np.sign(e.sum()) or 1.0
    flip = np.sign(e) * e_tot_sign < 0
    starts[flip], ends[flip] = ends[flip].copy(), starts[flip].copy()
    b_vec = (ends - starts).sum(axis=0)
    b_hat = b_vec / max(np.linalg.norm(b_vec), _EPS)
    p0 = beg if a_vec @ b_hat >= 0 else end

    order = np.argsort(starts @ b_hat)
    starts, ends = starts[order], ends[order]
    lengths = np.linalg.norm(ends - starts, axis=1)
    s = (starts[0] - p0) @ b_hat + np.r_[0.0, np.cumsum(lengths)[:-1]]

    cfg = {"starts": starts, "ends": ends, "s": s, "b_hat": b_hat, "p0": p0}
    _, u_ends = coordinates(cfg, np.vstack([starts, ends]))
    cfg["u_max"] = float(np.nanmax(u_ends))
    return cfg


def coordinates(cfg, sites):
    """GC2 (T, U) for sites, an (m, 2) array of planar km coordinates."""
    sites = np.atleast_2d(np.asarray(sites, float))
    sum_w = np.zeros(len(sites))
    sum_wt = np.zeros(len(sites))
    sum_wu = np.zeros(len(sites))
    on_T = np.full(len(sites), np.nan)
    on_U = np.full(len(sites), np.nan)
    for a, b, s in zip(cfg["starts"], cfg["ends"], cfg["s"]):
        r = b - a
        l = np.linalg.norm(r)
        if l < _EPS:
            continue
        u_hat = r / l
        t_hat = np.array([u_hat[1], -u_hat[0]])          # right of piece direction
        rel = sites - a
        t = rel @ t_hat
        u = rel @ u_hat
        w = np.empty(len(sites))
        nz = np.abs(t) > _EPS
        w[nz] = (np.arctan((l - u[nz]) / t[nz]) - np.arctan(-u[nz] / t[nz])) / t[nz]
        z = ~nz
        off = z & ((u < -_EPS) | (u > l + _EPS))
        w[off] = 1.0 / (u[off] - l) - 1.0 / u[off]
        on = z & ~off                                     # site lies on this piece
        on_T[on] = 0.0
        on_U[on] = u[on] + s
        w[on] = 0.0
        sum_w += w
        sum_wt += w * t
        sum_wu += w * (u + s)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = sum_wt / sum_w
        U = sum_wu / sum_w
    hit = np.isfinite(on_T)
    T[hit], U[hit] = on_T[hit], on_U[hit]
    return T, U


def rx_ry0(cfg, sites):
    """(Rx, Ry0, U) for sites."""
    T, U = coordinates(cfg, sites)
    ry0 = np.where(U < 0, -U, np.where(U > cfg["u_max"], U - cfg["u_max"], 0.0))
    return T, ry0, U
