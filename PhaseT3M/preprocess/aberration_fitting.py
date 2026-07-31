# Juhyeok Lee, LBNL, 2026.
#
# CTF / aberration fitting for HRTEM images (defocus C1 + two-fold astigmatism A1).
#
# Method
# ------
# The phase chi(q) is built from PhaseT3M's own `aberrations_basis_function`, so the
# fitted coefficients (C10, C12a, C12b, C30) already follow the PhaseT3M aberration
# convention used by the reconstruction code in `PhaseT3M.process`:
#     C10 = -defocus
#     C12a, C12b -> A1_magnitude = sqrt(C12a^2 + C12b^2), A1_angle = 0.5*atan2(C12b, C12a)
#
# The fit is done against the (background-subtracted) tile-averaged power spectrum
# ("diffractogram") of the image, matching CTF^2 Thon rings to the model.
#
# Sequential search (isotropic 1D defocus scan -> 2D A1 grid) fails once A1 is large:
# the elliptical rings no longer match the isotropic model, so the 1D scan locks onto
# the wrong local optimum and the local optimizer can't escape it. The default
# `method="sectors"` avoids this by fitting defocus independently and globally in each
# azimuthal wedge of the diffractogram. Within a wedge the local defocus is nearly
# isotropic even for large A1, so each 1D global search is robust; the per-sector
# defocus then obeys df(theta) = df_mean - C12a*cos(2*theta) - C12b*sin(2*theta), which
# is solved by linear least squares to get defocus, A1 magnitude and A1 angle in one
# shot, without needing an initial guess and without getting trapped in a local optimum.
#
# For the extreme regime (small defocus + large astigmatism), the local defocus per
# sector can even change sign (over/underfocus) and the CC landscape becomes highly
# multimodal, so both the sequential and sector seeds can fail. When the resulting CC is
# below `cc_rescue_threshold`, `rescue=True` reruns a global grid search over
# (df_mean, A1_magnitude, A1_angle) followed by a polish step to recover the global
# optimum; this only triggers when needed, so well-behaved cases stay fast.

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from PhaseT3M.process.utils import (
    aberrations_basis_function,
    electron_wavelength_angstrom,
    spatial_frequencies,
)


def compute_diffractogram(image, tile_size=512, step=None, window=True):
    """Tile-averaged power spectrum ("diffractogram") of an image, unshifted (DC at [0,0])."""
    img = np.asarray(image, float)
    H, W = img.shape
    tile = min(tile_size, H, W)
    step = step or tile // 2
    win = np.outer(np.hanning(tile), np.hanning(tile)) if window else 1.0
    acc = np.zeros((tile, tile))
    n = 0
    for y in (list(range(0, H - tile + 1, step)) or [0]):
        for x in (list(range(0, W - tile + 1, step)) or [0]):
            t = (img[y:y+tile, x:x+tile] - img[y:y+tile, x:x+tile].mean()) * win
            acc += np.abs(np.fft.fft2(t))**2
            n += 1
    return acc / max(n, 1)


def _parse_initial(initial):
    if initial is None:
        return None
    d = dict(initial)
    C1 = -d["defocus"] if "defocus" in d else d.get("C1", None)
    if "A1a" in d and "A1b" in d:
        A1a, A1b = d["A1a"], d["A1b"]
    elif "A1_magnitude" in d:
        az = np.radians(2*d.get("A1_angle_deg", 0.0))
        m = d["A1_magnitude"]
        A1a, A1b = m*np.cos(az), m*np.sin(az)
    else:
        A1a, A1b = 0.0, 0.0
    return (C1, A1a, A1b)


def fit_C1_A1(image, pixel_size, kV, Cs_mm=2.3, amplitude_contrast=0.07, tile_size=512,
              min_res=30.0, max_res=3.0, df_range=(3000., 60000.), df_step=500., bg_sigma=12.0,
              initial=None, coarse_search=True, local_window=4000.0, method="auto", n_sectors=12,
              # --- speed ---
              max_fit_points=30000, seed_grid=(14, 16),
              # --- fine tuning ---
              fit_Cs=False, fit_phase=False, fix=None,
              polish_xatol=1e-2, polish_fatol=1e-6, refine_iters=1,
              auto_resolution=True, n_target_rings=6, nyquist_frac=0.9,
              rescue=True, cc_rescue_threshold=0.35,
              rescue_ndf=22, rescue_na=18, rescue_nang=24, rescue_df_max=6000.0, rescue_A_max=2600.0,
              # --- seed search ranges (reuses df_range for defocus) ---
              A1_mag_range=None, A1_angle_range=None,
              # --- hard_bounds=True: constrains not only the seed search but also the
              #     final Nelder-Mead polish to df_range/A1_mag_range/A1_angle_range ---
              hard_bounds=False,
              return_diagnostics=False):
    """Fit defocus (C1) and two-fold astigmatism (A1) from the Thon rings of an HRTEM image.

    Returns a dict with defocus, A1_magnitude, A1_angle_deg, cc, and (if
    return_diagnostics=True) the diffractogram/basis needed by `show_ctf_fit`. See the
    module docstring for the fitting strategy (method="sectors"/"auto"/"sequential",
    rescue, hard_bounds).
    """
    fix = set(fix or [])
    dg = compute_diffractogram(image, tile_size=tile_size)
    N = dg.shape[0]
    energy = kV*1e3
    w = np.float32(amplitude_contrast)
    a = np.float32(np.sqrt(1 - amplitude_contrast**2))
    basis, mn = aberrations_basis_function((N, N), (pixel_size, pixel_size), energy,
                                            max_angular_order=2, min_radial_order=1, max_radial_order=4)
    idx = {tuple(r): i for i, r in enumerate(mn)}
    iC1, iCs, iA1a, iA1b = idx[(1, 0, 0)], idx[(3, 0, 0)], idx[(1, 2, 0)], idx[(1, 2, 1)]
    C30_0 = Cs_mm*1e7
    qx, qy = spatial_frequencies((N, N), (pixel_size, pixel_size))
    q = np.sqrt(qx[:, None]**2 + qy[None, :]**2)
    phi = np.arctan2(qy[None, :], qx[:, None])
    q_flat = q.ravel()
    phi_flat = phi.ravel()
    ps = np.log1p(dg - dg.min())
    base_exp = (ps - gaussian_filter(ps, bg_sigma)).ravel()
    lam = electron_wavelength_angstrom(energy)

    def build_band(min_r, max_r):
        m = np.where((q_flat >= 1.0/min_r) & (q_flat <= 1.0/max_r))[0]
        if max_fit_points and m.size > max_fit_points:
            m = np.random.default_rng(0).choice(m, max_fit_points, replace=False)
        Bm = basis[m].astype(np.float32)
        e = base_exp[m].astype(np.float32)
        em = e - e.mean()
        return dict(Bm=Bm, cC1=Bm[:, iC1], cA1a=Bm[:, iA1a], cA1b=Bm[:, iA1b], cCs=Bm[:, iCs],
                    phim=np.mod(phi_flat[m], np.pi).astype(np.float32),
                    exp=e, exp_m=em, exp_norm=np.float32(np.linalg.norm(em) + 1e-12))

    max_res_used = max_res
    if auto_resolution:
        # pass-1 rough defocus on the user band, then widen the band for small defocus
        b0 = build_band(min_res, max_res)
        chi = (C30_0*b0["cCs"])[None, :] + (-np.arange(min(df_range[0], 300.), df_range[1]+df_step, df_step, dtype=np.float32))[:, None]*b0["cC1"][None, :]
        mm = (a*np.sin(chi) + w*np.cos(chi))
        mm = mm*mm
        mm -= mm.mean(1, keepdims=True)
        ccv = (mm @ b0["exp_m"]) / (np.linalg.norm(mm, axis=1)*b0["exp_norm"] + 1e-12)
        df_rough = float(np.arange(min(df_range[0], 300.), df_range[1]+df_step, df_step)[int(np.argmax(ccv))])
        qNyq = 1.0/(2*pixel_size)
        q_hi_user = 1.0/max_res
        q_want = np.sqrt(max(n_target_rings/(lam*max(df_rough, 1.0)), 0.0))
        q_hi = max(q_hi_user, min(nyquist_frac*qNyq, q_want))
        max_res_used = 1.0/q_hi

    B = build_band(min_res, max_res_used)
    cC1, cA1a, cA1b, cCs = B["cC1"], B["cA1a"], B["cA1b"], B["cCs"]
    phim = B["phim"]
    exp = B["exp"]
    exp_m = B["exp_m"]
    exp_norm = B["exp_norm"]

    def model_of(chi):
        m = (a*np.sin(chi) + w*np.cos(chi))
        m *= m
        return m

    def cc_from_chi(chi):
        model = model_of(chi)
        mm = model - model.mean()
        return float(mm @ exp_m / ((np.linalg.norm(mm) + 1e-12)*exp_norm))

    def cc(C1, A1a, A1b, C30=C30_0):
        chi = C30*cCs + np.float32(C1)*cC1 + np.float32(A1a)*cA1a + np.float32(A1b)*cA1b
        return cc_from_chi(chi)

    df_lo = min(df_range[0], 300.0)
    dfs = np.arange(df_lo, df_range[1]+df_step, df_step, dtype=np.float32)
    chi_cs = C30_0*cCs

    def iso_defocus():
        chi = chi_cs[None, :] + (-dfs)[:, None]*cC1[None, :]
        model = model_of(chi)
        model -= model.mean(1, keepdims=True)
        ccv = (model @ exp_m) / (np.linalg.norm(model, axis=1)*exp_norm + 1e-12)
        return float(dfs[int(np.argmax(ccv))])

    def sector_seed():
        edges = np.linspace(0, np.pi, n_sectors+1)
        thc, dfk, wk = [], [], []
        for s in range(n_sectors):
            sel = (phim >= edges[s]) & (phim < edges[s+1])
            if sel.sum() < 20:
                continue
            e = exp[sel] - exp[sel].mean()
            en = np.linalg.norm(e) + 1e-12
            chi = chi_cs[sel][None, :] + (-dfs)[:, None]*cC1[sel][None, :]
            model = model_of(chi)
            model -= model.mean(1, keepdims=True)
            ccv = (model @ e) / (np.linalg.norm(model, axis=1)*en + 1e-12)
            k = int(np.argmax(ccv))
            thc.append(0.5*(edges[s]+edges[s+1]))
            dfk.append(float(dfs[k]))
            wk.append(max(float(ccv[k]), 1e-3))
        thc, dfk, wk = np.array(thc), np.array(dfk), np.array(wk)
        # Linear least squares fit of df(theta) = df_mean - C12a*cos(2*theta) - C12b*sin(2*theta)
        A = np.c_[np.ones_like(thc), np.cos(2*thc), np.sin(2*thc)]
        Wt = np.sqrt(wk)
        sol = np.linalg.lstsq(A*Wt[:, None], dfk*Wt, rcond=None)[0]
        resid = dfk - A@sol
        keep = np.abs(resid) <= 2.5*np.std(resid) + 1e-9
        if keep.sum() >= 4:
            sol = np.linalg.lstsq((A*Wt[:, None])[keep], (dfk*Wt)[keep], rcond=None)[0]
        dm, a2, b2 = sol
        return (-dm, -a2, -b2)

    def grid_seed(C1_fixed=None):
        df0 = (-C1_fixed) if C1_fixed is not None else iso_defocus()
        # A1 magnitude grid: use A1_mag_range if given, otherwise fall back to the
        # heuristic (0.25*df0 + 300)
        if A1_mag_range is not None:
            A_lo, A_hi = A1_mag_range
        else:
            A_lo, A_hi = 0.0, 0.25*df0 + 300.
        Av = np.linspace(A_lo, A_hi, seed_grid[0])
        # A1 angle grid: use A1_angle_range (degrees) if given, otherwise sweep the full
        # 0-180 deg range
        if A1_angle_range is not None:
            Th = np.linspace(np.radians(A1_angle_range[0]), np.radians(A1_angle_range[1]), seed_grid[1])
        else:
            Th = np.linspace(0, np.pi, seed_grid[1], endpoint=False)
        AA, TT = np.meshgrid(Av, Th, indexing='ij')
        ca = (AA*np.cos(2*TT)).ravel().astype(np.float32)
        cb = (AA*np.sin(2*TT)).ravel().astype(np.float32)
        chi0 = chi_cs + np.float32(-df0)*cC1
        chi = chi0[None, :] + ca[:, None]*cA1a[None, :] + cb[:, None]*cA1b[None, :]  # vectorised grid
        model = model_of(chi)
        model -= model.mean(1, keepdims=True)
        ccv = (model @ exp_m) / (np.linalg.norm(model, axis=1)*exp_norm + 1e-12)
        k = int(np.argmax(ccv))
        return (-df0, float(ca[k]), float(cb[k]))

    init = _parse_initial(initial)
    if init is not None and init[0] is not None and not coarse_search:
        seeds = [init] if (init[1] or init[2]) else [grid_seed(C1_fixed=init[0])]
    else:
        seeds = []
        if method in ("auto", "sectors"):
            seeds.append(sector_seed())
        if method in ("auto", "sequential"):
            seeds.append(grid_seed())
        if init is not None and init[0] is not None:
            seeds.append(init)
        if not seeds:
            seeds = [grid_seed()]

    # ---- flexible polish over [C1,A1a,A1b] + optional [C30]+[phase], with 'fix' ----
    # hard_bounds=True: reparameterize A1 as polar (mag, angle) and enforce bounds via
    # scipy's bounded Nelder-Mead.
    names = ["C1", "A1a", "A1b"] + (["C30"] if fit_Cs else []) + (["phase"] if fit_phase else [])
    names_polar = ["C1", "A1_mag", "A1_angle"] + (["C30"] if fit_Cs else []) + (["phase"] if fit_phase else [])
    STEP = {"C1": 500.0, "A1a": 300.0, "A1b": 300.0, "A1_mag": 300.0, "A1_angle": 10.0,
            "C30": 1.0e6, "phase": 0.2}  # simplex steps (avoid 0-start trap)

    def make_obj(seed):
        if not hard_bounds:
            base = dict(C1=seed[0], A1a=seed[1], A1b=seed[2], C30=C30_0, phase=0.0)
            free = [n for n in names if n not in fix]
            x0 = [base[n] for n in free]

            def unpack(x):
                d = dict(base)
                for n, val in zip(free, x):
                    d[n] = val
                return d

            def obj(x):
                d = unpack(x)
                chi = (d["C30"]*cCs + np.float32(d["C1"])*cC1 + np.float32(d["A1a"])*cA1a
                       + np.float32(d["A1b"])*cA1b + np.float32(d["phase"]))
                return -cc_from_chi(chi)
            return obj, x0, unpack, free, None
        # --- hard_bounds: polar (C1, A1_mag, A1_angle) parameterization, clipped/bounded
        #     by df_range/A1_mag_range/A1_angle_range ---
        C1_0, A1a_0, A1b_0 = seed[0], seed[1], seed[2]
        mag0 = float(np.clip(np.hypot(A1a_0, A1b_0), *(A1_mag_range if A1_mag_range is not None else (0.0, np.inf))))
        ang0 = float(0.5*np.degrees(np.arctan2(A1b_0, A1a_0)))
        if A1_angle_range is not None:
            ang0 = float(np.clip(ang0, A1_angle_range[0], A1_angle_range[1]))
        C1_0 = float(np.clip(C1_0, -df_range[1], -df_range[0]))
        base = dict(C1=C1_0, A1_mag=mag0, A1_angle=ang0, C30=C30_0, phase=0.0)
        free = [n for n in names_polar if n not in fix]
        x0 = [base[n] for n in free]
        bnd_map = {"C1": (-df_range[1], -df_range[0]),
                   "A1_mag": A1_mag_range if A1_mag_range is not None else (0.0, None),
                   "A1_angle": A1_angle_range if A1_angle_range is not None else (None, None),
                   "C30": (None, None), "phase": (None, None)}
        bounds = [bnd_map[n] for n in free]

        def unpack(x):
            d = dict(base)
            for n, val in zip(free, x):
                d[n] = val
            az = np.radians(2*d["A1_angle"])
            d["A1a"] = d["A1_mag"]*np.cos(az)
            d["A1b"] = d["A1_mag"]*np.sin(az)
            return d

        def obj(x):
            d = unpack(x)
            chi = (d["C30"]*cCs + np.float32(d["C1"])*cC1 + np.float32(d["A1a"])*cA1a
                   + np.float32(d["A1b"])*cA1b + np.float32(d["phase"]))
            return -cc_from_chi(chi)
        return obj, x0, unpack, free, bounds

    def _polish(x0, free, bounds, maxiter):
        x0a = np.asarray(x0, float)
        sim = np.vstack([x0a] + [x0a + STEP[free[i]]*np.eye(len(x0a))[i] for i in range(len(x0a))])
        if bounds is not None:
            for j, (lo, hi) in enumerate(bounds):
                if lo is not None:
                    sim[:, j] = np.maximum(sim[:, j], lo)
                if hi is not None:
                    sim[:, j] = np.minimum(sim[:, j], hi)
        kw = dict(initial_simplex=sim, xatol=polish_xatol, fatol=polish_fatol, maxiter=maxiter)
        return x0a, kw

    best = None
    best_d = None
    for s in seeds:
        obj, x0, unpack, free, bounds = make_obj(s)
        if not free:  # everything fixed -> just evaluate
            d = unpack([])
            val = obj([])
            if best is None or val < best:
                best, best_d = val, d
            continue
        for _ in range(max(1, refine_iters)):
            x0a, kw = _polish(x0, free, bounds, 4000)
            res = minimize(obj, x0a, method="Nelder-Mead", bounds=bounds, options=kw)
            x0 = res.x
        d = unpack(res.x)
        if best is None or res.fun < best:
            best, best_d = res.fun, d

    # ---- rescue: low CC often = small defocus + large A1 (very multimodal). ----
    # Global grid over (df_mean, A1_mag, angle) on the current band, then polish.
    if rescue and (-best) < cc_rescue_threshold:
        chi_cs = C30_0*cCs
        if A1_angle_range is not None:
            Tg = np.linspace(np.radians(A1_angle_range[0]), np.radians(A1_angle_range[1]), rescue_nang)
        else:
            Tg = np.linspace(0, np.pi, rescue_nang, endpoint=False)
        c2 = np.cos(2*Tg)
        s2 = np.sin(2*Tg)
        dfg = np.linspace(50.0, min(df_range[1], rescue_df_max), rescue_ndf)
        if A1_mag_range is not None:
            Ag = np.linspace(A1_mag_range[0], A1_mag_range[1], rescue_na)
        else:
            Ag = np.linspace(0.0, rescue_A_max, rescue_na)
        topcc, topseed = -9.0, None
        for dfv in dfg:
            base = chi_cs + np.float32(-dfv)*cC1
            for A_ in Ag:
                ca = (A_*c2).astype(np.float32)
                cb = (A_*s2).astype(np.float32)
                chi = base[None, :] + ca[:, None]*cA1a[None, :] + cb[:, None]*cA1b[None, :]
                model = (a*np.sin(chi) + w*np.cos(chi))
                model *= model
                model -= model.mean(1, keepdims=True)
                ccv = (model @ exp_m) / (np.linalg.norm(model, axis=1)*exp_norm + 1e-12)
                k = int(np.argmax(ccv))
                if ccv[k] > topcc:
                    topcc, topseed = float(ccv[k]), (-dfv, float(ca[k]), float(cb[k]))
        obj, x0, unpack, free, bounds = make_obj(topseed)
        x0a, kw = _polish(x0, free, bounds, 6000)
        rr = minimize(obj, x0a, method="Nelder-Mead", bounds=bounds, options=kw)
        if rr.fun < best:
            best, best_d = rr.fun, unpack(rr.x)

    C1, A1a, A1b, C30 = best_d["C1"], best_d["A1a"], best_d["A1b"], best_d["C30"]
    out = {"C1": float(C1), "defocus": float(-C1),
           "A1_magnitude": float(np.hypot(A1a, A1b)),
           "A1_angle_deg": float(0.5*np.degrees(np.arctan2(A1b, A1a))),
           "A1a": float(A1a), "A1b": float(A1b), "phase_shift": float(best_d.get("phase", 0.0)),
           "Cs_mm": float(C30/1e7), "cc": float(-best),
           "phaset3m_coeffs": {"C10": float(C1), "C12a": float(A1a), "C12b": float(A1b), "C30": float(C30)}}
    out["max_res_used"] = float(max_res_used)
    if return_diagnostics:
        out.update(diffractogram=dg, basis=basis, mn=mn, n_fit_points=len(exp))
    return out


def report(r):
    """Print a short summary of a `fit_C1_A1` result."""
    print(f"  C1 (defocus)   : {r['defocus']:12.1f} Å   [single value, C10 = {r['C1']:.1f}]")
    print(f"  A1 magnitude   : {r['A1_magnitude']:12.1f} Å")
    print(f"  A1 angle       : {r['A1_angle_deg']:12.1f} deg")
    print(f"  CC             : {r['cc']:.3f}   (max_res_used = {r.get('max_res_used', float('nan')):.2f} Å)")


def _radial_whiten(imgc):
    """Divide each pixel by the RMS of its radius -> equalises ring contrast at all radii."""
    N = imgc.shape[0]
    c = N // 2
    yy, xx = np.mgrid[0:N, 0:N] - c
    rr = np.round(np.sqrt(xx**2 + yy**2)).astype(int).ravel()
    ss = np.bincount(rr, (imgc.ravel()**2))
    cnt = np.bincount(rr)
    rms = np.sqrt(ss / np.maximum(cnt, 1))
    rms = np.maximum(rms, rms[rms > 0].mean()*0.05)
    return (imgc.ravel() / rms[rr]).reshape(imgc.shape)


def show_ctf_fit(r, pixel_size, display_res=None, zoom=1.30,
                  radial_norm=True, w=0.07, clip=3.0, figsize=(11, 5.4)):
    """Smart diffractogram/fit overlay: crop to the ring region + radial whitening.

    Requires `r` to come from `fit_C1_A1(..., return_diagnostics=True)`.
    `display_res` crops to that resolution (in Angstrom); otherwise crops to
    `zoom` times the fitted band.
    """
    dg, basis, mn = r["diffractogram"], r["basis"], r["mn"]
    idx = {tuple(row): i for i, row in enumerate(mn)}
    v = np.zeros(basis.shape[1])
    v[idx[(1, 0, 0)]] = r["C1"]
    v[idx[(1, 2, 0)]] = r["A1a"]
    v[idx[(1, 2, 1)]] = r["A1b"]
    v[idx[(3, 0, 0)]] = r["phaset3m_coeffs"]["C30"]
    chi = (basis @ v).reshape(dg.shape)
    model = (-(np.sqrt(1-w**2)*np.sin(chi) + w*np.cos(chi)))**2

    ps = np.log1p(dg - dg.min())
    ed = ps - gaussian_filter(ps, 12)
    if radial_norm:
        ed = _radial_whiten(ed)
        model = _radial_whiten(model - model.mean())
    ed = np.fft.fftshift(ed)
    mdl = np.fft.fftshift(model)

    N = dg.shape[0]
    c = N // 2
    qNyq = 1.0 / (2*pixel_size)
    # crop radius (pixels): out to a bit beyond the fitted band
    q_fit = 1.0 / r.get("max_res_used", 3.0)
    q_disp = (1.0/display_res) if display_res else min(zoom*q_fit, qNyq)
    rad = int(np.clip(q_disp/qNyq*c, 16, c))
    sl = slice(c-rad, c+rad)
    ed_c, mdl_c = ed[sl, sl], mdl[sl, sl]

    def norm(arr):
        arr = arr - np.median(arr)
        s = arr.std() + 1e-9
        return np.clip(arr/s, -clip, clip)
    ed_c, mdl_c = norm(ed_c), norm(mdl_c)
    half = ed_c.shape[1] // 2
    ext = [-q_disp, q_disp, -q_disp, q_disp]

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(ed_c, cmap="gray", origin="lower", extent=ext)
    ax[0].set_title(f"Diffractogram (zoom to {1/q_disp:.1f} Å)")
    combo = np.hstack([ed_c[:, :half], mdl_c[:, half:]])
    ax[1].imshow(combo, cmap="gray", origin="lower", extent=ext)
    ax[1].axvline(0, color="tab:red", lw=1, ls="--")
    ax[1].set_title("Left: experiment | Right: fitted CTF$^2$")
    # resolution rings
    for res in [8, 5, 4, 3]:
        qr = 1.0/res
        if qr < q_disp*0.98:
            for a in ax:
                a.add_patch(plt.Circle((0, 0), qr, fill=False, color="deepskyblue", lw=0.6, alpha=0.6))
            ax[0].text(0, qr, f"{res}Å", color="deepskyblue", fontsize=7, ha="center", va="bottom")
    for a in ax:
        a.set_xlabel("q (1/Å)")
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    return fig


def load_hrtem_image(path):
    """Load an HRTEM image from .mrc/.mrcs, .tif/.tiff, or .npy. Averages over frames if 3D."""
    p = str(path).lower()
    if p.endswith((".mrc", ".mrcs")):
        import mrcfile
        with mrcfile.open(path, permissive=True) as m:
            arr = np.asarray(m.data, float)
    elif p.endswith((".tif", ".tiff")):
        import tifffile
        arr = np.asarray(tifffile.imread(path), float)
    elif p.endswith(".npy"):
        arr = np.load(path).astype(float)
    else:
        raise ValueError("Supported formats: .mrc/.mrcs, .tif/.tiff, .npy")
    return arr.mean(0) if arr.ndim == 3 else arr


def make_hrtem_image(defocus, A1_mag, A1_angle_deg, N=1024, px=1.0, kV=300,
                      Cs_mm=2.3, w=0.07, sigmaV=0.06, dose=200, seed=0):
    """Simulate a noisy HRTEM image with a given defocus/A1, for testing `fit_C1_A1`."""
    rng = np.random.default_rng(seed)
    basis, mn = aberrations_basis_function((N, N), (px, px), kV*1e3,
                                            max_angular_order=2, min_radial_order=1, max_radial_order=4)
    idx = {tuple(r): i for i, r in enumerate(mn)}
    v = np.zeros(basis.shape[1])
    az = np.radians(2*A1_angle_deg)
    v[idx[(1, 0, 0)]] = -defocus
    v[idx[(1, 2, 0)]] = A1_mag*np.cos(az)
    v[idx[(1, 2, 1)]] = A1_mag*np.sin(az)
    v[idx[(3, 0, 0)]] = Cs_mm*1e7
    chi = (basis @ v).reshape(N, N)
    psi = np.exp(1j*sigmaV*rng.normal(0, 1, (N, N)))
    T = np.exp(-1j*chi)
    T = np.sqrt(1-w**2)*T + w*T.real
    img = np.abs(np.fft.ifft2(np.fft.fft2(psi)*T))**2
    if dose:
        img = rng.poisson(np.clip(img*dose, 0, None)) / dose
    return img


def rotate_A1(A1a, A1b, rot_angle_deg):
    """Rotate A1 (C12a, C12b) to match an image rotated by `rot_angle_deg`
    (e.g. via `rotate_fourier_shear(image, rot_angle_deg)`)."""
    psi = np.radians(2 * rot_angle_deg)  # A1 is a two-fold (spin-2) term
    c, s = np.cos(psi), np.sin(psi)
    A1a_new = A1a * c - A1b * s
    A1b_new = A1a * s + A1b * c
    return A1a_new, A1b_new
