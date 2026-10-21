#!/usr/bin/env python3
"""Figure 1: model set-up and the Earth / Venus reference case (50 mm/yr, depth-dependent permeability, sills at 5 km).

  a, b  set-up schematic (axisymmetric domain, initial permeability, initial isotherms, sill pile, boundary conditions)
  c, d  temperature / crystallinity at 100 kyr, ductile hatching, isotherms and Darcy-velocity arrows
        (arrows averaged over the 11 post-injection frames up to 100 kyr)
  e, f  mobile melt, crystal mush and induced ductile volumes vs time, conduction-only controls (dashed),
        cumulative magma emplaced; inset in e: Earth mobile melt and its mean crystallinity, 160-200 kyr

Usage:  python fig1.py --data-dir <unzipped Zenodo record> [--out-dir figures]
"""
import os
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import figstyle as fs

# ------------------------------------------------------------------------------------------------ runs and constants
MAP_RUNS = {'Earth': ('earth_depth_dependent', 'E_R50_NC_MI_LH'), 'Venus': ('venus', 'V_R50_NC_MI_LH_2MYR')}
STACK_RUNS = {'Earth': [('earth_depth_dependent', 'E_R50_NC_MI_HR200_LH'), ('earth_depth_dependent', 'E_R50_NC_MI_LH')],
              'Venus': [('venus', 'V_R50_NC_MI_LH_2MYR')]}
COND_RUNS = {'Earth': ('conduction_only', 'EC_R50_NC_LH'), 'Venus': ('conduction_only', 'VC_R50_NC_LH_2MYR')}
SNAP_YEAR, EMPL_END, T_END = 100000, 200.0, 2000.0          # map time (yr); end of emplacement, axis end (kyr)
MESH_H = 15000.0
X_VIEW, Z_VIEW = (7.0, 23.0), (0.0, 12.0)
T_SOLIDUS, T_DUCTILE = 950.0, 550.0

TEMP_B = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
CHI5_B = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0001]
CHI5_C = ['#7f0010', fs.MOBILE_C, '#e0702e', fs.MUSH_C, '#f7dba6']
ISO_RED = '#b2182b'
X_ISO_LABEL = {200: 18.3, 400: 18.3, 600: 18.3, 800: 16.4}    # label column (km) for the host-rock isotherms
X_ISO_MAGMA = {1000: (15.0, 'bottom')}                          # 1000 C, labelled on the base of the body
VEL_NX, VEL_NZ, VEL_GATE_T, VEL_SMOOTH, VEL_REL_MIN, VEL_MELT_CLEAR = 22, 18, 900.0, 1.2, 0.05, 0.45


def chi_to_T(c):
    return 950.0 + 300.0 * (1.0 - c) ** (1.0 / 2.28)


def blended_temp_colors():
    """Host-rock colours on one blue -> cream -> red scale that runs into the crystallinity colours."""
    mids = [0.5 * (chi_to_T(min(a, 1.0)) + chi_to_T(min(b, 1.0))) for a, b in zip(CHI5_B[:-1], CHI5_B[1:])]
    anchors = [(10.0, '#1f3f6e'), (250.0, '#3f6fa6'), (550.0, '#9dbcd9'), (850.0, '#eee9df')] + sorted(zip(mids, CHI5_C))
    tv, cv = zip(*anchors)
    rgb = np.array([mcolors.to_rgb(c) for c in cv])
    tm = [0.5 * (a + b) for a, b in zip(TEMP_B[:-1], TEMP_B[1:])]
    return [tuple(np.interp(t, tv, rgb[:, k]) for k in range(3)) for t in tm]


TEMP_C = blended_temp_colors()


# ------------------------------------------------------------------------------------------------ data
def read_series(data, suite, run):
    """timeseries.csv -> dict of arrays; t in kyr, frames ordered pre < plain < post at each instant."""
    with open(os.path.join(data, 'data', suite, run, 'timeseries.csv')) as f:
        rows = list(csv.DictReader(f))
    order = {'pre': 0, 'plain': 1, 'post': 2}
    rows.sort(key=lambda r: (float(r['t_kyr']), order[r['frame']]))
    num = lambda k: np.array([float(r[k]) if r[k] != '' else np.nan for r in rows])
    return dict(t=num('t_kyr'), kind=np.array([order[r['frame']] for r in rows]), mob=num('mobile_melt_km3'),
                mush=num('crystal_mush_km3'), duct=num('induced_ductile_km3'), chi=num('mean_chi_mobile'))


def run_meta(data, suite, run):
    with open(os.path.join(data, 'data', suite, run, 'run_meta.json')) as f:
        return json.load(f)


def stack_series(data, planet):
    """Volumes over the planet's runs, each later run used only after the previous run's last frame."""
    out, t_end = [], -np.inf
    for suite, run in STACK_RUNS[planet]:
        s = read_series(data, suite, run)
        k = s['t'] > t_end + 1e-9
        out.append({key: v[k] for key, v in s.items()})
        t_end = s['t'][k][-1]
    return {key: np.concatenate([o[key] for o in out]) for key in out[0]}


def emplaced(s):
    """Cumulative emplaced magma (km3): initial pluton = mobile melt at t = 0; each sill = the jump in
    mobile melt between the pre and post frame of the first injection."""
    v0 = s['mob'][(s['t'] == 0.0)][0]
    t1 = s['t'][s['kind'] == 2].min()
    v1 = s['mob'][(s['t'] == t1) & (s['kind'] == 2)][0] - s['mob'][(s['t'] == t1) & (s['kind'] == 0)][0]
    ts = np.unique(s['t'][(s['kind'] == 2) & (s['t'] <= EMPL_END + 1e-9)])
    return np.r_[0.0, ts], v0 + v1 * np.arange(len(ts) + 1), v0, v1


def load_snapshot(data, suite, run, name):
    d = os.path.join(data, 'data', suite, run, 'snapshots')
    m, f = np.load(os.path.join(d, 'mesh.npz')), np.load(os.path.join(d, name))
    return dict(x=m['x_m'] / 1000.0, d=(MESH_H - m['y_m']) / 1000.0, tris=m['triangles'],
                cx=m['cell_x_m'] / 1000.0, cd=(MESH_H - m['cell_y_m']) / 1000.0,
                T=f['T_C'].astype(float), chi=np.clip(f['chi'].astype(float), 0, 1),
                u=f['darcy_vx_m_per_s'].astype(float), w=f['darcy_vy_m_per_s'].astype(float))


def velocity_grid(data, suite, run):
    """Darcy velocity averaged over the post frames up to SNAP_YEAR (all in snapshots/), on the arrow grid."""
    d = os.path.join(data, 'data', suite, run, 'snapshots')
    names = sorted((n for n in os.listdir(d) if n.endswith('_post.npz')), key=lambda n: int(n.split('_')[1]))
    names = [n for n in names if int(n.split('_')[1]) <= SNAP_YEAR][-11:]
    XG, DG = np.meshgrid(np.linspace(*X_VIEW, VEL_NX), np.linspace(*Z_VIEW, VEL_NZ))
    Us, Ws, Ts = [], [], []
    for n in names:
        s = load_snapshot(data, suite, run, n)
        pv = np.column_stack((s['cx'], s['cd']))
        Us.append(np.nan_to_num(griddata(pv, s['u'], (XG, DG), method='linear')))
        Ws.append(np.nan_to_num(griddata(pv, s['w'], (XG, DG), method='linear')))
        Ts.append(griddata(np.column_stack((s['x'], s['d'])), s['T'], (XG, DG), method='nearest'))
    U = gaussian_filter(np.mean(Us, axis=0), VEL_SMOOTH)
    W = gaussian_filter(np.mean(Ws, axis=0), VEL_SMOOTH)
    return XG, DG, U, W, np.hypot(U, W), np.mean(Ts, axis=0)


# ------------------------------------------------------------------------------------------------ maps (c, d)
def nice_speed(v):
    e = np.floor(np.log10(v)); m = v / 10 ** e
    return (5.0 if m >= 5 else 2.0 if m >= 2 else 1.0) * 10 ** e


def speed_label(v):
    e = int(np.floor(np.log10(v) + 1e-9)); m = v / 10 ** e
    return (r'$10^{%d}$ m s$^{-1}$' % e) if abs(m - 1) < 1e-6 else (r'$%g\times10^{%d}$ m s$^{-1}$' % (m, e))


def draw_isotherms(ax, tri, T, x_of_level, z_line=6, z_text=9):
    """Each isotherm labelled on its line at the given x (uppermost crossing, or lowermost for 'bottom'),
    the label following the local direction and the line broken under it."""
    ax.apply_aspect()
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    halo = [pe.Stroke(linewidth=fs.HALO_LW, foreground='white'), pe.Normal()]
    for lev, x_lab in x_of_level.items():
        x_lab, where = x_lab if isinstance(x_lab, tuple) else (x_lab, 'top')
        tmp = ax.tricontour(tri, T, levels=[lev])
        segs = [np.asarray(sg) for sg in tmp.allsegs[0] if len(sg) > 1]
        tmp.remove()
        best = None
        for k, sg in enumerate(segs):
            for j, ((x0, z0), (x1, z1)) in enumerate(zip(sg[:-1], sg[1:])):
                if (x0 - x_lab) * (x1 - x_lab) <= 0 and x0 != x1:
                    z = z0 + (z1 - z0) * (x_lab - x0) / (x1 - x0)
                    if Z_VIEW[0] + 0.3 <= z <= Z_VIEW[1] - 0.3 and (
                            best is None or (z > best[1] if where == 'bottom' else z < best[1])):
                        best = (k, z, j, (x_lab - x0) / (x1 - x0))
        cut = {}
        if best is not None:
            k, z, j, f = best
            sg = segs[k]
            arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(sg, axis=0).T))]
            s_c = arc[j] + f * (arc[j + 1] - arc[j])
            near = sg[np.abs(arc - s_c) < 0.45]
            dvec = np.linalg.svd(near - near.mean(axis=0))[2][0] if len(near) > 2 else np.array([1.0, 0.0])
            dvec = dvec if dvec[0] >= 0 else -dvec
            q0, q1 = ax.transData.transform([(x_lab, z), (x_lab + dvec[0], z + dvec[1])])
            t = ax.text(x_lab, z, '%d °C' % lev, ha='center', va='center', fontsize=7, color=ISO_RED,
                        rotation_mode='anchor', zorder=z_text, path_effects=fs.halo())
            w_px = t.get_window_extent(renderer).width
            t.set_rotation(np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0])))
            half = (0.5 * w_px + 1.2 / 72.0 * fig.dpi) / np.hypot(*(q1 - q0))
            cut[k] = (s_c - half, s_c + half)
        for k, sg in enumerate(segs):
            parts = [sg]
            if k in cut:
                arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(sg, axis=0).T))]
                a, b = cut[k]
                pa = [np.interp(a, arc, sg[:, 0]), np.interp(a, arc, sg[:, 1])]
                pb = [np.interp(b, arc, sg[:, 0]), np.interp(b, arc, sg[:, 1])]
                parts = [np.vstack([sg[arc < a], pa]), np.vstack([pb, sg[arc > b]])]
            for part in parts:
                if len(part) > 1:
                    ax.plot(part[:, 0], part[:, 1], color=ISO_RED, lw=0.8, zorder=z_line,
                            solid_capstyle='butt', path_effects=halo)


def render_map(ax, S, vg, ref):
    """Host rock coloured by T, magma (any triangle node above the solidus) by crystallinity, ductile hatched,
    arrows scaled to this panel's reference speed. Returns the quiver."""
    T, chi = S['T'], S['chi']
    melt_tri = T[S['tris']].max(axis=1) >= T_SOLIDUS
    tri = mtri.Triangulation(S['x'], S['d'], S['tris'])
    triT = mtri.Triangulation(S['x'], S['d'], S['tris']); triT.set_mask(melt_tri)
    triC = mtri.Triangulation(S['x'], S['d'], S['tris']); triC.set_mask(~melt_tri)
    ax.tricontourf(triT, np.maximum(T, TEMP_B[0] + 1e-6), levels=TEMP_B, colors=TEMP_C, zorder=1)
    duct = ((T > T_DUCTILE) & (T < T_SOLIDUS)).astype(float)
    ax.tricontourf(tri, duct, levels=[0.5, 2.0], colors='none', hatches=['xx'], zorder=4)
    XG, DG, U, W, mag, Tg = vg
    keep = (Tg < VEL_GATE_T) & np.isfinite(mag) & (mag > VEL_REL_MIN * ref)
    hot = np.column_stack((S['x'], S['d']))[T >= T_SOLIDUS]
    if keep.any() and len(hot):                                     # no arrows on the magma body
        dist = np.hypot(XG.ravel()[:, None] - hot[None, :, 0], DG.ravel()[:, None] - hot[None, :, 1]).min(axis=1)
        keep &= ~(dist < VEL_MELT_CLEAR).reshape(XG.shape)
    Q = ax.quiver(XG, DG, np.where(keep, U, np.nan), np.where(keep, W, np.nan), angles='uv', scale=ref * 14,
                  scale_units='width', color='black', width=0.0048, headwidth=3.6, headlength=4.2, alpha=0.85,
                  pivot='mid', zorder=5)
    ax.set_xlim(*X_VIEW); ax.set_ylim(Z_VIEW[1], Z_VIEW[0]); ax.set_aspect('equal')
    draw_isotherms(ax, tri, T, X_ISO_LABEL)
    if melt_tri.any():
        ax.tricontourf(triC, chi, levels=CHI5_B, colors=CHI5_C, zorder=7)
        draw_isotherms(ax, tri, T, X_ISO_MAGMA, z_line=8, z_text=9.5)
    ax.set_xticks(np.arange(8, 23, 4))
    return Q


def map_colourbar(fig, rect):
    """One bar: host-rock temperature bands below, crystallinity buckets (with their temperatures) above."""
    cax = fig.add_axes(rect)
    combo = list(TEMP_C) + [CHI5_C[k] for k in range(4, -1, -1)]
    ncb, nt = len(combo), len(TEMP_C)
    for k, c in enumerate(combo):
        cax.add_patch(Rectangle((0, k), 1, 1, facecolor=c, edgecolor=c, lw=0.3, zorder=1))
    cax.set_xticks([]); cax.set_xlim(0, 1); cax.set_ylim(0, ncb)
    cax.axhline(nt, color='k', lw=0.8)
    cax.yaxis.set_ticks_position('left')
    cax.tick_params(length=1.8, pad=1.5)
    cax.text(-2.7, nt / 2.0, 'Temperature (°C)', rotation=90, va='center', ha='center', fontsize=7,
             transform=cax.get_yaxis_transform(), clip_on=False)
    duct_lo = 5.5
    cax.add_patch(Rectangle((0, duct_lo), 1, nt - duct_lo, fill=False, hatch='xx', edgecolor='0.38', lw=0.5, zorder=5))
    cax.text(1.7, duct_lo + (nt - duct_lo) / 2.0, 'Ductile', rotation=270, va='center', ha='center', fontsize=7,
             transform=cax.get_yaxis_transform(), clip_on=False)
    edges = CHI5_B[::-1]
    ys = [nt + k for k in range(len(edges))]
    cax.set_yticks([0, 2, 4, 6, 8] + ys)
    cax.set_yticklabels(['10', '200', '400', '600', '800', '950'] + ['%d' % round(chi_to_T(min(c, 1.0))) for c in edges[1:]],
                        fontsize=7)
    caxR = cax.twinx(); caxR.set_ylim(0, ncb)
    caxR.set_yticks(ys); caxR.set_yticklabels(['%g' % min(c, 1.0) for c in edges], fontsize=7)
    caxR.text(3.6, nt + 2.5, r'Crystallinity $\chi$', rotation=270, va='center', ha='center', fontsize=7,
              transform=caxR.get_yaxis_transform(), clip_on=False)
    cax.axhline(nt + 3, color='k', lw=0.6)                       # mobile / mush split (chi = 0.5)
    caxR.tick_params(length=1.8, pad=1.5)
    return cax


# ------------------------------------------------------------------------------------------------ schematic (a, b)
R, H = 15.0, 15.0                 # domain radius and depth (km)
ZE, ZE2 = 5.0, 2.0                # storage depths (km)
RS = 5.0                          # sill radius (km)
KTH = 2.5                         # thermal conductivity (W m-1 K-1) for the initial geotherm
EARTH, VENUS = dict(Ttop=10.0, q=0.120), dict(Ttop=460.0, q=0.045)
BDT0, BDT1, LOGK0, LOGK_MIN = 550.0, 750.0, -14.0, -22.0
K_EDGES = np.arange(-22, -13)
K_COLS = [plt.get_cmap('YlGnBu_r')(t) for t in np.linspace(0.02, 0.86, len(K_EDGES) - 1)]
K_CMAP = mcolors.ListedColormap(K_COLS)
K_NORM = mcolors.BoundaryNorm(K_EDGES, K_CMAP.N)
EL = np.radians(24.0)
CE, SE = np.cos(EL), np.sin(EL)
PHR = np.pi / 2 - np.radians(12.0)          # azimuth of the right side of the cut face
PHB = PHR + np.pi / 2
WALL_RGB = np.array(mcolors.to_rgb('#d3cdc2'))
OLD_C = '#f7dba6'
HS_NEW, HS_OLD, NPILE = 0.40, 0.55, 5       # drawn sill thicknesses (schematic) and number of sills drawn
FS = 7.0
SIDES = {'Earth': dict(p=EARTH, fluid='H$_2$O', bc=r'$T$ = 10 °C,  $P$ = 25 MPa', box='#136f53',
                       top=('#2a6aa0', '#7ab6df'), temps=(200, 400, 600), q=0.12),
         'Venus': dict(p=VENUS, fluid='CO$_2$', bc=r'$T$ = 460 °C,  $P$ = 10 MPa', box='#a34701',
                       top=('#b3652b', '#e4af73'), temps=(500, 600, 700), q=0.045)}
PAIR_WINDOW = (-19.5, 59.0, -19.6, 10.2)


def geotherm(z, p):
    return p['Ttop'] + p['q'] / KTH * np.asarray(z, float) * 1e3


def z_of_T(T, p):
    return (T - p['Ttop']) / (p['q'] / KTH * 1e3)


def k_bin(z, p):
    """Decade bin of the initial permeability at depth z (km): min(depth law, brittle-ductile reduction)."""
    z = np.maximum(np.asarray(z, float), 1e-9)
    f = np.clip((geotherm(z, p) - BDT0) / (BDT1 - BDT0), 0.0, 1.0)
    lk = np.minimum(np.minimum(LOGK0 - 3.2 * np.log10(z), LOGK0 * (1 - f) + LOGK_MIN * f), LOGK0 - 1e-9)
    return np.clip(np.searchsorted(K_EDGES, lk, side='right') - 1, 0, len(K_COLS) - 1)


def P(r, ph, z):
    r, ph, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(ph, float), np.asarray(z, float))
    return r * np.sin(ph), z * CE - r * np.cos(ph) * SE


def P2(r, ph, z):
    u, v = P(r, ph, z)
    return np.array([float(u), float(v)])


def F2(s, z, du=0.0):
    """Screen point on the cut face: s = signed distance from the axis (km), z up."""
    u, v = P(s, PHR, z)
    return np.array([float(u) + du, float(v)])


def rgba(C):
    return np.concatenate([C, np.ones(C.shape[:-1] + (1,))], -1)


def mesh(ax, U, V, C, zorder):
    q = np.stack([np.stack([U[:-1, :-1], V[:-1, :-1]], -1), np.stack([U[1:, :-1], V[1:, :-1]], -1),
                  np.stack([U[1:, 1:], V[1:, 1:]], -1), np.stack([U[:-1, 1:], V[:-1, 1:]], -1)], axis=2)
    C = C.reshape(-1, C.shape[-1])
    ax.add_collection(PolyCollection(q.reshape(-1, 4, 2), facecolors=C, edgecolors=C, linewidths=0.3,
                                     zorder=zorder, rasterized=True))


def seg(ax, a, b, **kw):
    ax.plot([a[0], b[0]], [a[1], b[1]], **kw)


def arrow(ax, a, b, color='k', lw=0.8, ms=6, style='-|>', z=6):
    ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def paint_face(ax, text, s0, z0, size, color, ha='left', halo=None, zorder=3.6, du=0.0):
    """Text lying in the cut face, reading left to right."""
    tp = TextPath((0, 0), text, size=size, prop=FontProperties(family=['sans-serif']))
    bb = tp.get_extents()
    x0 = {'left': 0.0, 'center': -0.5 * bb.width, 'right': -bb.width}[ha] - bb.x0
    u0, v0 = F2(s0 + x0, z0, du)
    m = np.array([[np.sin(PHR), 0.0, u0], [-np.cos(PHR) * SE, CE, v0], [0.0, 0.0, 1.0]])
    patch = PathPatch(tp, fc=color, ec='none', lw=0, transform=Affine2D(m) + ax.transData, zorder=zorder)
    if halo:
        patch.set_path_effects([pe.Stroke(linewidth=1.3, foreground=halo), pe.Normal()])
    ax.add_patch(patch)


def face_bands(ax, sd, s0, s1, z_top, z_bot, du, zorder):
    """Permeability decade bands of planet sd on the cut face between s0..s1 and depths z_top..z_bot (km)."""
    zs = np.linspace(z_top, z_bot, 3001 if z_bot - z_top > 5 else 400)
    bins = k_bin(0.5 * (zs[:-1] + zs[1:]), sd['p'])
    cuts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(bins)]
    for i0, i1 in zip(cuts[:-1], cuts[1:]):
        c = K_COLS[bins[i0]]
        ax.add_patch(Polygon([F2(s0, -zs[i0], du), F2(s1, -zs[i0], du), F2(s1, -zs[i1], du),
                              F2(s0, -zs[i1], du)], fc=c, ec=c, lw=0.3, zorder=zorder))


def pile_colour(i, n):
    """New sill = mobile-melt red; older sills by age from mush orange to near-solid cream."""
    if i == 0:
        return np.array(mcolors.to_rgb(fs.MOBILE_C))
    f = (i - 1) / max(n - 2, 1)
    return (1 - f) * np.array(mcolors.to_rgb(fs.MUSH_C)) + f * np.array(mcolors.to_rgb(OLD_C))


def draw_cylinder(ax, du, planet, ybot, ruler=False, side=False):
    """One cylinder cut along a diameter; the cut face is the 2-D axisymmetric model domain."""
    sd = SIDES[planet]
    ph = np.linspace(PHR, np.pi / 2, 12)                                       # outer wall sliver
    zz = np.linspace(0.0, -H, 31)
    U, V = P(R, *np.meshgrid(ph, zz, indexing='ij'))
    phc, zc = 0.5 * (ph[:-1] + ph[1:]), 0.5 * (zz[:-1] + zz[1:])
    lit = (0.74 + 0.26 * np.cos(phc + np.radians(30)))[:, None] * (1.0 + 0.05 * zc / H)[None, :]
    mesh(ax, U + du, V, rgba(np.clip(WALL_RGB[None, None, :] * lit[:, :, None], 0, 1)), zorder=1)
    RR, PHg = np.meshgrid(np.linspace(0.0, R, 25), np.linspace(PHR, PHR + np.pi, 61), indexing='ij')   # top
    U, V = P(RR, PHg, 0.0)
    vc = 0.25 * (V[:-1, :-1] + V[1:, :-1] + V[1:, 1:] + V[:-1, 1:])
    f = np.clip((vc + R * np.cos(PHR) * SE) / (R * SE + R * np.cos(PHR) * SE), 0.0, 1.0)[..., None]
    C = (1 - f) * np.array(mcolors.to_rgb(sd['top'][0])) + f * np.array(mcolors.to_rgb(sd['top'][1]))
    mesh(ax, U + du, V, rgba(C), zorder=2)
    face_bands(ax, sd, -R, R, 0.0, H, du, zorder=2)                            # initial permeability
    for zst, spans in ((ZE2, [(-R, R)]), (ZE, [(-R, -RS), (RS, R)])):          # storage depths
        for a0, a1 in spans:
            seg(ax, F2(a0, -zst, du), F2(a1, -zst, du), color='0.15', lw=0.6, ls=(0, (3, 2)), zorder=3.2)
    for T in sd['temps']:                                                      # initial isotherms
        zT = z_of_T(T, sd['p'])
        seg(ax, F2(-R, -zT, du), F2(R, -zT, du), color=ISO_RED, lw=0.7, zorder=3.3, solid_capstyle='butt',
            path_effects=[pe.Stroke(linewidth=1.7, foreground='white', alpha=0.85), pe.Normal()])
        paint_face(ax, f'{T} °C', R - 0.5, -zT + 0.25, 1.27, ISO_RED, ha='right', halo='white', du=du)
    for sgn in (-1, 1):
        arrow(ax, F2(sgn * 0.8, -0.65, du), F2(sgn * 4.0, -0.65, du), color='0.1', lw=0.7, ms=5, z=3.7)
        ax.text(*F2(sgn * 4.8, -0.65, du), '$r$', ha='center', va='center', fontsize=FS, color='0.1', zorder=9)
    if ruler:
        # domain label above the axis line, on a backing that repeats the permeability bands it covers
        txt, size, z0 = '2-D axisymmetric model domain', 1.27, -H + 0.45
        bb = TextPath((0, 0), txt, size=size, prop=FontProperties(family=['sans-serif'])).get_extents()
        face_bands(ax, sd, -R + 0.25, -R + 0.75 + bb.width, -(z0 + bb.y1 + 0.2), -(z0 + bb.y0 - 0.2), du, zorder=4.55)
        paint_face(ax, txt, -R + 0.5, z0, size, 'white', du=du, zorder=4.6)
    face = np.array([F2(-R, 0, du), F2(R, 0, du), F2(R, -H, du), F2(-R, -H, du), F2(-R, 0, du)])
    ax.plot(face[:, 0], face[:, 1], color='0.1', lw=0.9, zorder=4.2, solid_joinstyle='miter')
    EDGE = dict(color='0.22', lw=0.6, zorder=4, solid_capstyle='round')
    u, v = P(R, np.linspace(PHR, PHR + np.pi, 200), 0.0); ax.plot(u + du, v, **EDGE)
    u, v = P(R, np.linspace(PHR, np.pi / 2, 30), -H); ax.plot(u + du, v, **EDGE)
    seg(ax, P2(R, np.pi / 2, 0.0) + [du, 0], P2(R, np.pi / 2, -H) + [du, 0], **EDGE)
    seg(ax, F2(0, 4.0, du), F2(0, -H - 1.0, du), color='k', lw=0.8, ls=(0, (7, 2, 1.5, 2)), zorder=4.5)   # axis
    arrow(ax, F2(0, 3.2, du), F2(0, 4.6, du), color='k', lw=0.8, ms=6, z=4.6)
    ax.text(*(F2(0, 4.6, du) + [0.0, 0.3]), '$z$', ha='center', va='bottom', fontsize=7, zorder=9)
    rho, za, gap = 2.9, 2.4, np.radians(8.0)                                   # each side sweeps half a turn
    for ph0, ph1 in ((PHB - 2 * np.pi + gap, PHR - np.pi / 2 - gap), (PHR - np.pi / 2 + gap, PHB - gap)):
        cu, cv = P(rho, np.linspace(ph0, ph1, 80), za)
        ax.add_patch(FancyArrowPatch(path=Path(np.column_stack([cu + du, cv])), arrowstyle='-|>',
                                     mutation_scale=6, color='k', lw=0.9, zorder=4.7))
    # sill pile: new sill at the storage depth, older sills pushed down
    phs = np.linspace(-np.pi / 2, PHR, 50)
    phc = 0.5 * (phs[:-1] + phs[1:])
    tops = [-ZE] + [-ZE - HS_NEW - k * HS_OLD for k in range(NPILE)]
    for i in range(NPILE):
        U, V = P(RS, *np.meshgrid(phs, [tops[i], tops[i + 1]], indexing='ij'))
        mesh(ax, U + du, V, rgba(np.tile(pile_colour(i, NPILE), (len(phc), 1, 1))), zorder=5)
        if i:
            u, v = P(RS, phs, tops[i] + 0 * phs)
            ax.plot(u + du, v, color='white' if i > 1 else '0.22', lw=0.5 if i > 1 else 0.6, zorder=5.1)
    zb = tops[NPILE]
    u, v = P(RS, np.linspace(PHR - np.pi, PHR, 80), -ZE)
    ax.add_patch(Polygon(np.column_stack([u + du, v]), closed=True, fc=fs.MOBILE_C, ec='0.22', lw=0.45, zorder=5.2))
    for ph in (-np.pi / 2, PHR):
        seg(ax, P2(RS, ph, -ZE) + [du, 0], P2(RS, ph, zb) + [du, 0], color='0.22', lw=0.45, zorder=5.2)
    u, v = P(RS, phs, zb + 0 * phs); ax.plot(u + du, v, color='0.22', lw=0.45, zorder=5.2)
    zs = [tops[1], 0.5 * (tops[1] + zb), zb]                                   # two curly "pushed down" arrows
    for z_a, z_b in zip(zs[:-1], zs[1:]):
        a, b = P2(RS, PHR, z_a - 0.15) + [du, 0.0], P2(RS, PHR, z_b + 0.25) + [du, 0.0]
        cx, cy = 0.5 * (a[0] + b[0]) + 0.2, 0.5 * (a[1] + b[1])
        ry = 0.5 * (a[1] - b[1]); th = np.linspace(np.pi / 2, -np.pi / 2, 40)
        ax.add_patch(FancyArrowPatch(path=Path(np.column_stack([cx + 1.25 * ry * np.cos(th), cy + ry * np.sin(th)])),
                                     arrowstyle='-|>,head_length=0.22,head_width=0.13', mutation_scale=7,
                                     color='0.12', lw=0.65, zorder=6))
    ax.text(*(P2(2.2, PHR - np.pi / 2, -ZE) + [du, 0.0]), 'new sill', ha='center', va='center',
            fontsize=FS, color='white', zorder=9)
    ax.text(du, R * SE + 2.1, f'{planet}  ({sd["fluid"]})\n{sd["bc"]}', ha='center', va='center', fontsize=FS,
            color='white', linespacing=1.3, zorder=9,
            bbox=dict(boxstyle='round,pad=0.35', fc=sd['box'], ec='white', lw=0.6, alpha=0.95))
    for s_ in (-12.5, -7.5, -2.5, 2.5, 7.5, 12.5):                              # basal heat flux
        tip = F2(s_, -H, du)
        arrow(ax, tip + [0, -2.2], tip + [0, -0.15], color=ISO_RED, lw=0.9, ms=5.5, z=6)
    ax.text(du, ybot + 0.2, '$q$ = %g W m$^{-2}$' % sd['q'], ha='center', va='bottom', fontsize=FS,
            color='#7f0000', zorder=9)
    if ruler:                                                                   # depth ruler and radius
        for zt in (0, 2, 5, 10, 15):
            a = F2(-R, -zt, du)
            seg(ax, a, a - [0.55, 0], color='0.25', lw=0.6, zorder=4)
            ax.text(a[0] - 0.8, a[1], f'{zt}' + (' km' if zt in (0, 15) else ''), ha='right', va='center',
                    fontsize=FS, color='0.2', zorder=9)
        d0, d1 = F2(-R, 0, du) + [0, 0.8], F2(0, 0, du) + [0, 0.8]
        arrow(ax, d0, d1, color='white', lw=0.6, ms=4.5, style='<|-|>', z=4.7)
        for e, d in ((F2(-R, 0, du), d0), (F2(0, 0, du), d1)):
            seg(ax, e + [0, 0.15], d + [0, 0.3], color='white', lw=0.45, zorder=4.7)
        dm = 0.5 * (d0 + d1)
        ax.text(dm[0], dm[1] + 0.3, '15 km', ha='center', va='bottom', fontsize=FS, color='white', zorder=9,
                rotation=np.degrees(np.arctan2(d1[1] - d0[1], d1[0] - d0[0])), rotation_mode='anchor')
    if side:                                                                    # insulated side wall
        a, b = P2(R, np.pi / 2, 0.0) + [du, 0], P2(R, np.pi / 2, -H) + [du, 0]
        for v in np.linspace(b[1] + 0.4, a[1] - 0.4, 15):
            seg(ax, [a[0] + 0.05, v], [a[0] + 0.6, v + 0.45], color='0.35', lw=0.45, zorder=4)
        ax.text(a[0] + 1.3, 0.5 * (a[1] + b[1]), 'insulated, no flow', rotation=90, ha='center', va='center',
                fontsize=FS, color='0.1', zorder=9)


def draw_setup(ax, du_v, window, cbar_u):
    ax.set_xlim(window[0], window[1]); ax.set_ylim(window[2], window[3])
    ax.set_aspect('equal'); ax.axis('off')
    draw_cylinder(ax, 0.0, 'Earth', window[2], ruler=True)
    draw_cylinder(ax, du_v, 'Venus', window[2], side=True)
    cax = ax.inset_axes([cbar_u, -9.0, 0.8, 11.0], transform=ax.transData)       # permeability legend
    cb = ax.figure.colorbar(ScalarMappable(norm=K_NORM, cmap=K_CMAP), cax=cax, ticks=K_EDGES[::2])
    cb.ax.set_yticklabels([rf'$10^{{{e}}}$' for e in K_EDGES[::2]], fontsize=FS)
    cb.ax.tick_params(length=2.0, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    ax.text(cbar_u - 1.2, 2.9, 'Permeability (m$^2$)', ha='left', va='bottom', fontsize=FS, color='0.1', zorder=9)


# ------------------------------------------------------------------------------------------------ volume panels (e, f)
def inset_e(axE, s):
    """Earth mobile melt (top band) and its mean crystallinity (bottom band), 160-200 kyr, every frame."""
    T0, T1 = 160.0, EMPL_END
    IX0, IY0, IW, IH = 0.30, 0.36, 0.44, 0.47
    axE.add_patch(FancyBboxPatch((IX0 - 0.02, IY0 - 0.13), IW * 1.42 + 0.02, IH + 0.16, boxstyle='square,pad=0',
                                 transform=axE.transAxes, fc='white', ec='none', zorder=5.8))
    ins = axE.inset_axes([IX0, IY0, IW, IH], zorder=6)
    w = (s['t'] > T0 - 0.5) & (s['t'] <= T1 + 1e-9)
    t, mob, kind = s['t'][w], s['mob'][w], s['kind'][w]
    ins.plot(t, mob, color=fs.MOBILE_C, lw=0.8, zorder=4)
    tpost, vpost = t[kind == 2], mob[kind == 2]
    F_LO, F_HI = 0.42, 0.56
    vtop = 10.0 * np.ceil(vpost.max() / 10.0)
    ins.set_xlim(T0 - 1.0, T1 + 1.0)
    ins.set_ylim(vtop - vtop / (1.0 - F_HI), vtop)
    ins.set_yticks(np.arange(0.0, vtop + 1.0, 20.0))
    ins.yaxis.tick_right()
    ins.tick_params(labelsize=6, length=1.6, pad=1.2, top=False, right=False)
    ins.set_xlabel('Time (kyr)', fontsize=6, labelpad=1)
    for ts in tpost:
        ins.axvline(ts, color='0.55', lw=0.4, ls=(0, (2, 2)), zorder=1)
    CHI_LINE, C0 = '#2c7fb8', 0.15
    wc = (s['t'] >= T0 - 1.0) & (s['t'] <= T1 + 0.01)
    insR = ins.twinx()
    insR.plot(s['t'][wc], s['chi'][wc], color=CHI_LINE, lw=0.6, alpha=0.9)
    ins.set_zorder(insR.get_zorder() + 1); ins.patch.set_visible(False)
    insR.set_ylim(C0, C0 + (fs.CHI_MOBILE - C0) / F_LO)
    insR.set_yticks([0.2, 0.5])
    for sp in insR.spines.values():
        sp.set_visible(False)
    insR.tick_params(labelsize=6, length=1.6, pad=1.2, colors=CHI_LINE)
    ins.tick_params(axis='y', colors=fs.MOBILE_C, left=False, right=True, labelleft=False, labelright=True)
    for sp in ('left', 'right', 'top', 'bottom'):
        ins.spines[sp].set_visible(True); ins.spines[sp].set_color('k'); ins.spines[sp].set_linewidth(0.5)
        ins.spines[sp].set_bounds(*(ins.get_ylim() if sp in ('left', 'right') else ins.get_xlim()))
    for (y0, y1), col in (((F_HI, 1.0), fs.MOBILE_C), ((0.0, F_LO), CHI_LINE)):
        ins.plot([1.0, 1.0], [y0, y1], transform=ins.transAxes, color=col, lw=1.1, solid_capstyle='butt',
                 clip_on=False, zorder=10)
    lab = dict(fontsize=6, rotation=90, ha='center', va='center', rotation_mode='anchor', linespacing=1.0,
               transform=ins.transAxes, clip_on=False)
    ins.text(1.25, 0.5 * (F_HI + 1.0), 'mobile melt\n(km³)', color=fs.MOBILE_C, **lab)
    ins.text(1.25, 0.5 * F_LO, 'mean χ of\nmobile melt', color=CHI_LINE, **lab)
    x0z, x1z = T0 - 1.0, T1 + 1.0                                              # zoom box + connectors
    axE.add_patch(Rectangle((x0z, 0), x1z - x0z, vtop, fill=False, edgecolor='0.25', lw=0.7, zorder=7))
    for (xr, yr), yi in (((x0z, vtop), 1.0), ((x1z, 0.0), 0.0)):
        axE.add_artist(ConnectionPatch(xyA=(xr, yr), coordsA=axE.transData, xyB=(0, yi), coordsB=ins.transAxes,
                                       linestyle=(0, (3, 2)), lw=0.6, color='0.3', zorder=6.5))


def volume_panel(ax, data, planet, s, ymax):
    meta = run_meta(data, *STACK_RUNS[planet][-1])
    t, me, mu, du = s['t'], s['mob'], s['mush'], s['duct']
    short = t[-1] < T_END - 10.0
    if short and not meta['still_running_at_export'] and me[-1] + mu[-1] + du[-1] < 1.0:
        t, me, mu, du = (np.append(t, T_END), *(np.append(a, 0.0) for a in (me, mu, du)))
        short = False
    ax.stackplot(t, me, mu, du, colors=[fs.MOBILE_C, fs.MUSH_C, fs.DUCTILE_C], zorder=2, lw=0)
    c = read_series(data, *COND_RUNS[planet])                                   # conduction-only control
    for yv, col in ((c['mob'], '#7f0010'), (c['mob'] + c['mush'], '#a0611a'), (c['mob'] + c['mush'] + c['duct'], '#4b4580')):
        ax.plot(c['t'], yv, color=col, lw=0.7, ls=(0, (3, 1.6)), zorder=4)
    ts, cum, _, _ = emplaced(read_series(data, *STACK_RUNS[planet][0]))
    ax.step(np.r_[ts, T_END], np.r_[cum, cum[-1]], where='post', color='k', lw=0.7, zorder=5)
    if planet == 'Venus':
        ax.text(0.62 * T_END, cum[-1], 'cumulative magma emplaced', fontsize=7, ha='center', va='center', zorder=6,
                bbox=fs.label_box(fc=fs.DUCTILE_C))
    if short:                                                                   # output ends before the axis end
        ax.plot([t[-1]] * 2, [0, me[-1] + mu[-1] + du[-1]], color='0.25', lw=0.6, ls=':', zorder=3)
    ax.set_xlim(0, T_END); ax.set_ylim(0, ymax)
    ax.set_xlabel('Time (kyr)')
    ax.tick_params(which='both', top=False, right=False)
    fs.box(ax)


# ------------------------------------------------------------------------------------------------ numbers
def report(data, S):
    def at(s, t, kind):
        k = np.flatnonzero((np.abs(s['t'] - t) < 1e-6) & (s['kind'] == kind))
        return s['mob'][k[0]], s['mush'][k[0]]

    def zero_after(s, a):
        k = np.flatnonzero((s['t'] > EMPL_END + 0.5) & (a < 0.05))
        return s['t'][k[0]] if len(k) else None

    print('\nNumbers (Fig. 1):')
    for planet in ('Earth', 'Venus'):
        s = S[planet]
        mob_end, mush_end = at(s, EMPL_END, 0)
        mm = s['mob'] + s['mush']
        i_m, i_mm, i_d = np.argmax(s['mob']), np.argmax(mm), np.argmax(s['duct'])
        z_mob, z_mush = zero_after(s, s['mob']), zero_after(s, s['mush'])
        print(f'  {planet}: end of emplacement (200 kyr, before sill 50) mobile melt {mob_end:.1f} km3, mush {mush_end:.1f} km3')
        print(f'    peak mobile melt {s["mob"][i_m]:.1f} km3 at {s["t"][i_m]:.1f} kyr; peak melt+mush {mm[i_mm]:.1f} km3 '
              f'at {s["t"][i_mm]:.1f} kyr; peak induced ductile {s["duct"][i_d]:.1f} km3 at {s["t"][i_d]:.1f} kyr')
        print('    mobile melt gone at %s, mush gone at %s; induced ductile at end of output (%.0f kyr) %.1f km3' % (
            '%.0f kyr' % z_mob if z_mob is not None else 'not reached', '%.0f kyr' % z_mush if z_mush is not None else 'not reached',
            s['t'][-1], s['duct'][-1]))
    s = S['Earth']
    tr = (s['kind'] == 0) & (s['t'] > 159.9) & (s['t'] <= EMPL_END + 1e-9)
    ch = s['chi'][(s['t'] >= 159.0) & (s['t'] <= EMPL_END + 0.01)]
    first = s['t'][(s['kind'] == 0) & (s['t'] <= EMPL_END + 1e-9) & (s['mob'] > 0.5)]
    print(f'  Earth inset 160-200 kyr: mobile melt surviving between sills {s["mob"][tr].min():.1f}-{s["mob"][tr].max():.1f} km3; '
          f'mean chi of mobile melt {np.nanmin(ch):.2f}-{np.nanmax(ch):.2f}; first inter-sill survival at {first.min():.0f} kyr')
    _, cum, v0, v1 = emplaced(read_series(data, *STACK_RUNS['Venus'][0]))
    print(f'  intruded volume: initial pluton {v0:.1f} km3 + 50 sills x {v1:.2f} km3 = {cum[-1]:.1f} km3')
    ce, cv = read_series(data, *COND_RUNS['Earth']), read_series(data, *COND_RUNS['Venus'])
    me, ue = at(ce, EMPL_END, 0); mv, uv = at(cv, EMPL_END, 0)
    mE = at(S['Earth'], EMPL_END, 0)[0]; mV = at(S['Venus'], EMPL_END, 0)[0]
    print(f'  conduction only at 200 kyr: Earth mobile {me:.1f} / mush {ue:.1f} km3; Venus mobile {mv:.1f} / mush {uv:.1f} km3')
    print(f'  mobile-melt split: hotter crust x{mv / me:.1f} ({mv:.0f}/{me:.0f}); Earth fluid x{me / mE:.1f} ({me:.0f}/{mE:.0f}); '
          f'Venus fluid x{mv / mV:.2f} ({mv:.0f}/{mV:.0f}); total Venus/Earth x{mV / mE:.1f}')


# ------------------------------------------------------------------------------------------------ main
def main():
    args = fs.common_args(__doc__.splitlines()[0]).parse_args()
    fs.apply()
    plt.rcParams['hatch.color'] = '0.38'
    plt.rcParams['hatch.linewidth'] = 0.4
    data = args.data_dir

    # page: maps block (5.3 in) + schematic above it
    FW, H0 = fs.WIDTH['double'], 5.3
    SW_B, SW_T = PAIR_WINDOW[2:4]
    S_IN, GAP, TOPM = 1.95 / 25.4, -0.32, 0.05
    FH = H0 + GAP + (SW_T - SW_B) * S_IN + TOPM
    Y = lambda f: f * H0 / FH
    fig = plt.figure(figsize=(FW, FH))
    gs = GridSpec(1, 2, wspace=0.06, left=0.105, right=0.755, top=Y(0.905), bottom=Y(0.515))

    maps = {}
    for ci, planet in enumerate(('Earth', 'Venus')):
        suite, run = MAP_RUNS[planet]
        S = load_snapshot(data, suite, run, f'Variables_{SNAP_YEAR}_post.npz')
        vg = velocity_grid(data, suite, run)
        ref = np.percentile(vg[4][vg[5] < VEL_GATE_T], 95)
        ax = fig.add_subplot(gs[0, ci])
        Q = render_map(ax, S, vg, ref)
        vk = nice_speed(ref)
        ax.quiverkey(Q, 0.88, 1.035, vk, speed_label(vk), labelpos='W', coordinates='axes', color='0.15',
                     labelsep=0.04, fontproperties={'size': 7}).set_zorder(9)
        ax.set_xlabel('Distance (km)')
        if ci == 0:
            ax.set_ylabel('Depth (km)')
        else:
            ax.set_yticklabels([])
        ax.text(0.015, 1.012, 't = %d kyr' % round(SNAP_YEAR / 1000), transform=ax.transAxes, fontsize=7,
                ha='left', va='bottom')
        maps[planet] = ax
        print(f'{planet} map: {run} at {SNAP_YEAR / 1000:.0f} kyr; arrows p95 {ref:.2e} m/s, reference arrow {vk:.0e} m/s')

    fig.canvas.draw()
    pE, pV = maps['Earth'].get_position(), maps['Venus'].get_position()
    cax = map_colourbar(fig, [0.815, min(pE.y0, pV.y0), 0.016, max(pE.y1, pV.y1) - min(pE.y0, pV.y0)])

    axE = fig.add_axes([0.105, Y(0.075), 0.305, Y(0.32)])
    axV = fig.add_axes([0.450, Y(0.075), 0.305, Y(0.32)])
    S = {p: stack_series(data, p) for p in ('Earth', 'Venus')}
    ymax = 1.06 * max((s['mob'] + s['mush'] + s['duct']).max() for s in S.values())
    volume_panel(axE, data, 'Earth', S['Earth'], ymax)
    volume_panel(axV, data, 'Venus', S['Venus'], ymax)
    axE.set_ylabel('Volume (km³)', fontsize=7)
    axV.set_yticklabels([])
    pf = axV.get_position()
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=c, ec='none', label=l) for c, l in
                        ((fs.DUCTILE_C, 'induced ductile\n($\\chi$ = 1, > %d °C)' % T_DUCTILE),
                         (fs.MUSH_C, 'crystal mush\n(0.5 ≤ $\\chi$ < 1)'), (fs.MOBILE_C, 'mobile melt\n($\\chi$ < 0.5)'))]
                       + [Line2D([], [], color='0.25', lw=0.7, ls=(0, (3, 1.6)), label='conduction\nonly')],
               loc='center left', bbox_to_anchor=(pf.x1 + 0.012, 0.5 * (pf.y0 + pf.y1)), fontsize=7, frameon=False,
               handlelength=1.2, handleheight=0.9, labelspacing=0.8)
    inset_e(axE, read_series(data, *STACK_RUNS['Earth'][0]))

    # schematic: one half-cylinder per planet, axis over the centre of that planet's map
    xE, xV = 0.5 * (pE.x0 + pE.x1) * FW, 0.5 * (pV.x0 + pV.x1) * FW
    cbar_u = (cax.get_position().x0 * FW - xE) / S_IN
    SW = (PAIR_WINDOW[0], cbar_u + 6.0, SW_B, SW_T)
    w_in, h_in = (SW[1] - SW[0]) * S_IN, (SW[3] - SW[2]) * S_IN
    y0_in = H0 + GAP
    axS = fig.add_axes([(xE + SW[0] * S_IN) / FW, y0_in / FH, w_in / FW, h_in / FH])
    draw_setup(axS, (xV - xE) / S_IN, SW, cbar_u)

    pe_, pf_ = axE.get_position(), axV.get_position()
    letters = [((pE.x0, (y0_in + h_in) / FH), 'a', 'top'), ((pV.x0, (y0_in + h_in) / FH), 'b', 'top'),
               ((pE.x0, pE.y1 + 0.035), 'c', 'bottom'), ((pV.x0, pV.y1 + 0.035), 'd', 'bottom'),
               ((pE.x0, pe_.y1 + 0.03), 'e', 'bottom'), ((pV.x0, pf_.y1 + 0.03), 'f', 'bottom')]
    for (x, y), l, va in letters:
        fig.text(x, y, l, fontsize=10, fontweight='bold', ha='right', va=va)
    fs.save(fig, 'fig1_setup_depth', args.out_dir)
    report(data, S)


if __name__ == '__main__':
    main()
