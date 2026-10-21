#!/usr/bin/env python3
"""Extended Data Fig. 1: Venus (CO2) magma state at the end of emplacement and effective Nusselt number.

  a  regime map as Fig. 2a for Venus (measured cells only; no k = 1e-13 row)
  b  Nu_eff versus emplacement rate for k = 1e-16, 1e-15, 1e-14 m2 and the two depth-dependent suites

Definitions as in fig2.py: whole-domain mobile melt / mush at the last inter-sill trough; Nu_eff from the sill-complex
heat budget relative to the conduction-only run at the same rate (uniform k: VC_R*_NC_LH; depth-dependent 2 km:
VC_R*_NC_2KM_LH; depth-dependent 5 km: output-schedule-matched VC_R*_NC_PP_LH).

Usage:  python ed_fig1.py --data-dir <unzipped Zenodo record> [--out-dir figures]
"""
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Wedge
from matplotlib.ticker import FuncFormatter

import figstyle as fs

RATES = [5, 10, 20, 50, 100, 200, 400]
ROWS = ['cond', 'K16', 'MI5', 'K15', 'MI2', 'K14']             # bottom to top
UNIFORM_K = ['K16', 'K15', 'K14']
ROW_LABEL = {'cond': 'conduction', 'K16': '$10^{-16}$', 'K15': '$10^{-15}$', 'K14': '$10^{-14}$',
             'MI5': 'depth dep. (5 km)', 'MI2': 'depth dep. (2 km)'}
THR = 0.5                 # km3
RSCALE = 0.0165           # pie radius (data units) per sqrt(km3)
N_SILLS = 50


class Data:
    """Reads data/<suite>/<run>/{timeseries.csv, run_meta.json} from the unzipped Zenodo record."""

    def __init__(self, root):
        self.dirs = {}
        base = os.path.join(root, 'data')
        if not os.path.isdir(base):
            raise SystemExit(f'no data/ directory under {root!r}: pass --data-dir <unzipped Zenodo record>')
        for suite in os.listdir(base):
            for run in os.listdir(os.path.join(base, suite)):
                if os.path.exists(os.path.join(base, suite, run, 'timeseries.csv')):
                    self.dirs[run] = os.path.join(base, suite, run)
        self._ts = {}

    def has(self, run):
        return run in self.dirs

    def series(self, run):
        if run not in self._ts:
            with open(os.path.join(self.dirs[run], 'timeseries.csv')) as fh:
                self._ts[run] = [(round(float(r['t_kyr']) * 1000.0), r['frame'], r) for r in csv.DictReader(fh)]
        return self._ts[run]

    def meta(self, run):
        return json.load(open(os.path.join(self.dirs[run], 'run_meta.json')))


def regime(data, run, rate):
    """(class, mobile km3, mush km3) at the last inter-sill trough, or None if the run is missing / unfinished."""
    if not data.has(run):
        return None
    emp = 1e7 / rate
    pres = [(t, r) for t, f, r in data.series(run) if f == 'pre' and t <= emp + 1]
    if not pres or max(t for t, _ in pres) < emp * 0.999:
        return None
    row = max(pres, key=lambda p: p[0])[1]
    mob, mush = float(row['mobile_melt_km3']), float(row['crystal_mush_km3'])
    return ('mobile' if mob > THR else 'mush' if mush > THR else 'solid'), mob, mush


def heat_budget(data, run, rate):
    """(f_ext, P_supply in J/yr) over the emplacement period, or None."""
    if not data.has(run):
        return None
    emp = 1e7 / rate
    pre, post = {}, {}
    for t, f, r in data.series(run):
        if t <= emp + 1 and f in ('pre', 'post'):
            (pre if f == 'pre' else post)[t] = float(r['H_sill_J'])
    if not post or max(post) < emp * 0.999:
        return None
    inj = [post[t] - pre[t] for t in sorted(post) if t in pre]
    if not inj:
        return None
    S = sum(inj) * max(N_SILLS, len(inj)) / len(inj) + data.meta(run)['H_initial_pluton_at_emplacement_J']
    return 1.0 - post[max(post)] / S, S / emp


def run_name(r, k):
    if k == 'cond':
        return f'VC_R{r}_NC_LH'
    if k == 'MI5':
        return f'V_R{r}_NC_MI_LH'
    if k == 'MI2':
        return f'V_R{r}_NC_MI_2KM_LH'
    return f'V_R{r}_NC_{k}_LH'


def control_name(r, k):
    return {'MI2': f'VC_R{r}_NC_2KM_LH', 'MI5': f'VC_R{r}_NC_PP_LH'}.get(k, f'VC_R{r}_NC_LH')


def nu_eff(data, r, k):
    a, c = heat_budget(data, run_name(r, k), r), heat_budget(data, control_name(r, k), r)
    return a[0] * a[1] / (c[0] * c[1]) if a and c else np.nan


def main():
    args = fs.common_args(__doc__.splitlines()[0]).parse_args()
    fs.apply()
    data = Data(args.data_dir)
    reg_col = {'mobile': fs.MOBILE_C, 'mush': fs.MUSH_C, 'solid': fs.SOLID_C}

    fig = plt.figure(figsize=(fs.WIDTH['double'], 3.15))
    axA = fig.add_axes([0.165, 0.13, 0.33, 0.66])
    axB = fig.add_axes([0.615, 0.13, 0.37, 0.66])

    # (a) regime map, measured cells only
    cells = {}
    for j, r in enumerate(RATES):
        for i, k in enumerate(ROWS):
            g = cells[(r, k)] = regime(data, run_name(r, k), r)
            if g is None:
                axA.text(j, i, '–', ha='center', va='center', color='0.6', fontsize=7)
                continue
            _, mob, mush = g
            tot = mob + mush
            if tot < THR:
                axA.scatter(j, i, s=14, marker='x', color=reg_col['solid'], lw=1.0, zorder=4)
                continue
            R = RSCALE * np.sqrt(tot); mf = mob / tot
            axA.add_patch(Wedge((j, i), R, 90, 90 + 360 * mf, fc=reg_col['mobile'], ec='0.3', lw=0.3, zorder=3))
            axA.add_patch(Wedge((j, i), R, 90 + 360 * mf, 450, fc=reg_col['mush'], ec='0.3', lw=0.3, zorder=3))
    axA.set_xticks(range(len(RATES))); axA.set_xticklabels([str(r) for r in RATES])
    axA.set_yticks(range(len(ROWS))); axA.set_yticklabels([ROW_LABEL[k] for k in ROWS])
    for lab, k in zip(axA.get_yticklabels(), ROWS):
        if k in ('MI5', 'MI2'):
            lab.set_fontstyle('italic')
    axA.set_xlim(-0.6, len(RATES) - 0.4); axA.set_ylim(-0.6, len(ROWS) - 0.4)
    axA.set_aspect('equal', adjustable='box')
    axA.set_axisbelow(True); axA.grid(True, color='0.92', lw=0.4)
    for sp in ('top', 'right'):
        axA.spines[sp].set_visible(True)
    axA.set_xlabel(r'Emplacement rate ($\mathrm{mm\ yr^{-1}}$)'); axA.set_ylabel('Permeability (m²)')

    # (b) Nu_eff
    nu = {k: [nu_eff(data, r, k) for r in RATES] for k in UNIFORM_K + ['MI5', 'MI2']}
    axB.axhline(1.0, color='k', lw=0.6, ls=fs.REF_DOT, zorder=2)
    for k in UNIFORM_K:
        pts = [(r, v) for r, v in zip(RATES, nu[k]) if np.isfinite(v)]
        axB.plot(*zip(*pts), '-', color=fs.KCOL[k], lw=1.0, zorder=3)
        axB.plot(*zip(*pts), 'o', color=fs.KCOL[k], ms=4, mew=0.8, zorder=4)
    for k in ('MI5', 'MI2'):
        pts = [(r, v) for r, v in zip(RATES, nu[k]) if np.isfinite(v)]
        st = fs.MI_STYLE[k]
        axB.plot(*zip(*pts), ls=st['ls'], color=st['color'], lw=1.0, zorder=5, dash_capstyle='round')
        axB.plot(*zip(*pts), ls='', marker=st['marker'], ms=4, mec=st['color'], mfc='white', mew=0.8, zorder=6)
    axB.set_xscale('log'); axB.set_xlim(4.2, 480); axB.set_ylim(0.98, 1.04)
    axB.set_xticks(RATES); axB.xaxis.set_major_formatter(FuncFormatter(lambda v, _: '%g' % v))
    axB.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    axB.minorticks_off(); fs.box(axB)
    axB.grid(True, which='major', color='0.92', lw=0.4)
    axB.set_xlabel(r'Emplacement rate ($\mathrm{mm\ yr^{-1}}$)')
    axB.set_ylabel(r'Effective Nusselt number, $Nu$$_{\mathrm{eff}}$')
    h = [Line2D([], [], color='k', lw=0.6, ls=fs.REF_DOT, label='conduction')]
    h += [Line2D([], [], color=fs.KCOL[k], lw=1.0, marker='o', ms=4, label=f'$k$ = {fs.KLAB[k]} m²') for k in UNIFORM_K]
    h += [Line2D([], [], color=fs.MI_STYLE[k]['color'], ls=fs.MI_STYLE[k]['ls'], lw=1.0, marker=fs.MI_STYLE[k]['marker'],
                 ms=4, mfc='white', mew=0.8, label=fs.MI_NAME[k]) for k in ('MI5', 'MI2')]
    axB.legend(handles=h, loc='upper left', frameon=False, handlelength=2.6, ncol=2, columnspacing=1.0)

    # regime legend and 500 km3 pie-size key
    fig.canvas.draw()
    pA = axA.get_position()
    reg = [Line2D([], [], marker='o', ls='', ms=5.5, mfc=reg_col['mobile'], mec='0.3', mew=0.3, label='mobile melt (χ < 0.5)'),
           Line2D([], [], marker='o', ls='', ms=5.5, mfc=reg_col['mush'], mec='0.3', mew=0.3, label='crystal mush (0.5 ≤ χ < 1)'),
           Line2D([], [], marker='x', ls='', ms=4.5, mew=1.0, color=reg_col['solid'], label='solid (χ = 1)')]
    fig.legend(handles=reg, loc='lower left', bbox_to_anchor=(0.02, pA.y1 + 0.035), ncol=3, fontsize=7,
               frameon=False, handletextpad=0.2, columnspacing=0.8)
    p0 = axA.transData.transform((0, 0)); p1 = axA.transData.transform((RSCALE * np.sqrt(500.0), 0))
    r_px = abs(p1[0] - p0[0]); W, H = fig.bbox.width, fig.bbox.height
    cx, cy = pA.x1 + 0.035, pA.y1 - r_px / H
    fig.add_artist(Ellipse((cx, cy), 2 * r_px / W, 2 * r_px / H, transform=fig.transFigure, fc='0.85', ec='0.4', lw=0.5))
    fig.text(cx, cy - r_px / H - 0.012, '500 km³', ha='center', va='top', fontsize=7, color='0.3')
    fs.panel(axA, 'a'); fs.panel(axB, 'b')
    fs.save(fig, 'ED_fig1_venus_regime', args.out_dir)

    # numbers quoted in the paper
    print('\nED Fig. 1a  Venus end-of-emplacement state, mobile melt / crystal mush (km3)')
    print('  %-6s' % 'row' + ''.join('%16d' % r for r in RATES))
    for k in reversed(ROWS):
        print('  %-6s' % k + ''.join('%16s' % ('-' if cells[(r, k)] is None else '%s %.1f/%.1f' % (
            cells[(r, k)][0][:3], cells[(r, k)][1], cells[(r, k)][2])) for r in RATES))
    print('\nED Fig. 1b  Venus effective Nusselt number (mm/yr: ' + ', '.join(map(str, RATES)) + ')')
    for k in UNIFORM_K + ['MI5', 'MI2']:
        print('  %-5s ' % k + ' '.join('%6s' % ('-' if not np.isfinite(v) else '%.4f' % v) for v in nu[k]))
    allv = [v for k in nu for v in nu[k] if np.isfinite(v)]
    print('  Venus Nu_eff range %.4f - %.4f' % (min(allv), max(allv)))


if __name__ == '__main__':
    main()
