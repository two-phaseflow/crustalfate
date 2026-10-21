#!/usr/bin/env python3
"""Figure 2: Earth (H2O) magma state at the end of emplacement and effective Nusselt number.

  a  regime map: mobile melt / crystal mush pies (pie area proportional to volume) or solid crosses, emplacement rate x
     host permeability, including the depth-dependent permeability suites with the sills at 5 km and 2 km
  b  effective Nusselt number Nu_eff versus emplacement rate, with the supply limit of the 5 km geometry

Definitions (see the Zenodo README):
  * Volumes: whole-domain mobile melt (chi < 0.5) and crystal mush (0.5 <= chi < 1) at the last inter-sill trough, i.e.
    the last `pre` frame at or before the end of emplacement (10^7 / rate yr). A cell is 'mobile' if mobile melt
    > 0.5 km3, else 'mush' if mush > 0.5 km3, else 'solid'.
  * Heat budget of the sill complex: supplied heat S = enthalpy of the initial pluton at emplacement + sum over sills of
    (H_sill[post] - H_sill[pre]); unreadable sill events are compensated by rescaling the sum to 50 events.
    f_ext = 1 - H_sill[last post] / S, P = S / emplacement duration.
  * Nu_eff = (f_ext P)_run / (f_ext P)_control, control = conduction-only run at the same rate (5 km uniform-k:
    EC_R*_NC_LH; depth-dependent 2 km: EC_R*_NC_2KM_LH; depth-dependent 5 km: output-schedule-matched EC_R*_NC_PP_LH,
    except at 5 mm/yr where EC_R5_NC_LH already has that schedule).
  * Supply limit = 1 / f_ext of the 5 km conduction-only run (all supplied heat extracted).
  * Cells without a run are inferred solid where a faster rate at the same (or a lower) permeability already
    solidifies between sills: 5 mm/yr at all k, 10 mm/yr at k >= 1e-15, 20 and 50 mm/yr at 1e-13, and the
    depth-dependent cells at 5-10 mm/yr. They are drawn with the same solid cross.

Usage:  python fig2.py --data-dir <unzipped Zenodo record> [--out-dir figures]
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
ROWS = ['cond', 'K16', 'MI5', 'K15', 'MI2', 'K14', 'K13']          # bottom to top
UNIFORM_K = ['K16', 'K15', 'K14', 'K13']
ROW_LABEL = {'cond': 'conduction', 'K16': '$10^{-16}$', 'K15': '$10^{-15}$', 'K14': '$10^{-14}$',
             'K13': '$10^{-13}$', 'MI5': 'depth dep. (5 km)', 'MI2': 'depth dep. (2 km)'}
THR = 0.5                 # km3: a phase 'persists' above this volume
RSCALE = 0.0165           # pie radius (data units) per sqrt(km3)
N_SILLS = 50
FORCE_SOLID = ({(5, k) for k in ('cond', 'K16', 'K15', 'K14', 'K13')}
               | {(10, 'K15'), (10, 'K14'), (10, 'K13'), (20, 'K13'), (50, 'K13')})


# ------------------------------------------------------------------------------------------- data access
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
        """list of (t_yr, frame, row-dict) in file order"""
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
        return f'EC_R{r}_NC_LH'
    if k == 'MI5':
        return f'E_R{r}_NC_MI_LH'
    if k == 'MI2':
        return f'E_R{r}_NC_MI_2KM_LH'
    return f'E_R{r}_{k}_NC_LH'


def control_name(r, k):
    if k == 'MI2':
        return f'EC_R{r}_NC_2KM_LH'
    if k == 'MI5':
        return 'EC_R5_NC_LH' if r == 5 else f'EC_R{r}_NC_PP_LH'
    return f'EC_R{r}_NC_LH'


def nu_eff(data, r, k):
    a, c = heat_budget(data, run_name(r, k), r), heat_budget(data, control_name(r, k), r)
    return a[0] * a[1] / (c[0] * c[1]) if a and c else np.nan


def supply_limit(data, r):
    c = heat_budget(data, f'EC_R{r}_NC_LH', r)
    return 1.0 / c[0] if c else np.nan


def cell(data, r, k):
    g = regime(data, run_name(r, k), r)
    if g is not None:
        return g
    if (r, k) in FORCE_SOLID or k in ('MI5', 'MI2'):
        return 'solid', 0.0, 0.0
    return None


# ------------------------------------------------------------------------------------------- figure
def main():
    args = fs.common_args(__doc__.splitlines()[0]).parse_args()
    fs.apply()
    data = Data(args.data_dir)
    reg_col = {'mobile': fs.MOBILE_C, 'mush': fs.MUSH_C, 'solid': fs.SOLID_C}

    fig = plt.figure(figsize=(fs.WIDTH['double'], 3.3))
    axA = fig.add_axes([0.150, 0.13, 0.33, 0.66])
    axB = fig.add_axes([0.615, 0.13, 0.37, 0.66])

    # (a) regime map
    cells = {}
    for j, r in enumerate(RATES):
        for i, k in enumerate(ROWS):
            c = cells[(r, k)] = cell(data, r, k)
            if c is None:
                axA.text(j, i, '·', ha='center', va='center', color='0.7', fontsize=7)
                continue
            _, mob, mush = c
            total = mob + mush
            if total < THR:
                axA.scatter(j, i, s=16, marker='x', color=reg_col['solid'], lw=1.0, zorder=4)
                continue
            R = RSCALE * np.sqrt(total); mf = mob / total
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
    axA.set_xlabel(r'Emplacement rate ($\mathrm{mm\ yr^{-1}}$)')
    axA.set_ylabel('Permeability (m$^{2}$)')

    # (b) Nu_eff
    nu = {k: [nu_eff(data, r, k) for r in RATES] for k in UNIFORM_K + ['MI5', 'MI2']}
    ceil = [supply_limit(data, r) for r in RATES]
    axB.axhline(1.0, color='k', lw=0.6, ls=fs.REF_DOT, zorder=2)
    cx = [(r, c) for r, c in zip(RATES, ceil) if np.isfinite(c)]
    axB.plot(*zip(*cx), ls=(0, (4, 2.5)), color='0.5', lw=0.6, zorder=2)
    axB.text(7.0, 0.5 * (ceil[0] + ceil[1]) + 0.03, 'supply limit\n(5 km)', color='0.4', fontsize=7, ha='center',
             va='center', linespacing=1.0, bbox=fs.label_box(), zorder=7)
    axB.text(13.5, 1.0, 'conduction', color='k', fontsize=7, ha='center', va='center', bbox=fs.label_box(), zorder=7)
    for k in UNIFORM_K:
        pts = [(r, v) for r, v in zip(RATES, nu[k]) if np.isfinite(v)]
        axB.plot(*zip(*pts), '-', color=fs.KCOL[k], lw=1.0, zorder=3)
        axB.plot(*zip(*pts), 'o', color=fs.KCOL[k], ms=4, mew=0.8, zorder=4)
    for k in ('MI5', 'MI2'):
        pts = [(r, v) for r, v in zip(RATES, nu[k]) if np.isfinite(v)]
        st = fs.MI_STYLE[k]
        axB.plot(*zip(*pts), ls=st['ls'], color=st['color'], lw=1.0, zorder=5, dash_capstyle='round')
        axB.plot(*zip(*pts), ls='', marker=st['marker'], ms=4, mec=st['color'], mfc='white', mew=0.8, zorder=6)
    fs.box(axA); fs.box(axB)
    axB.set_xscale('log'); axB.set_xlim(4.2, 480); axB.set_ylim(0.8, 6.3)
    axB.set_xticks(RATES); axB.set_yticks(range(1, 7))
    plain = FuncFormatter(lambda v, _: '%g' % v)
    axB.xaxis.set_major_formatter(plain); axB.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    axB.yaxis.set_major_formatter(plain)
    axB.minorticks_off()
    axB.grid(True, which='major', color='0.92', lw=0.4)
    axB.set_xlabel(r'Emplacement rate ($\mathrm{mm\ yr^{-1}}$)')
    axB.set_ylabel('Effective Nusselt number, $Nu$$_{\\mathrm{eff}}$')
    h = [Line2D([], [], color=fs.KCOL[k], lw=1.0, marker='o', ms=4, label=f'$k$ = {fs.KLAB[k]} m$^{{2}}$')
         for k in UNIFORM_K]
    h += [Line2D([], [], color=fs.MI_STYLE[k]['color'], ls=fs.MI_STYLE[k]['ls'], lw=1.0,
                 marker=fs.MI_STYLE[k]['marker'], ms=4, mfc='white', mew=0.8, label=fs.MI_NAME[k])
          for k in ('MI5', 'MI2')]
    axB.legend(handles=h, loc='upper left', fontsize=7, frameon=False, handlelength=2.6, labelspacing=0.35)

    # regime legend above (a) and the 500 km3 pie-size key
    fig.canvas.draw()
    pA = axA.get_position()
    reg = [Line2D([], [], marker='o', ls='', ms=6, mfc=reg_col['mobile'], mec='0.3', mew=0.3, label='mobile melt'),
           Line2D([], [], marker='o', ls='', ms=6, mfc=reg_col['mush'], mec='0.3', mew=0.3, label='crystal mush'),
           Line2D([], [], marker='x', ls='', ms=5, mew=1.0, color=reg_col['solid'], label='solid')]
    fig.legend(handles=reg, loc='lower center', bbox_to_anchor=(0.5 * (pA.x0 + pA.x1), pA.y1 + 0.005), ncol=3,
               fontsize=7, frameon=False, handletextpad=0.2, columnspacing=0.9)
    p0 = axA.transData.transform((0, 0)); p1 = axA.transData.transform((RSCALE * np.sqrt(500.0), 0))
    r_px = abs(p1[0] - p0[0]); W, H = fig.bbox.width, fig.bbox.height
    cxk, cyk = pA.x1 + 0.035, pA.y1 - r_px / H
    fig.add_artist(Ellipse((cxk, cyk), 2 * r_px / W, 2 * r_px / H, transform=fig.transFigure,
                           fc='0.85', ec='0.4', lw=0.5, clip_on=False))
    fig.text(cxk, cyk - r_px / H - 0.012, '500 km$^{3}$', ha='center', va='top', fontsize=7, color='0.3')
    fs.panel(axA, 'a'); fs.panel(axB, 'b')
    fs.save(fig, 'fig2_earth_2panel', args.out_dir)

    # numbers quoted in the paper
    print('\nFig. 2a  end-of-emplacement state, mobile melt / crystal mush (km3); * = inferred solid, no run')
    print('  %-6s' % 'row' + ''.join('%15d' % r for r in RATES))
    for k in reversed(ROWS):
        cols = []
        for r in RATES:
            c = cells[(r, k)]
            measured = regime(data, run_name(r, k), r) is not None
            cols.append('-' if c is None else ('%s %.1f/%.1f%s' % (c[0][:3], c[1], c[2], '' if measured else '*')))
        print('  %-6s' % k + ''.join('%15s' % c for c in cols))
    print('\nFig. 2b  effective Nusselt number (mm/yr: ' + ', '.join(map(str, RATES)) + ')')
    for k in UNIFORM_K + ['MI5', 'MI2']:
        print('  %-5s ' % k + ' '.join('%6s' % ('-' if not np.isfinite(v) else '%.3f' % v) for v in nu[k]))
    print('  supply limit (5 km) ' + ' '.join('%.2f' % c for c in ceil))
    for k in UNIFORM_K + ['MI5', 'MI2']:
        v = np.array(nu[k]); i = int(np.nanargmax(v))
        print('  peak Nu_eff %-4s %.2f at %d mm/yr' % (k, v[i], RATES[i]))
    mi5 = [v for v in nu['MI5'] if np.isfinite(v)]
    print('  depth dependent 5 km range %.2f-%.2f' % (min(mi5), max(mi5)))


if __name__ == '__main__':
    main()
