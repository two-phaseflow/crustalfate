"""Figure 4: (a) post-emplacement time to solidus versus host permeability and emplacement rate, Earth and Venus;
(b) recovery of crustal competence at 200 mm/yr.

    python fig4.py --data-dir <unzipped Zenodo record> [--out-dir figures]

Prints the times to solidus (a) and the competence-recovery times (b) quoted in the paper.
"""
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as fs

T_SOLIDUS = 950.0                 # degC
COMPETENCE = 1.0e12               # N/m (1 TN/m)
RATES = [50, 100, 200, 400]       # mm/yr
RATE_B = 200                      # rate of panel (b)
EARTH_K = ['cond', 'K16', 'K5e16', 'K15', 'K5e15', 'K14']
VENUS_K = ['cond', 'K16', 'K15', 'K14']
PERM_X = {k: i for i, k in enumerate(EARTH_K)}
PERM_LAB = ['conduction', r'$10^{-16}$', r'$5{\times}10^{-16}$', r'$10^{-15}$', r'$5{\times}10^{-15}$', r'$10^{-14}$']
XMI = 6.4                         # x position of the depth-dependent (5 km) column
YLIM = (0.5, 1300)


# ----------------------------------------------------------------------------------------------- data
def run_dir(data_dir, run):
    hits = [h for h in glob.glob(os.path.join(data_dir, 'data', '*', run))
            if os.path.exists(os.path.join(h, 'run_meta.json'))]           # skip data/full_output/<run>
    return hits[0] if hits else None


def read_timeseries(data_dir, run):
    d = run_dir(data_dir, run)
    if d is None:
        return None
    with open(os.path.join(d, 'timeseries.csv')) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r['t_kyr']) for r in rows])
    tmax = np.array([float(r['T_max_sill_C']) if r['T_max_sill_C'] else np.nan for r in rows])
    return t, tmax


def time_to_solidus(data_dir, run, rate):
    """Post-emplacement time (kyr) at which the maximum temperature in the sill complex first drops below the
    solidus; (value, True) or, if the run ends still molten, (time since emplacement end, False)."""
    ts = read_timeseries(data_dir, run)
    if ts is None:
        return None
    t, tmax = ts
    emp = 1e4 / rate                                   # emplacement duration, kyr (10 km of sills)
    if t[-1] < emp:
        return None
    b = np.where((tmax < T_SOLIDUS) & (t > emp))[0]
    if b.size:
        return t[b[0]] - emp, True
    return t[-1] - emp, False


def earth_run(k, r):
    return f'EC_R{r}_NC_LH' if k == 'cond' else f'E_R{r}_{k}_NC_LH'


def venus_run(k, r):
    return f'VC_R{r}_NC_LH' if k == 'cond' else f'V_R{r}_NC_{k}_LH'


def read_competence(data_dir, run, pore_pressure='simulated'):
    """time (kyr) and footprint-averaged integrated strength (N/m) at strain rates 1e-16, 1e-15, 1e-14 1/s"""
    with open(os.path.join(run_dir(data_dir, run), 'competence.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['pore_pressure'] == pore_pressure]
    col = lambda k: np.array([float(r[k]) for r in rows])
    return dict(t=col('t_kyr'), lo=col('F_strain_1e-16_N_per_m'), mid=col('F_strain_1e-15_N_per_m'),
                hi=col('F_strain_1e-14_N_per_m'))


def recovery_time(t, F):
    """First time (kyr) after the strength minimum at which F rises back above 1 TN/m (linear interpolation)."""
    i0 = int(np.argmin(F))
    j = np.where(F[i0:] >= COMPETENCE)[0]
    if not j.size:
        return np.nan
    k = i0 + j[0]
    if k == i0:                                        # never dropped below the limit
        return np.nan
    return t[k - 1] + (COMPETENCE - F[k - 1]) * (t[k] - t[k - 1]) / (F[k] - F[k - 1])


# ----------------------------------------------------------------------------------------------- panels
def draw_time_to_solidus(ax, data_dir):
    table = {}
    for planet, perms, namer, ls, mk in (('Earth', EARTH_K, earth_run, '-', 'o'),
                                         ('Venus', VENUS_K, venus_run, fs.VENUS_DASH, 's')):
        for r in RATES:
            xs, ys = [], []
            for k in perms:
                res = time_to_solidus(data_dir, namer(k, r), r)
                if res is None:
                    continue
                table[(planet, k, r)] = res
                if res[1]:
                    xs.append(PERM_X[k]); ys.append(res[0])
                else:                                  # still molten when the output ends: lower bound
                    ax.scatter(PERM_X[k], res[0], marker=mk, s=16, facecolors='white', edgecolors=fs.RATE_COL[r],
                               linewidths=0.8, zorder=4)
            hl = r == RATE_B
            if xs:
                ax.plot(xs, ys, ls=ls, marker=mk, ms=5.5 if hl else 4, color=fs.RATE_COL[r], lw=1.8 if hl else 1.0,
                        zorder=5 if hl else 3)
    # depth-dependent permeability, sills at 5 km: runs with a cooling tail (50 mm/yr = the Fig. 1 runs)
    ax.axvline(5.7, color='0.6', lw=0.5, ls=(0, (2, 2)), zorder=1)
    for r in RATES:
        for planet, run, mk, dx in (('Earth', f'E_R{r}_NC_MI_LH', 'o', -0.12), ('Venus', f'V_R{r}_NC_MI_LH_2MYR', 's', 0.12)):
            res = time_to_solidus(data_dir, run, r)
            if res and res[1] and res[0] > 0:
                table[(planet, 'MI5', r)] = res
                ax.plot(XMI + dx, res[0], marker=mk, ls='', ms=5.5 if r == RATE_B else 4.5, mfc='white',
                        mec=fs.RATE_COL[r], mew=1.0, zorder=5)
    ax.set_xlim(-0.4, XMI + 0.5)
    ax.set_xticks(list(range(6)) + [XMI]); ax.set_xticklabels(PERM_LAB + ['depth dep.\n(5 km)'])
    ax.set_xlabel(r'Permeability (m$^2$)')
    ax.set_ylabel('Post-emplacement time to solidus  (kyr)')
    ax.set_yscale('log'); ax.set_ylim(*YLIM)
    ax.set_axisbelow(True); ax.grid(True, axis='y', which='major', color='0.92', lw=0.4)
    h = [Line2D([], [], color=fs.RATE_COL[r], lw=1.8 if r == RATE_B else 1.0,
                label=f'{r} mm yr$^{{-1}}$' + ('  (panel b)' if r == RATE_B else '')) for r in RATES]
    h += [Line2D([], [], color='0.3', lw=1.0, ls='-', marker='o', ms=4, label='Earth (H$_2$O)'),
          Line2D([], [], color='0.3', lw=1.0, ls=fs.VENUS_DASH, marker='s', ms=4, label='Venus (CO$_2$)')]
    ax.legend(handles=h, loc='lower left', frameon=False, handlelength=3.6)
    return table


def draw_competence(ax, data_dir):
    sc = 1e-12                                         # N/m -> TN/m
    curves = [(f'E_R{RATE_B}_K14_NC_LH', fs.KCOL['K14'], '-', r'Earth  $k=10^{-14}$', False),
              (f'E_R{RATE_B}_K15_NC_LH', fs.KCOL['K15'], '-', r'Earth  $k=10^{-15}$', False),
              (f'E_R{RATE_B}_K16_NC_LH', fs.KCOL['K16'], '-', r'Earth  $k=10^{-16}$', True),
              (f'V_R{RATE_B}_NC_K15_LH_2MYR', fs.KCOL['K15'], fs.VENUS_DASH, r'Venus  $k=10^{-15}$', True)]
    rec = {}
    for run, c, ls, lab, band in curves:
        d = read_competence(data_dir, run)
        ax.plot(d['t'], d['mid'] * sc, ls=ls, color=c, lw=1.0, label=lab, zorder=3)
        if band:
            ax.fill_between(d['t'], d['lo'] * sc, d['hi'] * sc, color=c, alpha=0.15, lw=0, zorder=1)
        rec[run] = (d['mid'].min() * sc, recovery_time(d['t'], d['mid']),
                    recovery_time(d['t'], d['hi']), recovery_time(d['t'], d['lo']))
    ax.axhline(COMPETENCE * sc, color='k', ls=fs.REF_DOT, lw=0.6, zorder=2)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'Integrated yield strength (TN m$^{-1}$)')
    ax.grid(True, color='0.92', lw=0.4)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor='0.6', alpha=0.3, edgecolor='none'))
    l.append('strain rate $10^{-16}$–$10^{-14}$ s$^{-1}$\n(lines: $10^{-15}$ s$^{-1}$)')
    ax.legend(h, l, loc='upper right', frameon=False, handlelength=3.0, ncol=2, columnspacing=1.0)
    ax.set_ylim(0, 2.45)
    ax.text(ax.get_xlim()[1] * 0.99, COMPETENCE * sc - 0.025, 'competence limit\n(1 TN m$^{-1}$)',
            ha='right', va='top', fontsize=7, color='0.2', linespacing=1.1)
    return rec


def main():
    a = fs.common_args(__doc__.splitlines()[0]).parse_args()
    fs.apply()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(fs.WIDTH['double'], 3.0))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.92, wspace=0.27)
    table = draw_time_to_solidus(axA, a.data_dir)
    rec = draw_competence(axB, a.data_dir)
    fs.box(axA); fs.box(axB)
    fs.panel(axA, 'a'); fs.panel(axB, 'b')
    fs.save(fig, 'fig4_solidus_competence', a.out_dir)

    print('\nTime to solidus after the last sill (kyr; "+" = still molten when output ends)')
    print('  %-7s %-6s' % ('planet', 'k') + ''.join('%10s' % f'{r} mm/yr' for r in RATES))
    for planet, perms in (('Earth', EARTH_K + ['MI5']), ('Venus', VENUS_K + ['MI5'])):
        for k in perms:
            cells = [table.get((planet, k, r)) for r in RATES]
            if not any(cells):
                continue
            print('  %-7s %-6s' % (planet, k) +
                  ''.join('%10s' % ('-' if c is None else '%.1f%s' % (c[0], '' if c[1] else '+')) for c in cells))
    print('\nCompetence recovery at %d mm/yr (kyr; strain rate 1e-15 1/s, range 1e-14 .. 1e-16 1/s)' % RATE_B)
    for run, (fmin, tc, thi, tlo) in rec.items():
        if not np.isfinite(tc):
            print('  %-24s minimum %.2f TN/m: never drops below 1 TN/m' % (run, fmin))
        else:
            print('  %-24s minimum %.2f TN/m: recovery %.0f (%.0f-%.0f)' % (run, fmin, tc, thi, tlo))


if __name__ == '__main__':
    main()
