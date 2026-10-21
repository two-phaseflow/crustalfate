"""Extended Data Fig. 4: sensitivity of crustal-competence recovery to permeability and pore pressure (200 mm/yr).

    python ed_fig4.py --data-dir <unzipped Zenodo record> [--out-dir figures]

(a) the three permeable Venus runs (k = 1e-16, 1e-15, 1e-14 m2); (b) simulated versus hydrostatic pore pressure
for Earth k = 1e-16 and Venus k = 1e-15. Prints the recovery times quoted in the paper.
"""
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

COMPETENCE = 1.0e12               # N/m (1 TN/m)
SC = 1e-12                        # N/m -> TN/m


def read_competence(data_dir, run, pore_pressure='simulated'):
    """time (kyr) and footprint-averaged integrated strength (N/m) at strain rates 1e-16, 1e-15, 1e-14 1/s"""
    d = [h for h in glob.glob(os.path.join(data_dir, 'data', '*', run))
         if os.path.exists(os.path.join(h, 'competence.csv'))][0]
    with open(os.path.join(d, 'competence.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['pore_pressure'] == pore_pressure]
    col = lambda k: np.array([float(r[k]) for r in rows])
    return dict(t=col('t_kyr'), lo=col('F_strain_1e-16_N_per_m'), mid=col('F_strain_1e-15_N_per_m'),
                hi=col('F_strain_1e-14_N_per_m'))


def recovery_time(t, F):
    """First time (kyr) after the strength minimum at which F rises back above 1 TN/m (linear interpolation)."""
    i0 = int(np.argmin(F))
    j = np.where(F[i0:] >= COMPETENCE)[0]
    if not j.size or j[0] == 0:
        return np.nan
    k = i0 + j[0]
    return t[k - 1] + (COMPETENCE - F[k - 1]) * (t[k] - t[k - 1]) / (F[k] - F[k - 1])


def main():
    a = fs.common_args(__doc__.splitlines()[0]).parse_args()
    fs.apply()
    fig = plt.figure(figsize=(fs.WIDTH['double'], 2.9))
    axA = fig.add_axes([0.075, 0.15, 0.40, 0.75])
    axB = fig.add_axes([0.58, 0.15, 0.40, 0.75])
    out = []

    # (a) Venus permeability
    ins = axA.inset_axes([0.20, 0.62, 0.32, 0.33])
    for k in ('K16', 'K15', 'K14'):
        run = f'V_R200_NC_{k}_LH_2MYR'
        d = read_competence(a.data_dir, run)
        tc = recovery_time(d['t'], d['mid'])
        axA.plot(d['t'], d['mid'] * SC, ls=fs.VENUS_DASH, color=fs.KCOL[k], lw=1.0,
                 label=f'Venus $k$ = {fs.KLAB[k]} m²: {tc:.0f} kyr')
        ins.plot(d['t'], d['mid'] * SC, ls=fs.VENUS_DASH, color=fs.KCOL[k], lw=1.1)
        ins.axvline(tc, color=fs.KCOL[k], lw=0.6, ls=':')
        out.append(('V_R200_NC_%s_LH_2MYR, simulated Pf' % k, tc, None, None))
    axA.axhline(1.0, color='k', ls=fs.REF_DOT, lw=0.6)
    axA.set_xlim(0, 2000); axA.set_ylim(0, 1.8)
    ins.axhline(1.0, color='0.35', ls=':', lw=0.8)
    ins.set_xlim(1060, 1130); ins.set_ylim(0.97, 1.03); ins.tick_params(labelsize=7, length=2)
    axA.indicate_inset_zoom(ins, edgecolor='0.5', lw=0.6)
    axA.set_xlabel('Time (kyr)'); axA.set_ylabel('Integrated yield strength (TN m$^{-1}$)')
    axA.legend(loc='lower right', handlelength=3.6, title='recovery at $10^{-15}$ s$^{-1}$', title_fontsize=7)

    # (b) simulated versus hydrostatic pore pressure
    for run, col, ls, lab in (('E_R200_K16_NC_LH', fs.KCOL['K16'], '-', 'Earth $10^{-16}$ m²'),
                              ('V_R200_NC_K15_LH_2MYR', fs.KCOL['K15'], fs.VENUS_DASH, 'Venus $10^{-15}$ m$^{2}$')):
        for pf, alpha in (('simulated', 1.0), ('hydrostatic', 0.6)):
            d = read_competence(a.data_dir, run, pf)
            tc, thi, tlo = (recovery_time(d['t'], d[key]) for key in ('mid', 'hi', 'lo'))
            axB.plot(d['t'], d['mid'] * SC, ls=ls if pf == 'simulated' else (0, (1, 1.2)), color=col, lw=1.0,
                     alpha=alpha, label='%s, %s $P_f$: %.0f (%.0f–%.0f)' % (lab, pf, tc, min(thi, tlo), max(thi, tlo)))
            out.append(('%s, %s Pf' % (run, pf), tc, min(thi, tlo), max(thi, tlo)))
    axB.axhline(1.0, color='k', ls=fs.REF_DOT, lw=0.6)
    axB.set_xlim(0, 2000); axB.set_ylim(0, 2.9)
    axB.set_xlabel('Time (kyr)'); axB.set_ylabel('Integrated yield strength (TN m$^{-1}$)')
    axB.legend(loc='upper right', handlelength=3.6,
               title='recovery (kyr) at $10^{-15}$ s$^{-1}$ ($10^{-14}$–$10^{-16}$ s$^{-1}$)', title_fontsize=7)
    for ax in (axA, axB):
        ax.text(1985, 1.0, 'competence limit\n(1 TN m$^{-1}$)', ha='right', va='center', fontsize=7, color='0.2',
                linespacing=1.0, bbox=fs.label_box(), zorder=6)
        fs.box(ax)
    fs.panel(axA, 'a'); fs.panel(axB, 'b')
    fs.save(fig, 'ED_fig4_strength_sensitivity', a.out_dir)

    print('\nCompetence recovery at 200 mm/yr (kyr after the start of emplacement; strain rate 1e-15 1/s, '
          'range 1e-14..1e-16 1/s)')
    for lab, tc, lo, hi in out:
        print('  %-44s %.0f' % (lab, tc) + ('' if lo is None else ' (%.0f-%.0f)' % (lo, hi)))


if __name__ == '__main__':
    main()
