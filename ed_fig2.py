"""Extended Data Fig. 2: effect of seawater salinity on hydrothermal cooling.

    python ed_fig2.py --data-dir <unzipped Zenodo record> [--out-dir figures]

(a) effective Nusselt number versus salinity, (b) mobile melt and crystal mush at the end of emplacement, for
100 mm/yr with k = 1e-15 m2 and 200 mm/yr with k = 1e-14 m2, each with pure water, 3.2 and 6.4 wt% NaCl.
Prints the Nu_eff values and volumes quoted in the paper.
"""
import csv
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import figstyle as fs

CASES = [(100, 'K15', r'100 mm yr$^{-1}$, $k$ = $10^{-15}$ m$^{2}$', fs.KCOL['K15'], 'o'),
         (200, 'K14', r'200 mm yr$^{-1}$, $k$ = $10^{-14}$ m$^{2}$', fs.KCOL['K14'], 's')]
SALS = [('', 0.0), ('_SW', 3.2), ('_SW64', 6.4)]       # run-name suffix, wt% NaCl


# ----------------------------------------------------------------------------------------------- data
def run_dir(data_dir, run):
    hits = [h for h in glob.glob(os.path.join(data_dir, 'data', '*', run))
            if os.path.exists(os.path.join(h, 'run_meta.json'))]
    return hits[0]


def load_run(data_dir, run):
    d = run_dir(data_dir, run)
    with open(os.path.join(d, 'timeseries.csv')) as f:
        rows = list(csv.DictReader(f))
    meta = json.load(open(os.path.join(d, 'run_meta.json')))
    return rows, meta


def heat_drained(rows, meta):
    """f_ext * P (W): heat extracted from the sill complex during emplacement per unit time.
    Supplied heat S = enthalpy of the initial pluton at emplacement + the enthalpy added by each sill
    (post - pre); f_ext = 1 - H(after the last sill) / S."""
    emp = meta['emplacement_duration_yr']
    pre, post = {}, {}
    for r in rows:
        t = round(float(r['t_kyr']) * 1000.0)
        if t > emp + 1:
            continue
        if r['frame'] == 'pre':
            pre[t] = float(r['H_sill_J'])
        elif r['frame'] == 'post':
            post[t] = float(r['H_sill_J'])
    inj = [post[t] - pre[t] for t in sorted(post) if t in pre]
    n_ev = max(int(round(emp / np.median(np.diff(sorted(post))))), len(inj))
    S = sum(inj) * n_ev / len(inj) + meta['H_initial_pluton_at_emplacement_J']   # rescale if events are missing
    f_ext = 1.0 - post[max(post)] / S
    return f_ext * S / (emp * 3.15576e7)


def end_volumes(rows, meta):
    """Mobile melt and crystal mush (km3, whole domain) at the last inter-sill trough of emplacement."""
    emp_kyr = meta['emplacement_duration_yr'] / 1000.0
    last = [r for r in rows if r['frame'] == 'pre' and float(r['t_kyr']) <= emp_kyr + 1e-3][-1]
    return float(last['mobile_melt_km3']), float(last['crystal_mush_km3'])


def main():
    a = fs.common_args(__doc__.splitlines()[0]).parse_args()
    fs.apply()
    nu, vol = {}, {}
    for r, k, _, _, _ in CASES:
        ctrl = heat_drained(*load_run(a.data_dir, f'EC_R{r}_NC_LH'))
        for suf, sal in SALS:
            rows, meta = load_run(a.data_dir, f'E_R{r}_{k}_NC{suf}_LH')
            nu[(r, sal)] = heat_drained(rows, meta) / ctrl
            vol[(r, sal)] = end_volumes(rows, meta)

    fig = plt.figure(figsize=(fs.WIDTH['double'], 2.7))
    axA = fig.add_axes([0.08, 0.16, 0.38, 0.72])
    axB = fig.add_axes([0.58, 0.16, 0.40, 0.72])
    sal = np.array([s for _, s in SALS])
    for r, k, lab, col, mk in CASES:
        y = [nu[(r, s)] for s in sal]
        axA.plot(sal, y, '-', color=col, lw=1.0, marker=mk, ms=4, label=lab)
        for s_, y_ in zip(sal, y):
            axA.annotate('%.2f' % y_, (s_, y_), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=7,
                         color=col, bbox=fs.label_box())
    axA.axhline(1.0, color='k', lw=0.9)
    axA.set_xticks(sal); axA.set_xlim(-0.6, 7.0); axA.set_ylim(0.8, 4.0)
    axA.set_xlabel('Salinity (wt% NaCl)'); axA.set_ylabel(r'Effective Nusselt number, $Nu$$_{\mathrm{eff}}$')
    axA.legend(loc='center right')
    fs.box(axA)
    wbar = 0.75
    for i, (r, k, lab, col, mk) in enumerate(CASES):
        for j, s_ in enumerate(sal):
            mob, mush = vol[(r, s_)]
            axB.bar(j + i * 4, mob, wbar, color=fs.MOBILE_C, ec='0.3', lw=0.3)
            axB.bar(j + i * 4, mush, wbar, bottom=mob, color=fs.MUSH_C, ec='0.3', lw=0.3)
        axB.text(1 + i * 4, 505, lab, ha='center', va='bottom', fontsize=7)
    axB.set_xticks([j + i * 4 for i in range(2) for j in range(3)])
    axB.set_xticklabels(['%g' % s_ for _ in range(2) for s_ in sal])
    axB.set_xlabel('Salinity (wt% NaCl)')
    axB.set_ylabel('Volume at end of emplacement (km³)')
    axB.set_ylim(0, 560)
    fs.box(axB)
    axB.legend(handles=[Patch(fc=fs.MUSH_C, ec='0.3', lw=0.3, label='crystal mush (0.5 ≤ χ < 1)'),
                        Patch(fc=fs.MOBILE_C, ec='0.3', lw=0.3, label='mobile melt (χ < 0.5)')],
               loc='upper right', bbox_to_anchor=(1.0, 0.88))
    fs.panel(axA, 'a'); fs.panel(axB, 'b')
    fs.save(fig, 'ED_fig2_seawater', a.out_dir)

    print('\nSeawater sensitivity (Nu_eff relative to the conduction-only run at the same rate; volumes at the '
          'end of emplacement)')
    for r, k, _, _, _ in CASES:
        for _, s in SALS:
            print('  %3d mm/yr k=%s  %.1f wt%%  Nu_eff %.2f  mobile %.0f km3  mush %.0f km3'
                  % (r, k, s, nu[(r, s)], *vol[(r, s)]))


if __name__ == '__main__':
    main()
