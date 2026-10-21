"""Extended Data Fig. 3: the crystallinity-temperature law of the simulations compared with rhyolite-MELTS.

    python ed_fig3.py --data-dir <unzipped Zenodo record> [--out-dir figures]

(a) equilibrium and (b) fractional crystallization of a MORB starting composition. Prints the RMS difference between
each MELTS curve and the law.
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

T_SOLIDUS, T_LIQUIDUS, B_EXP = 950.0, 1250.0, 2.28
EQ = [('E_H2O02wt_100MPa_Eq', '0.2 wt% H$_2$O, 100 MPa', '#6baed6', '-'),
      ('E_H2O02wt_200MPa_Eq', '0.2 wt% H$_2$O, 200 MPa', '#08519c', '-'),
      ('E_H2O0wt_100MPa_Eq', 'dry, 100 MPa', '#6baed6', (0, (1, 1.5))),
      ('E_H2O0wt_200MPa_Eq', 'dry, 200 MPa', '#08519c', (0, (1, 1.5)))]
FR = [('E_H2O02wt_200MPa', '0.2 wt% H$_2$O, 200 MPa', '#6baed6', '-'),
      ('E_H2O02wt_500MPa', '0.2 wt% H$_2$O, 500 MPa', '#08519c', '-'),
      ('E_H2O05wt_200MPa', '0.5 wt% H$_2$O, 200 MPa', '#6baed6', (0, (4, 2))),
      ('E_H2O05wt_500MPa', '0.5 wt% H$_2$O, 500 MPa', '#08519c', (0, (4, 2)))]


def chi_law(T):
    """crystallinity of the simulations: 1 - ((T - T_sol) / (T_liq - T_sol))^2.28, clipped to [0, 1]"""
    return 1.0 - np.clip((T - T_SOLIDUS) / (T_LIQUIDUS - T_SOLIDUS), 0, 1) ** B_EXP


def read_melts(data_dir, stem):
    with open(os.path.join(data_dir, 'data', 'reference_melts', stem + '.csv')) as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r['T_C']) for r in rows]),
            np.clip(np.array([float(r['crystallinity']) for r in rows]), 0, 1))


def main():
    a = fs.common_args(__doc__.splitlines()[0]).parse_args()
    fs.apply()
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH['double'], 2.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.92, wspace=0.08)
    Tl = np.linspace(900, 1300, 400)
    rms_out = []
    for ax, runs, title in zip(axs, (EQ, FR), ('equilibrium crystallization', 'fractional crystallization')):
        ax.axhline(0.5, color='0.5', lw=0.6, ls=fs.REF_DOT)
        ax.text(910, 0.5, 'lock-up, χ = 0.5', fontsize=7, color='0.2', ha='left', va='center',
                bbox=fs.label_box(), zorder=6)
        for stem, lab, col, ls in runs:
            T, x = read_melts(a.data_dir, stem)
            rms = np.sqrt(np.mean((x - chi_law(T)) ** 2))
            rms_out.append((title, lab, rms))
            ax.plot(T, x, color=col, ls=ls, lw=1.0, label='%s (RMS %.2f)' % (lab, rms))
        ax.plot(Tl, chi_law(Tl), color='k', lw=1.6, label='model law', zorder=5)
        ax.set_xlim(900, 1300); ax.set_ylim(0, 1.02)
        ax.set_xlabel('Temperature (°C)')
        ax.set_title(title, fontsize=7, loc='center', pad=3)
        ax.legend(loc='lower left', handlelength=2.6, labelspacing=0.3)
        fs.box(ax)
    axs[0].set_ylabel('Crystallinity χ')
    fs.panel(axs[0], 'a'); fs.panel(axs[1], 'b')
    fs.save(fig, 'ED_fig3_melts_chi_T', a.out_dir)
    print('\nRMS difference between rhyolite-MELTS and the model crystallinity law')
    for title, lab, rms in rms_out:
        print('  %-28s %-26s %.2f' % (title, lab.replace('$_2$', '2'), rms))


if __name__ == '__main__':
    main()
