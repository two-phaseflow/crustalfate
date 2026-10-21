#!/usr/bin/env python3
"""
fig3.py  --  CrustalFate main-text Figure 3 (fluxibility) and the planetary-pressure
comparison quoted in the text.

Self-contained: the ONLY inputs are fluid-property calls to CoolProp. No simulation
output and no data files are read, so nothing needs to be archived alongside it.

Fluxibility (Jupp & Schultz, 2000, Nature 403, 880):

    F = (rho0 - rho) * rho * (h - h0) / mu                      [kg m^-3 s^-1]

where, at a given pressure P,
    rho  [kg m^-3]      density of the pore fluid at the fluid (plume) temperature T,
    h    [J kg^-1]      specific enthalpy of the pore fluid at T,
    mu   [Pa s]         dynamic viscosity of the pore fluid at T,
    rho0, h0            the same density and enthalpy at the recharge (surface)
                        temperature T0 and the same pressure P.
F is a comparative measure of buoyant advective heat-transport capacity; only relative
values are physically meaningful.

Fluid properties come from CoolProp (default HEOS backend):
    Water -> IAPWS-95 ;   CO2 -> Span & Wagner (1996).
Pin CoolProp==7.2.0 (see environment.yml).

Run:
    python fig3.py [--out-dir figures]

Output (written to --out-dir): fig3_fluxibility_25MPa.pdf/.png and fluxibility_numbers.txt.
--data-dir is accepted for a uniform interface with the other figure scripts but not used.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import figstyle as fs

fs.apply()
args = fs.common_args('Figure 3: fluxibility of H2O and CO2 at 25 MPa').parse_args()
MM = fs.MM
W2 = fs.WIDTH['double']               # Nature double-column width (inches)
FLUX_CMAP = 'YlOrRd'
FLUX_LEVELS = np.arange(12.5, 16.5001, 0.5)     # log10 F contour levels: 12.5..16.5 step 0.5
DEG = "\u00b0"
HALO = fs.halo()

# ----------------------------------------------------------------------- fluxibility
T = np.linspace(5, 800, 400)          # x axis: fluid (plume) temperature, deg C  (400 pts)
T0 = np.linspace(5, 490, 240)         # y axis: surface (recharge) temperature, deg C (240 pts)


def _prop(fluid, Ta, P):
    """Density [kg/m3], specific enthalpy [J/kg], dynamic viscosity [Pa s] at T=Ta (deg C), P (Pa)."""
    r = np.array([CP.PropsSI('D', 'T', t + 273.15, 'P', P, fluid) for t in Ta])
    h = np.array([CP.PropsSI('H', 'T', t + 273.15, 'P', P, fluid) for t in Ta])
    m = np.array([CP.PropsSI('V', 'T', t + 273.15, 'P', P, fluid) for t in Ta])
    return r, h, m


def logF_grid(fluid, P):
    """log10 of the fluxibility field over (T0, T), masked to the buoyant branch T > T0."""
    r, h, m = _prop(fluid, T, P)
    r0, h0, _ = _prop(fluid, T0, P)
    F = (r0[:, None] - r[None, :]) * r[None, :] * (h[None, :] - h0[:, None]) / m[None, :]
    return np.log10(np.where((T[None, :] > T0[:, None]) & (F > 0), F, np.nan))


def F_point(fluid, P, t, t0):
    """Fluxibility [kg m^-3 s^-1] for a single (fluid T = t, recharge T0 = t0, pressure P)."""
    r0 = CP.PropsSI('D', 'T', t0 + 273.15, 'P', P, fluid)
    h0 = CP.PropsSI('H', 'T', t0 + 273.15, 'P', P, fluid)
    r = CP.PropsSI('D', 'T', t + 273.15, 'P', P, fluid)
    h = CP.PropsSI('H', 'T', t + 273.15, 'P', P, fluid)
    m = CP.PropsSI('V', 'T', t + 273.15, 'P', P, fluid)
    return (r0 - r) * r * (h - h0) / m


# ----------------------------------------------------------------------- the figure
# Fig. 3: BOTH fluids at a common 25 MPa (both supercritical, single-phase everywhere).
P_FIG = 25e6
Lw = logF_grid('Water', P_FIG)
Lc = logF_grid('CO2', P_FIG)


def _draw(ax, L, letter, title, Tc):
    cs = ax.contourf(T, T0, L, levels=FLUX_LEVELS, cmap=FLUX_CMAP, extend='min')
    cs.set_rasterized(True)                                  # keep the PDF small
    ax.axvline(Tc, color='0.25', lw=0.6, ls=':')             # critical temperature (fluid-T axis)
    ax.text(Tc + 8, 486, '$T_c$ = %d %sC' % (Tc, DEG), color='k', fontsize=7,
            va='top', ha='left', path_effects=HALO)
    ax.plot([5, 490], [5, 490], color='0.4', lw=0.6)         # buoyant-branch boundary T = T0
    ax.set_xlim(5, 800); ax.set_ylim(5, 490)
    ax.set_xlabel('Fluid temperature (%sC)' % DEG)
    ax.set_ylabel('Surface (recharge) temperature (%sC)' % DEG)
    ax.tick_params(labelleft=True)
    ax.set_title(title)
    fs.panel(ax, letter)
    fs.box(ax)                                               # full frame, as in the manuscript figure
    return cs


fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.35), constrained_layout=True)
cs = _draw(a1, Lw, 'a', 'H$_2$O  (25 MPa)', 374)
_draw(a2, Lc, 'b', 'CO$_2$  (25 MPa)', 31)
cb = fig.colorbar(cs, ax=[a1, a2], ticks=FLUX_LEVELS, aspect=28, pad=0.015)
cb.set_label('log$_{10}$ fluxibility   (kg m$^{-3}$ s$^{-1}$)', fontsize=8)
cb.ax.set_yticklabels([(f'{v:.0f}' if abs(v - round(v)) < 1e-6 else f'{v:.1f}') for v in FLUX_LEVELS])
cb.ax.tick_params(labelsize=7)

# freeze the layout before adding the angled label so its on-screen angle is exact
fig.canvas.draw()
fig.set_layout_engine('none')
for _ax in (a1, a2):
    p1 = _ax.transData.transform((60, 60))
    p2 = _ax.transData.transform((440, 440))
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    _ax.text(238, 282, r'$T_\mathrm{surface} > T_\mathrm{fluid}$', rotation=ang,
             rotation_mode='anchor', color='0.45', fontsize=7, ha='center', va='center')

fs.save(fig, 'fig3_fluxibility_25MPa', args.out_dir)

# ---------------------------------------------------- numbers quoted in the paper
L = np.log10


def rr(num, den):
    return "ratio %.2f (log10 %.2f)" % (num / den, L(num / den))


out = []
out.append("CrustalFate fluxibility numbers")
out.append("CoolProp %s ; Water = IAPWS-95, CO2 = Span & Wagner (1996)" % CP.get_global_param_string("version"))
out.append("F = (rho0 - rho) * rho * (h - h0) / mu   [kg m^-3 s^-1];  rho0,h0 at recharge temperature T0.")
out.append("")
out.append("PLANETARY-PRESSURE COMPARISON  (the 'factor of ~7' in the text and the ED3 legend)")
c25 = F_point('CO2', 25e6, 600, 460)
c93 = F_point('CO2', 9.3e6, 600, 460)
out.append("  CO2, recharge 460 C, fluid 600 C, 25.0 MPa : F = %.3e  (log10 %.3f)" % (c25, L(c25)))
out.append("  CO2, recharge 460 C, fluid 600 C,  9.3 MPa : F = %.3e  (log10 %.3f)" % (c93, L(c93)))
out.append("  ratio (25 MPa / 9.3 MPa) = %.1f   (CO2 at its own 9.3 MPa is ~7x lower)" % (c25 / c93))
out.append("")
out.append("FIGURE 3  (both fluids at 25 MPa; fluid/plume temperature 600 C unless noted)")
Tg = np.arange(11, 801, 1.0)
for fl in ['Water', 'CO2']:
    Fg = np.array([F_point(fl, 25e6, t, 10.0) for t in Tg])
    i = int(np.nanargmax(Fg))
    out.append("  %-5s max F = %.3e (log10 %.2f) at fluid T = %d C ; F(600) = %.3e (log10 %.2f) ; max/F(800) = %.2f"
               % (fl, Fg[i], L(Fg[i]), int(Tg[i]),
                  F_point(fl, 25e6, 600, 10), L(F_point(fl, 25e6, 600, 10)),
                  Fg[i] / F_point(fl, 25e6, 800, 10)))
wc, wh = F_point('Water', 25e6, 600, 10), F_point('Water', 25e6, 600, 460)
cc, ch = F_point('CO2', 25e6, 600, 10), F_point('CO2', 25e6, 600, 460)
out.append("  surface-temperature effect, water (recharge 10 vs 460 C): " + rr(wc, wh))
out.append("  surface-temperature effect, CO2   (recharge 10 vs 460 C): " + rr(cc, ch))
out.append("  molecule effect, cold surface (water / CO2 at 10 C) : " + rr(wc, cc))
out.append("  molecule effect, hot surface  (water / CO2 at 460 C): " + rr(wh, ch))
out.append("  combined, cold water / hot CO2 : " + rr(wc, ch))
with open(os.path.join(args.out_dir, 'fluxibility_numbers.txt'), 'w') as f:
    f.write("\n".join(out) + "\n")

print("\n".join(out))
print("wrote", os.path.join(args.out_dir, 'fluxibility_numbers.txt'))
