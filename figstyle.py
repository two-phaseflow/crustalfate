"""Shared figure style for the CrustalFate figures (Nature format).

Every figure script does

    import figstyle as fs
    fs.apply()
    ...
    fs.panel(ax, 'a')
    fs.save(fig, 'fig2_earth_2panel', out_dir)

Sizes: figures are built at final print size (183 mm double column). Arial (fallback Liberation Sans) throughout,
axis labels 8 pt, ticks / legends / annotations 7 pt, panel letters 10 pt bold. PDF text stays editable (TrueType,
fonttype 42); PNG at 300 dpi from the same figure object.
"""
import os
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.text as mtext

MM = 1.0 / 25.4
WIDTH = {'single': 88 * MM, 'double': 183 * MM}          # inches

# crystallinity split used everywhere: mobile melt chi < 0.5, crystal mush 0.5 <= chi < 1
CHI_MOBILE = 0.5

# fixed palette
KCOL = {'K16': '#1f77b4', 'K5e16': '#17becf', 'K15': '#2ca02c', 'K5e15': '#ff7f0e',
        'K14': '#d62728', 'K13': '#9467bd', 'EC': '#000000'}
KLAB = {'K16': r'$10^{-16}$', 'K5e16': r'$5\times10^{-16}$', 'K15': r'$10^{-15}$', 'K5e15': r'$5\times10^{-15}$',
        'K14': r'$10^{-14}$', 'K13': r'$10^{-13}$'}
MI_STYLE = {'MI2': dict(color='0.0', ls=(0, (5, 2.2)), marker='D'),    # depth-dependent 2 km: black diamonds, dashed
            'MI5': dict(color='0.45', ls=(0, (1, 1.6)), marker='s')}   # depth-dependent 5 km: grey squares, dotted
MI_NAME = {'MI5': 'depth dependent, 5 km', 'MI2': 'depth dependent, 2 km'}
RATE_COL = {50: '#1f77b4', 100: '#9467bd', 200: '#d62728', 400: '#ff7f0e'}
MOBILE_C, MUSH_C, DUCTILE_C, SOLID_C = '#c62828', '#f2a93b', '#9a8fd1', '#1f77b4'
VENUS_DASH = (0, (4, 2))         # Venus: dashed lines + squares; Earth: solid lines + circles
REF_DOT = (0, (1, 2))            # conduction-only as a reference level: black dotted
HALO_LW = 2.5


def halo():
    return [pe.withStroke(linewidth=HALO_LW, foreground='white')]


def apply():
    mpl.use('Agg')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'mathtext.sf': 'Arial', 'mathtext.cal': 'Arial', 'mathtext.default': 'it',
        'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 4,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': False, 'ytick.right': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'contour.linewidth': 0.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 150, 'savefig.dpi': 300, 'axes.grid': False,
    })


def box(ax, lw=0.6):
    """Full black frame on all four sides (ticks stay left/bottom)."""
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw); sp.set_color('k')


def label_box(**kw):
    """Thin box behind an in-panel label that sits on a line."""
    d = dict(boxstyle='square,pad=0.15', fc='white', ec='none'); d.update(kw); return d


def panel(ax, letter, x=-0.12, y=1.02, **kw):
    """Panel letter: 10 pt bold lower case at (-0.12, 1.02) in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left', **kw)


def editable_text(fig):
    """Matplotlib draws text that carries path effects (white halos) as outlines. Split each such text into a
    halo-only copy underneath and the plain text on top, so the PDF text stays selectable as text."""
    for t in list(fig.findobj(mtext.Text)):
        eff = t.get_path_effects()
        if not eff or not t.get_visible() or not t.get_text():
            continue
        strokes = [pe.Stroke(**e._gc, offset=e._offset) if isinstance(e, pe.withStroke) else e
                   for e in eff if isinstance(e, (pe.withStroke, pe.Stroke))]
        t.set_path_effects([])
        if not strokes:
            continue
        h = mtext.Text()
        h.update_from(t)
        h.set_text(t.get_text()); h.set_position(t.get_position()); h.set_rotation(t.get_rotation())
        h.set_rotation_mode(t.get_rotation_mode()); h.set_ha(t.get_ha()); h.set_va(t.get_va())
        h.set_clip_on(t.get_clip_on())
        if t.get_clip_on():
            h.set_clip_box(t.get_clip_box()); h.set_clip_path(t.get_clip_path())
        h.set_path_effects(strokes); h.set_zorder(t.get_zorder() - 1e-3)
        (t.axes if t.axes is not None else fig).add_artist(h)
        h.set_transform(t.get_transform())


def save(fig, stem, out_dir='figures'):
    """Write <out_dir>/<stem>.pdf and .png (300 dpi) at the true figure size."""
    os.makedirs(out_dir, exist_ok=True)
    editable_text(fig)
    pdf, png = os.path.join(out_dir, stem + '.pdf'), os.path.join(out_dir, stem + '.png')
    fig.savefig(pdf, facecolor='white')
    fig.savefig(png, dpi=300, facecolor='white')
    print('saved', pdf, '+', png)


def common_args(description):
    """--data-dir (the unzipped Zenodo record) and --out-dir for every figure script."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--data-dir', default='zenodo_data',
                    help='directory holding the unzipped Zenodo record (contains data/<suite>/...)')
    ap.add_argument('--out-dir', default='figures', help='where the figure is written')
    return ap
