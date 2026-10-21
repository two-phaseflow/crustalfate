#!/usr/bin/env python3
"""Reduce archived CSMP++ VTU output to the per-frame time series of the Zenodo record (timeseries.csv).

Input:  <data-dir>/data/full_output/<run>/{Initial.vtu.gz, Variables_<yr>[_pre|_post].vtu.gz}
Output: a CSV with the columns of <data-dir>/data/<suite>/<run>/timeseries.csv

    python reduce/reduce_vtu.py --data-dir zenodo_data --run V_R50_NC_MI_LH_2MYR --out reduced.csv --check

Definitions (identical to the archived files, see the Zenodo README):
  mobile melt      chi < 0.5, whole domain
  crystal mush     0.5 <= chi < 1, whole domain
  induced ductile  chi = 1 and T > 550 C at depths that were NOT already between 550 C and the 950 C solidus on the
                   run's initial far-field geotherm (Initial.vtu, |x - 15 km| > 12 km, 121 depth levels)
  whole-domain volumes use the true control-volume (lumped triangle) areas x 2 pi |x - 15 km| / 2;
  sill-mask quantities (H_sill_J, V_sill_m3) use a uniform 200 m x 200 m nodal area x 2 pi |x - 15 km| / 2,
  H = sum rho V [c T + L (1 - chi)] with rho 2800 kg m-3, c 1000 J kg-1 K-1, L 4.2e5 J kg-1, reference 0 C.
"""
import os
import re
import csv
import gzip
import glob
import argparse
import tempfile
from multiprocessing import Pool

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as tn

RHO_ROCK, CP_ROCK, L_LATENT = 2800.0, 1000.0, 4.2e5
T_SOL, T_DUCT, CHI_MOBILE = 950.0, 550.0, 0.5
MESH_H, AXIS_X, CELL_DX = 15000.0, 15000.0, 200.0
COLS = ['t_kyr', 'frame', 'mobile_melt_km3', 'crystal_mush_km3', 'induced_ductile_km3', 'mean_chi_mobile',
        'T_max_sill_C', 'H_sill_J', 'V_sill_m3']


def read_gz(path):
    """Read a gzipped VTU (decompressed to a temporary file; the VTK XML reader needs a file name)."""
    with tempfile.NamedTemporaryFile(suffix='.vtu', delete=False) as tmp:
        with gzip.open(path, 'rb') as f:
            tmp.write(f.read())
        name = tmp.name
    try:
        r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(name); r.Update()
        return r.GetOutput()
    finally:
        os.remove(name)


def nodal_area(g):
    """Per-node lumped area (m2): each triangle contributes a third of its area to each of its nodes."""
    pts = tn(g.GetPoints().GetData())[:, :2]
    conn = tn(g.GetCells().GetConnectivityArray()).reshape(-1, 3)
    a, b, c = pts[conn[:, 0]], pts[conn[:, 1]], pts[conn[:, 2]]
    tri = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    area = np.zeros(len(pts))
    for k in range(3):
        np.add.at(area, conn[:, k], tri / 3.0)
    return area, pts[:, 0], pts[:, 1]


def ambient_geotherm(initial_path):
    """Initial far-field temperature vs depth (depth-bin centres, T), as used for the ductile exclusion."""
    g = read_gz(initial_path)
    _, x, y = nodal_area(g)
    depth = MESH_H - y
    zg = np.linspace(0, MESH_H, 121)
    zc = 0.5 * (zg[:-1] + zg[1:])
    far = np.abs(x - AXIS_X) > 12000.0
    idx = np.clip(np.digitize(depth, zg) - 1, 0, len(zg) - 2)
    T0 = tn(g.GetPointData().GetArray('temperature'))
    amb = np.array([T0[far & (idx == b)].mean() if (far & (idx == b)).any() else np.nan for b in range(len(zc))])
    ok = ~np.isnan(amb)
    return zc[ok], amb[ok]


def frames(run_dir):
    out = []
    for f in glob.glob(os.path.join(run_dir, 'Variables_*.vtu.gz')):
        m = re.match(r'Variables_(\d+)(_pre|_post)?\.vtu\.gz$', os.path.basename(f))
        if m:
            out.append((int(m.group(1)), {'_pre': 0, None: 1, '_post': 2}[m.group(2)], f))
    return sorted(out)


def reduce_frame(args):
    y, kind, path, zc, amb = args
    g = read_gz(path)
    area, x, yy = nodal_area(g)
    w = 2 * np.pi * np.abs(x - AXIS_X) / 2
    av = area * w                                              # true control-volume volumes (m3)
    avc = CELL_DX * CELL_DX * w                                # uniform-area weighting for the sill-mask budget
    Tpri = np.interp(MESH_H - yy, zc, amb)
    amb_ductile = (Tpri > T_DUCT) & (Tpri < T_SOL)
    pd = g.GetPointData()
    T = tn(pd.GetArray('temperature'))
    chi = np.clip(tn(pd.GetArray('crystallinity')), 0, 1)
    sc = pd.GetArray('sill complex')
    mask = (tn(sc) > 0.5) if sc is not None else np.zeros(T.size, bool)
    solid = chi >= 1.0 - 1e-9
    mob = chi < CHI_MOBILE
    H = float(np.sum(RHO_ROCK * avc[mask] * (CP_ROCK * T[mask] + L_LATENT * (1.0 - chi[mask]))))
    mchi = float((av[mob] * chi[mob]).sum() / av[mob].sum()) if mob.any() else np.nan
    tmax = float(T[mask].max()) if mask.any() else np.nan
    return ('%.4f' % (y / 1000.0), ('pre', 'plain', 'post')[kind],
            '%.6g' % (av[mob].sum() / 1e9), '%.6g' % (av[(~mob) & (~solid)].sum() / 1e9),
            '%.6g' % (av[(T > T_DUCT) & solid & ~amb_ductile].sum() / 1e9),
            '' if not np.isfinite(mchi) else '%.5f' % mchi, '' if not np.isfinite(tmax) else '%.3f' % tmax,
            '%.8e' % H, '%.8e' % avc[mask].sum())


def suite_dir(data_dir, run):
    hits = [d for d in glob.glob(os.path.join(data_dir, 'data', '*', run)) if os.sep + 'full_output' + os.sep not in d]
    return hits[0] if hits else None


def check(rows, ref_path):
    ref = list(csv.DictReader(open(ref_path)))
    print('rows: reduced %d, archived %d' % (len(rows), len(ref)))
    key = lambda r: (r['t_kyr'], r['frame'])
    rd = {key(dict(zip(COLS, r))): dict(zip(COLS, r)) for r in rows}
    worst = {}
    missing = 0
    for r in ref:
        k = key(r)
        if k not in rd:
            missing += 1; continue
        for c in COLS[2:]:
            a, b = r[c], rd[k][c]
            if a == '' or b == '':
                d = (0.0, 0.0) if a == b else (np.inf, np.inf)
            else:
                fa, fb = float(a), float(b)
                d = (abs(fa - fb), abs(fa - fb) / max(abs(fa), 1e-30))
            worst[c] = max(worst.get(c, (0, 0)), d)
    print('archived frames missing from reduction: %d' % missing)
    for c in COLS[2:]:
        print('  %-22s max abs diff %.3g   max rel diff %.3g' % (c, *worst.get(c, (0, 0))))
    exact = sum(1 for r in ref if key(r) in rd and all(r[c] == rd[key(r)][c] for c in COLS[2:]))
    print('rows identical to printed precision: %d of %d' % (exact, len(ref)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', required=True, help='unzipped Zenodo record')
    ap.add_argument('--run', required=True, help='run name, e.g. V_R50_NC_MI_LH_2MYR')
    ap.add_argument('--out', default=None, help='output CSV (default <run>_reduced.csv)')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--check', action='store_true', help='compare with the archived timeseries.csv')
    a = ap.parse_args()
    rdir = os.path.join(a.data_dir, 'data', 'full_output', a.run)
    zc, amb = ambient_geotherm(os.path.join(rdir, 'Initial.vtu.gz'))
    jobs = [(y, k, p, zc, amb) for y, k, p in frames(rdir)]
    with Pool(a.jobs) as P:
        rows = P.map(reduce_frame, jobs, chunksize=4)
    out = a.out or a.run + '_reduced.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(COLS); w.writerows(rows)
    print('wrote', out, len(rows), 'frames')
    if a.check:
        sd = suite_dir(a.data_dir, a.run)
        check(rows, os.path.join(sd, 'timeseries.csv'))
