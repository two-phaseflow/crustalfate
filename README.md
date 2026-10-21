# crustalfate

Code to reproduce every figure and every quoted number in

> Scott, S., Lourenço, D., Ikrar, J. S. K., Weis, P. & Stefánsson, A. Surface temperature controls fluid-mediated cooling of intrusive crust on Earth and Venus (in review).

The simulations model incremental basaltic sill emplacement into Earth crust with hydrothermal (H₂O) circulation and
into Venus crust with carbothermal (CO₂) circulation. Their reduced output is archived on Zenodo:
**[10.5281/zenodo.23539324](https://doi.org/10.5281/zenodo.23539324)** (CC BY 4.0).

## Contents

| File | Produces |
|---|---|
| `fig1.py` | Fig. 1: model set-up, Earth and Venus at 100 kyr, melt/mush/ductile volume histories |
| `fig2.py` | Fig. 2: Earth regime map and effective Nusselt number |
| `fig3.py` | Fig. 3: fluxibility of H₂O and CO₂ (fluid properties only, no data needed) |
| `fig4.py` | Fig. 4: time to solidus and crustal competence recovery |
| `ed_fig1.py` | Extended Data Fig. 1: Venus regime map and effective Nusselt number |
| `ed_fig2.py` | Extended Data Fig. 2: effect of seawater salinity |
| `ed_fig3.py` | Extended Data Fig. 3: crystallinity–temperature law vs rhyolite-MELTS |
| `ed_fig4.py` | Extended Data Fig. 4: competence recovery sensitivity to permeability and pore pressure |
| `figstyle.py` | Shared figure style (the only module the scripts share) |
| `reduce/` | From raw model output (VTU) to the archived time series, with a check (see `reduce/README.md`) |

Each script contains everything it needs, reads only the Zenodo record, writes the figure (PDF + 300 dpi PNG) to
`figures/`, and prints the numbers quoted in the paper that it is responsible for.

## Setup

```bash
git clone https://github.com/two-phaseflow/crustalfate.git
cd crustalfate
conda env create -f environment.yml && conda activate crustalfate     # or: pip install -r requirements.txt

# download and unzip the data record (≈550 MB; full_output.zip is only needed for reduce/)
mkdir zenodo_data && cd zenodo_data
for f in earth_uniform_k earth_depth_dependent venus seawater_tests conduction_only reference_melts full_output; do
  curl -L -O "https://zenodo.org/records/23539324/files/$f.zip?download=1" && unzip -q "$f.zip?download=1"
done
cd ..
```

Python 3.10 with numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.8, CoolProp 7.2.0 and vtk 9.6.0 (vtk is used only by
`reduce/`). Arial is used if installed; otherwise Liberation Sans.

## Reproducing the figures

```bash
python fig1.py    --data-dir zenodo_data
python fig2.py    --data-dir zenodo_data
python fig3.py
python fig4.py    --data-dir zenodo_data
python ed_fig1.py --data-dir zenodo_data
python ed_fig2.py --data-dir zenodo_data
python ed_fig3.py --data-dir zenodo_data
python ed_fig4.py --data-dir zenodo_data
```

Each runs in seconds (Fig. 1 ≈15 s).

## Where each number in the text comes from

| Quantity | Script |
|---|---|
| Volumes of mobile melt, mush and induced ductile rock for the Fig. 1 runs; peaks; times at which melt and mush disappear; inter-sill survival of mobile melt (onset 124 kyr, 11–23 km³ late in emplacement); mean χ of mobile melt | `fig1.py` |
| Intruded volume (initial pluton + 50 sills); conduction-only volumes at 50 mm yr⁻¹ and the crust-vs-fluid split of the mobile-melt contrast | `fig1.py` |
| Darcy-velocity scales at 100 kyr (Earth vs Venus) | `fig1.py` |
| Effective Nusselt numbers for Earth (uniform and depth-dependent permeability), supply limit, regime of each cell | `fig2.py` |
| Fluxibility values, including the factor of ~7 for CO₂ at 9.3 vs 25 MPa | `fig3.py` (also writes `figures/fluxibility_numbers.txt`) |
| Times to solidus; competence recovery times and their strain-rate ranges | `fig4.py` |
| Venus effective Nusselt numbers (range 1.00–1.02) and regime cells | `ed_fig1.py` |
| Seawater Nusselt numbers (1.62 / 1.27 / 1.17; 3.52 / 3.40 / 2.93) and volumes | `ed_fig2.py` |
| RMS misfit of the crystallinity law to rhyolite-MELTS | `ed_fig3.py` |
| Venus recovery for three permeabilities; simulated vs hydrostatic pore pressure | `ed_fig4.py` |

## Simulations

The simulations were run with CSMP++ (Complex Systems Modelling Platform), CrustalFate project, development tree
`co2_sandbox` (October 2026), with the latent-heat correction and CO₂ (Span & Wagner) and H₂O–NaCl
(Driesner & Heinrich) fluid modules. The CSMP++ source is not part of this repository; it is available from the
authors on request. Run settings for every simulation are listed in `runs.csv` of the Zenodo record.

## Notes

- Extended Data Fig. 2 in this repository has two panels (Nusselt number and volumes versus salinity).
- The Earth Fig. 1 run was still extending when the data were exported: its time series ends at 1496 kyr, and its
  archived VTU output contains one further frame (1501 kyr).

## Citation and licence

Code: MIT (`LICENSE`); see `CITATION.cff`. Data: CC BY 4.0, cite 10.5281/zenodo.23539324.
