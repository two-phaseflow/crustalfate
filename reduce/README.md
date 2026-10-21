# reduce/ — from raw model output to the archived time series

`reduce_vtu.py` turns the complete CSMP++ output archived for the Fig. 1 reference pair (`data/full_output/<run>/`
in the Zenodo record: gzipped VTU files, one per output frame) into the per-frame time series used by the figure
scripts (`timeseries.csv`): mobile-melt, crystal-mush and induced-ductile volumes, the mean crystallinity of the
mobile melt, the maximum temperature in the sill complex (time to solidus), and the sill-complex enthalpy and volume
(effective Nusselt number). The definitions are written out in the script header and in the Zenodo README; they are
the ones used for every run in the record.

    python reduce/reduce_vtu.py --data-dir zenodo_data --run E_R50_NC_MI_LH --check
    python reduce/reduce_vtu.py --data-dir zenodo_data --run V_R50_NC_MI_LH_2MYR --check

`--check` compares the result frame by frame with the archived `timeseries.csv` of the same run and prints the
largest absolute and relative difference per column. For both runs every archived row is reproduced identically to
its printed precision (357 of 357 Earth rows, 331 of 331 Venus rows). The Earth VTU archive holds one more frame
(1501 kyr) than its time series, written by the still-running simulation between the two exports; it is reduced but
has no archived counterpart. Requires the `vtk` package (in `environment.yml`); each run takes well under a minute
on a few cores.
