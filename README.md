# ECH

ECH combines external exposure (E), particle characteristics relevant to contact (C), and material-associated hazard (H). Fixed references put the components on comparable scales within each medium. Softmax weights vary with those coordinates. Exposure-anchored fusion produces a screening score from 0 to 100.

## Extension interfaces

- **Exposure:** Use `exposure(abundance, reference)` for compatible abundance measures, or supply a validated exposure coordinate for another indicator, medium, or monitoring program.
- **Characteristics:** `contact_component()` accepts size, shape and colour fractions. For additional contact attributes, including polymer composition, implement the revised component separately and calibrate its reference scale.
- **Hazard:** `hazard_component()` takes polymer fractions, category log coefficients and a reference mixture. Update these parameters as new hazard evidence becomes available, using the schema in `parameters.json`.
- **Fusion:** Pass calibrated E, C and H arrays to `fuse()`. Implement and validate alternative fusion rules independently before comparison.
- **Models:** `modeling.py` separates `design()` from `fit()`. Predictors and learners can change without changing ECH scoring. `dml.py` and `spatial.py` implement adjusted associations and spatial sensitivity.

## Usage

```sh
python -m pip install -r requirements.txt
python ech.py states.npz water
python modeling.py analysis.xlsx --rebuild-oof
python figures/render.py 1 --output outputs
```

Supply NPZ arrays `A`, `size`, `shape`, `color` and `polymer` in the configured category order. Use Python 3.13 for the prediction benchmark and Python 3.12 for DML and spatial diagnostics: fold assignment or numerical results can differ between dependency versions. Use each command's `--help` for other inputs.

## Data availability

See the article's Supplementary Materials for the observation inventory and published tables. No study data are included. Full reruns require author-supplied conditional states, model design and reference predictions, DML assignments and station linkage, and plot inputs. Figure 2 requires the base plate and ordination tables. Plotting reads the attribution and sensitivity summaries; `modeling.py --attribution-output` can regenerate the model attribution tables.
