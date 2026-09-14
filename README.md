# MPrisk

Main method code for the manuscript **“An Exposure–Characteristics–Hazard Framework Reveals Divergent Microplastic Risks in Chinese Freshwaters.”**

## Scope

The repository publishes the core analytical methods without releasing the study database:

1. E-C-H score calculation and contribution-scaling scenarios;
2. direct random-forest prediction;
3. cascaded E-C-H random-forest reconstruction;
4. five-fold cross-fitted partially linear DML.

The included CSV files are fully synthetic demonstrations. They contain no manuscript observations, DOI values, coordinates, or original environmental and socioeconomic measurements. Demo results only verify that the software runs; they do not reproduce or represent the paper's numerical findings.

## Run the demonstration

Python 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
python run_core_analysis.py --quick
```

To regenerate the deterministic synthetic data first:

```bash
python generate_demo_data.py
python run_core_analysis.py --quick
```

Generated results are written to `outputs/`, which is excluded from version control.

## Code map

- `src/ech.py`: E-C-H components, adaptive weights, interactions, final score, and scenarios.
- `src/ml.py`: direct random forest and cascaded E-C-H reconstruction model.
- `src/dml.py`: cross-fitted DML estimates of adjusted associations.
- `src/common.py`: released predictor lists and shared helpers.
- `config/ech_parameters.json`: score mappings and interaction coefficients.
- `generate_demo_data.py`: fixed-seed synthetic demo generator.
- `run_core_analysis.py`: one-command demonstration of the whole workflow.
- `tests/test_core.py`: schema, naming, normalization, and scenario checks.

## Apply the code to study data

Replace `data/demo_observations.csv` and `data/demo_covariates.csv` with private files using the same column schema, then update the two input paths in `run_core_analysis.py`. Score normalization is performed separately for rivers and lakes, matching the manuscript workflow.

Missing particle attributes mean “not reported,” not confirmed absence. For compatibility with the historical calculation, unreported composition fractions contribute zero to the weighted component; explicit reporting flags are returned with the scores.


## Version note

`config/ech_parameters.json` freezes the parameter values used by the released implementation. A few earlier working or supplementary-table versions contained rounded or revised values, so small differences may occur across historical versions.

The cascade metric is an internal reconstruction check and should not be interpreted as an independent out-of-sample improvement over the direct model. The compact DML module shows one five-fold cross-fitting run; repeated-fold sensitivity analysis can be added by rerunning it across seeds.

## Data availability

The study database is not distributed in this public repository. Data-access conditions can be stated in the manuscript's Data Availability section.
