# Synthetic demonstration data

`demo_observations.csv` and `demo_covariates.csv` are fully synthetic records generated with a fixed random seed by `generate_demo_data.py`. They contain no manuscript observations, DOI values, sampling locations, or original covariate measurements.

The files exist only to demonstrate the required schema and to make every released function executable. Results produced from them are software checks and must not be interpreted as scientific findings or compared with manuscript tables.

Researchers who obtain the study data should map their columns to this schema. Blank particle-composition fields are treated as zero contribution for compatibility with the historical calculation and are flagged in the E-C-H output.
