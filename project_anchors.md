# Project anchors

Five claims from the course project. Each one names the split or the rule it depends on. Quote from here, then open the file under `context/hko/` if someone asks for the table. Do not refit these numbers on the sprint data, and do not cite them when the sprint question uses a different metric.

The published comparison is on the 2025 test year. Train is before 2024-07-01, validation is from 2024-07-01 until 2025-01-01, and test is from 2025-01-01. Thresholds were chosen on validation CSI. Test metrics are POD, FAR, CSI, ETS, and PR-AUC. Source: `context/hko/GRAPH_REPORT.md`.

1. The CoWIN station graph is undirected: mutual 4-nearest neighbours, plus any pair closer than 8 km. Gaussian weights use a bandwidth of 5.77 km, the median edge length, fixed from geography before any rainfall model was fit. The published graph has 39 nodes and 221 edges. Station 6110 was dropped for missing coordinates. Two connected components.

2. At a 1-hour lead, territory amber persistence CSI is 0.373 (POD 0.544, FAR 0.456, PR-AUC 0.303, base rate 0.0170). Inverse-distance neighbours reach CSI 0.385. Logistic regression sits beside persistence (own 0.371, with graph features 0.361). XGBoost is lower (own 0.127, with graph features 0.123). GraphSAGE CSI is 0.306.

3. At a 3-hour lead, neighbour features raise logistic CSI from 0.108 to 0.141 and XGBoost from 0.051 to 0.102. Inverse-distance neighbours (CSI 0.214) and GraphSAGE (CSI 0.204) are above persistence (CSI 0.192). A higher CSI for `features=graph` than `features=own`, in the same model family, is the graph effect in that report.

4. The 9-day forecast is one territory-wide maximum temperature. Mean absolute error rises from 1.54 C at lead 1 day (1095 days) to 2.20 C at lead 9 days (1087 days). Moran's I of the error stays near -0.129 at every lead, with a one-sided permutation p-value of 1. That is a stable station contrast, not clustered spatial error. Coordinates in that check were unverified.

5. Rainfall code `M` is missing and was never filled with zero. Amber, red, and black thresholds are 30, 50, and 70 mm in one hour. Clocks are Hong Kong time. Open-Meteo is a covariate, not the label. Source for the rules: `context/hko/data_dictionary_v0.md`. The early station feed had temperature and not a rain forecast; that limit is written up in `context/hko/EDA_report.md`.
