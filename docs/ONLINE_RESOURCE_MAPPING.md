# Mapping between Springer Online Resources, GitHub, and Zenodo

## Recommended publication architecture

### Online Resource 1 - statistical supplement
**Springer file:** `ESM_1.pdf`

Purpose: complete statistical tables and checkpoint-selection summaries.

Recommended handling:
- upload `ESM_1.pdf` directly to the Springer submission system as **Online Resource 1**;
- also archive the same file in the GitHub/Zenodo reproducibility package for persistence and discoverability.

### Online Resource 2 - numerical dataset and metadata
**Springer file:** `ESM_2.zip`

Purpose: exact run-level data and experiment metadata used in the manuscript.

Recommended handling:
- keep `ESM_2.zip` as an exact journal supplementary archive if the journal accepts it;
- the same contents are also provided in extracted form under `data/` for GitHub browsing;
- cite the Zenodo DOI in the manuscript Data Availability statement.

### Online Resource 3 - frozen source code
**Springer file:** `ESM_3.zip`

Purpose: exact frozen implementation and publication experiment runner.

Recommended handling:
- keep `ESM_3.zip` as an exact journal supplementary archive if desired;
- expose the same code in `code/` on GitHub for version control and readability;
- archive a tagged GitHub release in Zenodo, or include `ESM_3.zip` in a manual Zenodo reproducibility-package deposit.

## Recommended manuscript wording after DOI assignment

**Supplementary information**  
Online Resource 1 contains the complete omnibus Friedman tests, Holm-corrected paired Wilcoxon comparisons, matched rank-biserial effect sizes, collocation-budget analyses, cross-budget sample-efficiency comparisons, and checkpoint-selection summaries. The numerical dataset and frozen source code are provided as Online Resources 2 and 3 and are additionally archived in the public reproducibility repository cited in the Data and Code Availability statements.

**Data availability**  
The complete 270-run dataset, summary tables, integrity report, and experiment metadata supporting this study are available in Zenodo at `https://doi.org/<ZENODO_DOI>`.

**Code availability**  
The frozen source code used to implement the PINN models, sampling strategies, publication experiment, and evaluation pipeline is available at `https://github.com/<GITHUB_USERNAME>/<REPOSITORY_NAME>` and is archived in Zenodo at `https://doi.org/<ZENODO_DOI>`.

## Important distinction

The journal supplementary files and the repository are complementary, not competing copies:

- Springer Online Resources travel with the article;
- GitHub provides transparent version control and readable source code;
- Zenodo provides an immutable, citable archival snapshot with a DOI.
