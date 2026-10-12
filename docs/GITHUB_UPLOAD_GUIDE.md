# GitHub upload guide

## Recommended repository name

`shuttlecock-pinn-adaptive-collocation`

## Web-interface workflow

1. Sign in to GitHub.
2. Create a new **public** repository with the recommended name (or another stable, descriptive name).
3. Do not initialize it with a README, license, or `.gitignore` if you plan to upload this prepared package as-is, because those files are already included.
4. Upload the **contents** of this repository-ready folder, not the outer ZIP itself.
5. Commit the files to the default branch (`main`).
6. Check that `README.md` renders correctly and that GitHub shows the `CITATION.cff` citation panel.
7. Before the public release, resolve `LICENSES_TO_CONFIRM.md` and add the team-approved license files.
8. Add the eventual Zenodo DOI badge and DOI link to the top of `README.md` after Zenodo has archived the release.

## Recommended first release

Use tag `v1.0.0` and title:

`v1.0.0 - Publication reproducibility package`

Suggested release notes:

> Frozen reproducibility package supporting the 270-run MC-R/LHS-R/ALHS-R shuttlecock PINN study. Includes publication source code, run-level numerical data, experiment metadata, and the statistical supplementary PDF. This release corresponds to the version submitted to the International Journal of Applied and Computational Mathematics.

If you use the Zenodo-GitHub integration, enable the repository in Zenodo **before** publishing the GitHub release.
