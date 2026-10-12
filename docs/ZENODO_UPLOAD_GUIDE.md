# Zenodo upload guide

Two valid workflows are available. For this manuscript, the **manual reproducibility-package deposit** is the most controllable because it lets you reserve a DOI before final manuscript submission.

## Recommended workflow: manual Zenodo deposit + GitHub as related source repository

1. Create/sign in to Zenodo, preferably using ORCID or GitHub.
2. Click **+ -> New upload**.
3. Upload a single release archive of this repository-ready package (for example `shuttlecock_pinn_reproducibility_v1.0.0.zip`), or upload the three journal ESM files plus the repository archive. If you want one simple citable object, the single repository ZIP is cleaner.
4. Under DOI, answer that the upload does not yet have a DOI and click **Get a DOI now!** to reserve one.
5. Keep the draft open. Copy the reserved DOI into the manuscript Data Availability/Code Availability statements and, if desired, the README and supplementary cover page.
6. Choose the resource type that best represents the package. Because data and software are both substantial, either:
   - use `Other` for a single mixed reproducibility package, or
   - split into separate `Dataset` and `Software` records if the team wants separately citable outputs.
7. Use version `1.0.0`.
8. Add creators in manuscript order. Link Wahyu Hardyanto to ORCID `0000-0002-2555-1437`. Add other ORCIDs only after they are verified.
9. Add affiliations for all creators.
10. Paste the title, description, and keywords from `zenodo_metadata_template.json`.
11. Add the GitHub repository URL as a related identifier after the repository exists.
12. Select the team-approved license(s). Zenodo supports multiple licenses for mixed-license records.
13. Set file visibility to **Public** unless the team intentionally needs an embargo.
14. Save the draft and use **Preview** to inspect the landing page.
15. Do not publish until author order, licenses, GitHub URL, repository content, and manuscript DOI wording have all been checked.
16. Click **Publish**. Zenodo registers the DOI when the record is published.
17. Replace all `<ZENODO_DOI>` placeholders in the manuscript with the final DOI and verify the DOI resolves.

## Alternative: GitHub-to-Zenodo automatic release archiving

1. Create and populate the public GitHub repository.
2. Sign in to Zenodo and connect the GitHub account.
3. In Zenodo, open the GitHub integration page, click **Sync now**, find the repository, and enable it.
4. Keep `CITATION.cff` in the GitHub repository. If you later add an active `.zenodo.json`, note that Zenodo will use `.zenodo.json` instead of `CITATION.cff` for release metadata.
5. Create a GitHub release tagged `v1.0.0`.
6. Zenodo ingests the release and creates an archived software record with a DOI.
7. Add the DOI back to the live GitHub README/CITATION metadata and to the manuscript.

## Recommendation for this paper

Use:

- Springer `ESM_1.pdf` as Online Resource 1;
- optionally retain `ESM_2.zip` and `ESM_3.zip` as exact journal supplements;
- GitHub for transparent source browsing and future version control;
- one Zenodo DOI for the frozen version corresponding to the submitted/accepted paper.
