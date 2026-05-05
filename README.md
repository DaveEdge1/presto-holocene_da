[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8274756.svg)](https://doi.org/10.5281/zenodo.8274756)

# PReSto Holocene DA Template

By [David Edge](https://orcid.org/0000-0001-6938-2850), [Michael Erb](https://orcid.org/0000-0002-1763-2522), [Nicholas McKay](https://orcid.org/0000-0003-3598-5113), [Deborah Khider](https://orcid.org/0000-0001-7501-8430), & [Julien Emile-Geay](https://orcid.org/0000-0001-5920-4751).

[PReSto](https://paleopresto.com) (Paleoclimate Reconstruction Storehouse) lowers the barriers to utilizing, reproducing, and customizing paleoclimate reconstructions. This repository is a template used by PReSto to run the [Holocene DA reconstruction](https://github.com/Holocene-Reconstruction/Holocene-code) via GitHub Actions.

## Holocene DA Method

This template reproduces and customizes the Holocene temperature reconstruction of [Erb et al. (2022)](https://doi.org/10.5194/cp-18-2599-2022), which uses offline paleoclimate data assimilation to reconstruct spatially complete temperature fields over the past 12,000 years.

Proxy observations are drawn from either:
- **Archived compilations** (e.g., Temperature 12k) downloaded directly from [LiPDverse](https://lipdverse.org)
- **Filtered selections** queried from LiPDverse via PReSto's interactive map interface

The prior is constructed from transient climate model simulations (HadCM3 and TraCE-21ka), regridded and time-averaged to the user-specified resolution. Pre-processed model data at standard resolutions (10–1000 yr) are stored as GitHub release assets; non-standard resolutions trigger a download of the original data from [Zenodo](https://zenodo.org/records/7407116).

The original reconstruction code is available at [Holocene-Reconstruction/Holocene-code](https://github.com/Holocene-Reconstruction/Holocene-code).

## File Structure

| Path | Purpose |
|------|---------|
| `config/user_config.yml` | Reconstruction parameters (overwritten per run by PReSto) |
| `query_params.json` | Data query filters (committed by PReSto to trigger the workflow) |

## Workflows

### `holocene_da.yml` — Holocene DA Reconstruction

Two-job pipeline triggered by a push to `query_params.json` or manual dispatch:

1. **prepare-data** — Acquires proxy data via one of two pathways:
   - *Archived*: downloads a pre-built compilation pickle from LiPDverse
   - *Filtered*: runs the `lipdGenerator` Docker container to query LiPDverse and produce a legacy pickle
2. **reconstruct** — Downloads pre-processed model data (or raw data from Zenodo for non-standard resolutions), runs the Holocene DA algorithm inside the `davidedge/lipd_webapps:holocene_da` Docker container, and commits results to the repository

### `visualize.yml` — Visualization

Triggered automatically after a successful `holocene_da.yml` run (or manually). Calls the [presto-viz](https://github.com/DaveEdge1/presto-viz) reusable workflow to generate an interactive visualization and deploys it to GitHub Pages.

## How to Use

1. **Fork or clone** this repository
2. Edit `config/user_config.yml` to customize reconstruction parameters (time resolution, age range, proxy archives, localization radius, etc.)
3. Push your changes; the workflow triggers automatically when `query_params.json` is updated, or run it manually from the **Actions** tab
4. Reconstruction results are saved as artifacts (90-day retention) and committed to the `results/` directory
5. Visualizations are deployed to the repository's GitHub Pages site
