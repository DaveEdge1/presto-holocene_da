# Changelog

All notable changes to this template are documented here. This file also
serves as the record of modifications relative to the upstream
[Holocene-Reconstruction/Holocene-code](https://github.com/Holocene-Reconstruction/Holocene-code),
as required by Section 5(a) of the GNU General Public License v3.

This project adapts and builds upon the Holocene paleoclimate data
assimilation code of Erb et al. (2022). The reconstruction algorithm and
associated scientific code originate with the upstream authors; the
modifications below were made to package that code as a reproducible,
customizable template driven by PReSto.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## v0.0.0 - 2026-06-04

### Added
- Containerized execution environment (`Dockerfile`,
  `davidedge/lipd_webapps:holocene_da`) so reconstructions run in a fixed,
  reproducible software stack.
- GitHub Actions workflow (`holocene_da.yml`) that orchestrates the full
  pipeline: proxy-data acquisition, model-prior preparation, and the
  data assimilation run.
- YAML-based parameterization (`config/user_config.yml`) exposing
  reconstruction settings (time resolution, age range, proxy archives,
  localization radius, etc.) for customization without editing source code.
- PReSto-driven data query layer (`query_params.json`) that triggers the
  workflow and selects proxy observations from LiPDverse, via either an
  archived compilation (e.g., Temperature 12k) or a filtered map-based query
  using the `lipdGenerator` container.
- Model-prior data handling: pre-processed HadCM3 and TraCE-21ka data served
  as GitHub release assets at standard resolutions (10–1000 yr), with
  automatic fallback to download original data from Zenodo for non-standard
  resolutions.
- Visualization workflow (`visualize.yml`) calling the `presto-viz` reusable
  workflow to generate an interactive visualization deployed to GitHub Pages.
- Release workflow (`release-recon.yml`) that archives each successful
  reconstruction as a tagged GitHub Release (xz-compressing or splitting
  NetCDFs that exceed the asset size limit), with a companion
  `merge-recon-netcdf.yml` workflow to reassemble split assets.
- `CITATION.cff` with software citation metadata and a `preferred-citation`
  pointer to the underlying method paper.
- This `CHANGELOG.md` and a `NOTICE` file documenting provenance and
  modifications.

### Changed
- Restructured the upstream code layout to operate as a template repository
  (config/, scripts/, reference_data/) consumed programmatically by PReSto
  rather than run manually.
- Adapted input/output handling so proxy data and model priors are acquired
  and staged automatically by the workflow instead of being supplied manually.

### Notes
- See the Git commit history for line-level detail of all changes.
- The upstream commit/version this template is based on is recorded in
  `NOTICE`.
