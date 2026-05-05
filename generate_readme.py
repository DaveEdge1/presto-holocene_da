"""Regenerate README.md for a custom PReSto Holocene DA reconstruction.

Reads `query_params.json`, `config/user_config.yml`, and (if present)
`cleaning_report.json` and `README_NOTES.md` from the repo root, and writes
a `README.md` that surfaces the run-specific data selection, data-cleaning
summary, reconstruction parameters, and any user-authored notes.

`README_NOTES.md` is the author's escape hatch — its contents are inserted
verbatim near the top of the README and survive regenerations. Everything
else in the README is a pure function of the input files, so re-running
with unchanged inputs is byte-stable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


TEMPLATE_REPO = "https://github.com/DaveEdge1/presto-holocene_da"
PRESTO_URL = "https://paleopresto.com"

MODE_DESCRIPTIONS = {
    "filtered": (
        "filtered (records hand-selected via PReSto's interactive map "
        "interface and queried from LiPDverse)"
    ),
    "archived": (
        "archived (a pre-built compilation pickle was downloaded directly "
        "from LiPDverse)"
    ),
}

MODEL_LABELS = {
    "hadcm3_regrid": "HadCM3 (regridded)",
    "trace_regrid": "TraCE-21ka (regridded)",
}


def _format_compilations(raw):
    """'CoralHydro2k-1_0_0,Pages2k' → 'CoralHydro2k 1.0.0, Pages2k'."""
    if not raw:
        return None
    out = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            name, _, version = token.partition("-")
            out.append(f"{name.strip()} {version.replace('_', '.').strip()}")
        else:
            out.append(token)
    return ", ".join(out)


def _split_csv(raw):
    """Parse a 'a,b,c' string (or list) into a clean list, dropping empties."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = str(raw).split(",")
    return [str(x).strip() for x in items if str(x).strip()]


def _bp_value(year):
    try:
        return int(float(year))
    except (TypeError, ValueError):
        return None


def _format_bp_range(raw):
    """'0,1000' (BP) → '0–1000 yr BP'; older year (larger BP) shown last."""
    parts = _split_csv(raw)
    if len(parts) < 2:
        return None
    a, b = _bp_value(parts[0]), _bp_value(parts[1])
    if a is None or b is None:
        return f"{parts[0]}–{parts[1]}"
    lo, hi = sorted((a, b))
    return f"{lo:,}–{hi:,} yr BP"


def _format_int(value, suffix=""):
    if value is None or value == "" or str(value).lower() == "none":
        return None
    try:
        return f"{int(float(value)):,}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _format_locrad(km):
    """'None' / null → 'no localization (global)'; numeric → '15,000 km'."""
    if km is None or km == "" or str(km).lower() == "none":
        return "no localization (global)"
    try:
        return f"{int(float(km)):,} km"
    except (TypeError, ValueError):
        return str(km)


def _format_models(raw):
    keys = _split_csv(raw)
    if not keys:
        return None
    return ", ".join(MODEL_LABELS.get(k, k) for k in keys)


def _format_seasons(raw):
    keys = _split_csv(raw)
    if not keys:
        return None
    pretty = {"annual": "annual", "summerOnly": "summer-only",
              "winterOnly": "winter-only"}
    return ", ".join(pretty.get(k, k) for k in keys)


def _format_archives(value):
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def summarize_cleaning(report):
    """Aggregate a PReSto cleaning_report.json (list of duplicate groups).

    Returns a dict with `groups`, `considered`, `kept`, `removed`, and
    `top_reason` (string or None) — or None if the report is unrecognized.
    """
    if not isinstance(report, list) or not report:
        return None
    groups = len(report)
    considered = kept = removed = 0
    removals_by_reason = {}
    for group in report:
        if not isinstance(group, dict):
            continue
        records = group.get("records") or []
        considered += len(records)
        n_removed_here = 0
        for rec in records:
            decision = (rec.get("decision") or "").strip().lower()
            if decision == "keep":
                kept += 1
            elif decision == "remove":
                removed += 1
                n_removed_here += 1
        if n_removed_here:
            note = (group.get("notes") or "uncategorized").strip() or "uncategorized"
            removals_by_reason[note] = removals_by_reason.get(note, 0) + n_removed_here

    top_reason = None
    if removals_by_reason and removed:
        note, count = max(removals_by_reason.items(), key=lambda kv: kv[1])
        if count / removed >= 0.5:
            top_reason = (note, count)

    return {
        "groups": groups,
        "considered": considered,
        "kept": kept,
        "removed": removed,
        "top_reason": top_reason,
    }


def _cleaning_bullet(summary):
    if not summary or not summary["considered"]:
        return None
    parts = [
        f"{summary['considered']} records reviewed across "
        f"{summary['groups']} duplicate-detection groups; "
        f"{summary['removed']} removed"
    ]
    if summary["top_reason"]:
        note, count = summary["top_reason"]
        for prefix in ("removed by ", "removed "):
            if note.lower().startswith(prefix):
                note = note[len(prefix):]
                break
        parts.append(f" (predominantly *{note}* — {count} of {summary['removed']})")
    parts.append(
        ". See [`cleaning_report.json`](cleaning_report.json) for per-record decisions."
    )
    return (
        "**Data cleaning ([PReSto data-cleaning app]"
        "(https://paleopresto.com)):** " + "".join(parts)
    )


def build_readme(query, configs, *, cleaning_report=None,
                 user_notes=None, pages_url=None, releases_url=None):
    mode = (query.get("mode") or "").strip().lower()
    mode_desc = MODE_DESCRIPTIONS.get(mode, mode or "—")

    compilations = _format_compilations(query.get("compilation"))
    archive_types = _format_archives(query.get("archiveTypes"))
    interp_var = query.get("interpVars") or query.get("variableName") or "temperature"

    tsids = query.get("tsids") or []
    removed_tsids = query.get("removedTsids") or []

    recon_range = _format_bp_range(configs.get("age_range_to_reconstruct")) or "—"
    ref_period = _format_bp_range(configs.get("reference_period")) or "—"
    time_res = _format_int(configs.get("time_resolution"), suffix="-yr") or "—"
    prior_window = _format_int(configs.get("prior_window"), suffix=" yr") or "—"
    loc_rad = _format_locrad(configs.get("localization_radius"))
    models = _format_models(configs.get("models_for_prior")) or "—"
    seasons = _format_seasons(configs.get("assimilate_selected_seasons")) or "—"
    pct_prior = _format_int(configs.get("percent_of_prior"), suffix="%") or "—"
    pct_assim = _format_int(configs.get("percent_to_assimilate"), suffix="%") or "—"
    proxy_ds = configs.get("proxy_datasets_to_assimilate") or "—"
    recon_type = configs.get("reconstruction_type") or "—"
    seed_prior = configs.get("seed_for_prior")
    seed_proxy = configs.get("seed_for_proxy_choice")

    lines = []
    lines.append("# Custom Holocene DA Reconstruction")
    lines.append("")
    lines.append(
        f"This repository was generated by [PReSto]({PRESTO_URL}) "
        "(Paleoclimate Reconstruction Storehouse) from the "
        f"[Holocene DA Template]({TEMPLATE_REPO}). It runs the offline "
        "paleoclimate data-assimilation method of "
        "[Erb et al. (2022)](https://doi.org/10.5194/cp-18-2599-2022) — "
        "implemented in the original [Holocene-Reconstruction/Holocene-code]"
        "(https://github.com/Holocene-Reconstruction/Holocene-code) "
        "repository — to reconstruct spatially complete temperature fields "
        "over the parameters and proxy selection captured below."
    )
    lines.append("")

    notes_text = (user_notes or "").strip()
    if notes_text:
        lines.append(notes_text)
        lines.append("")

    lines.append("## Reconstruction parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Reconstruction window | {recon_range} |")
    lines.append(f"| Time resolution | {time_res} |")
    lines.append(f"| Anomaly reference period | {ref_period} |")
    lines.append(f"| Prior models | {models} |")
    lines.append(f"| Prior window | {prior_window} |")
    lines.append(f"| Variables reconstructed | {configs.get('vars_to_reconstruct') or 'tas'} |")
    lines.append(f"| Reconstruction type | {recon_type} |")
    lines.append(f"| Seasons assimilated | {seasons} |")
    lines.append(f"| Prior sample fraction | {pct_prior} |")
    lines.append(f"| Proxy assimilation fraction | {pct_assim} |")
    lines.append(f"| Localization radius | {loc_rad} |")
    if seed_prior is not None or seed_proxy is not None:
        seeds_label = (
            f"prior={seed_prior if seed_prior is not None else '—'}, "
            f"proxy={seed_proxy if seed_proxy is not None else '—'}"
        )
        lines.append(f"| Random seeds | {seeds_label} |")
    lines.append("")
    lines.append("(See `config/user_config.yml` for the authoritative settings.)")
    lines.append("")

    lines.append("## Proxy data selection")
    lines.append("")
    lines.append(f"- **Mode:** {mode_desc}")
    lines.append(f"- **Proxy dataset:** {proxy_ds}")
    if compilations:
        lines.append(f"- **Source compilations:** {compilations}")
    if archive_types:
        lines.append(f"- **Archive types requested:** {archive_types}")
    if tsids:
        lines.append(f"- **Records selected:** {len(tsids)}")
    if removed_tsids:
        lines.append(f"- **Records explicitly excluded:** {len(removed_tsids)}")
    cleaning_summary = summarize_cleaning(cleaning_report) if cleaning_report else None
    cleaning_bullet = _cleaning_bullet(cleaning_summary)
    if cleaning_bullet:
        lines.append(f"- {cleaning_bullet}")
    if pages_url:
        validation_url = pages_url.rstrip("/") + "/validation/index.html"
        lines.append(
            "- **Validation:** see the "
            f"[validation page]({validation_url}) for GMST and 6 ka "
            "spatial comparisons against published Holocene reconstructions "
            "(Kaufman 2020 Temp12k, Erb et al. 2022, optional Marcott 2013)."
        )
    lines.append("")
    lines.append("(See `query_params.json` for the full TSID list.)")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "- Reconstruction NetCDFs are committed to `results/` after each "
        "successful run (files larger than 100 MB are uploaded as Release "
        "assets instead)."
    )
    if pages_url:
        lines.append(
            "- Validation page (vs published Holocene reconstructions) "
            "and the interactive visualization: "
            f"<{pages_url}>"
        )
    else:
        lines.append(
            "- A validation page (against published Holocene "
            "reconstructions) and the interactive visualization are "
            "deployed to GitHub Pages — see this repository's "
            "**Settings → Pages** for the deployed URL."
        )
    lines.append("")

    if releases_url:
        lines.append("## Citation & archive")
        lines.append("")
        lines.append(
            "Each successful reconstruction is bundled into a tagged "
            f"GitHub Release named `recon-<run_id>` at <{releases_url}>. "
            "The release preserves the recon NetCDFs, the proxy database "
            "that was assimilated (`lipd_legacy.pkl`), the validation "
            "page, and the input configs — these survive beyond the "
            "Actions artifact retention so the run remains fully "
            "auditable and reproducible long after the workflow logs "
            "expire."
        )
        lines.append("")
        lines.append(
            "If [GitHub–Zenodo integration]"
            "(https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content) "
            "is enabled on this repository, each release also receives a "
            "citable DOI, with a stable concept DOI for the "
            "reconstruction series as a whole."
        )
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "This reconstruction reproduces the offline paleoclimate "
        "data-assimilation method of [Erb et al. (2022)]"
        "(https://doi.org/10.5194/cp-18-2599-2022). The prior is "
        "constructed from transient climate model simulations (HadCM3 "
        "and TraCE-21ka), regridded and time-averaged to the user-"
        "specified resolution. Proxies are sourced from "
        "[LiPDverse](https://lipdverse.org). The original reconstruction "
        "code is available at [Holocene-Reconstruction/Holocene-code]"
        "(https://github.com/Holocene-Reconstruction/Holocene-code)."
    )
    lines.append("")

    lines.append("## Acknowledgements")
    lines.append("")
    lines.append(
        f"Built from the [PReSto Holocene DA Template]({TEMPLATE_REPO}) by "
        "[David Edge](https://orcid.org/0000-0001-6938-2850), "
        "[Michael Erb](https://orcid.org/0000-0002-1763-2522), "
        "[Nicholas McKay](https://orcid.org/0000-0003-3598-5113), "
        "[Deborah Khider](https://orcid.org/0000-0001-7501-8430), & "
        "[Julien Emile-Geay](https://orcid.org/0000-0001-5920-4751). "
        f"Hosted by [PReSto]({PRESTO_URL})."
    )
    lines.append("")

    lines.append("---")
    lines.append(
        "*This README is regenerated automatically by "
        "`generate_readme.py` from `query_params.json`, "
        "`config/user_config.yml`, and (if present) "
        "`cleaning_report.json`. Hand edits to this file will be "
        "overwritten on the next run — to add commentary that survives "
        "regenerations, write it in `README_NOTES.md` (created at repo "
        "root), where it will appear verbatim near the top of this page.*"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="query_params.json",
                        help="Path to query_params.json (default: %(default)s)")
    parser.add_argument("--configs", default="config/user_config.yml",
                        help="Path to user_config.yml (default: %(default)s)")
    parser.add_argument("--cleaning-report", default="cleaning_report.json",
                        help="Path to cleaning_report.json; silently "
                             "skipped if missing (default: %(default)s)")
    parser.add_argument("--notes", default="README_NOTES.md",
                        help="Path to user-authored notes file inserted "
                             "verbatim near the top of the README; "
                             "silently skipped if missing "
                             "(default: %(default)s)")
    parser.add_argument("--pages-url",
                        help="Public GitHub Pages URL for this repo, "
                             "linked from the Results section.")
    parser.add_argument("--releases-url",
                        help="GitHub Releases URL for this repo, linked "
                             "from the Citation & archive section. When "
                             "omitted the section is suppressed.")
    parser.add_argument("--out", default="README.md",
                        help="Output README path (default: %(default)s)")
    args = parser.parse_args()

    query_path = Path(args.query)
    configs_path = Path(args.configs)

    if not query_path.exists():
        print(f"ERROR: {query_path} not found", file=sys.stderr)
        return 1
    if not configs_path.exists():
        print(f"ERROR: {configs_path} not found", file=sys.stderr)
        return 1

    with query_path.open("r", encoding="utf-8") as f:
        query = json.load(f)
    with configs_path.open("r", encoding="utf-8") as f:
        configs = yaml.safe_load(f) or {}

    cleaning_report = None
    cleaning_path = Path(args.cleaning_report)
    if cleaning_path.exists():
        try:
            with cleaning_path.open("r", encoding="utf-8") as f:
                cleaning_report = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARN: could not parse {cleaning_path}: {exc}",
                  file=sys.stderr)

    user_notes = None
    notes_path = Path(args.notes)
    if notes_path.exists():
        try:
            user_notes = notes_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"WARN: could not read {notes_path}: {exc}", file=sys.stderr)

    text = build_readme(
        query, configs,
        cleaning_report=cleaning_report,
        user_notes=user_notes,
        pages_url=args.pages_url,
        releases_url=args.releases_url,
    )
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"Wrote {args.out} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
