"""Proxy database comparison: custom Holocene DA reconstruction vs the
Temp12k v1.0.2 reference (the version used by the published Erb et al.
2022 reconstruction at https://lipdverse.org/Temp12k/1_0_2/).

Produces every artifact the validation HTML needs — `comparison.json`,
a stacked-archive temporal-coverage PNG, a Robinson spatial map, and
CSVs under `downloads/` (full record lists for shared / only-custom /
only-reference, and a single combined records.csv).

Runs inside davidedge/lipd_webapps:holocene_da so it can use the `lipd`
library (the legacy-pickle access pattern matches `da_load_proxies.py`).

Usage:
  python compare_to_temp12k_v102.py \
      --custom-pickle  /custom/lipd_legacy.pkl \
      --reference-url  https://lipdverse.org/Temp12k/1_0_2/Temp12k1_0_2.pkl \
      --out-dir        /validation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import urllib.request
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import lipd


REFERENCE_URL = 'https://lipdverse.org/Temp12k/1_0_2/Temp12k1_0_2.pkl'
REFERENCE_LABEL = 'Temp12k 1.0.2 (Erb et al. 2022)'

# Standard paleoclimate archive-type colors
ARCHIVE_COLORS = {
    'Wood': '#228B22', 'Tree': '#228B22',
    'Coral': '#FF6347', 'Sclerosponge': '#20B2AA',
    'GlacierIce': '#4169E1', 'Ice': '#4169E1',
    'LakeSediment': '#8B4513', 'Lake': '#8B4513',
    'MarineSediment': '#006400', 'Marine': '#006400',
    'Speleothem': '#9370DB',
    'Borehole': '#FF8C00',
    'Documents': '#808080',
    'MolluskShell': '#DEB887', 'Bivalve': '#DEB887',
    'Peat': '#9ACD32',
    'TerrestrialSediment': '#A0522D',
    'FluvialSediment': '#5F9EA0',
    'GroundIce': '#87CEEB',
    'Midden': '#CD853F',
    'Shoreline': '#F4A460',
    'Hybrid': '#C0C0C0',
    'Other': '#999999',
}


# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════

def fetch_reference(url, dst_path):
    """Download the reference legacy pickle if not cached."""
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1_000_000:
        print(f'Using cached reference pickle at {dst_path}')
        return dst_path
    print(f'Downloading reference from {url} ...')
    urllib.request.urlretrieve(url, dst_path)
    print(f'  saved {os.path.getsize(dst_path)/1e6:.1f} MB to {dst_path}')
    return dst_path


def load_records(pickle_path, label):
    """Load a Temp12k legacy pickle, apply the same Temp12k+degC filtering
    that da_load_proxies.py uses, and return a dict keyed by TSid."""
    print(f'Loading {label} from {pickle_path} ...')
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    D = data['D'] if isinstance(data, dict) and 'D' in data else data

    ts = lipd.extractTs(D)
    print(f'  {label}: extracted {len(ts)} timeseries')
    # Match da_load_proxies.py filtering exactly
    ts = lipd.filterTs(ts, 'paleoData_inCompilation == Temp12k')
    ts = lipd.filterTs(ts, 'paleoData_units == degC')
    print(f'  {label}: {len(ts)} after Temp12k+degC filter')

    out = {}
    skipped = 0
    for rec in ts:
        tsid = rec.get('paleoData_TSid')
        if not tsid:
            skipped += 1; continue
        if tsid in out:
            continue  # dedupe: Temp12k records sometimes appear twice
        try:
            ages = np.asarray(rec.get('age') or [], dtype=float)
            ages = ages[np.isfinite(ages)]
        except (TypeError, ValueError):
            ages = np.array([])
        try:
            lat = float(rec.get('geo_meanLat'))
        except (TypeError, ValueError):
            lat = None
        try:
            lon = float(rec.get('geo_meanLon'))
        except (TypeError, ValueError):
            lon = None
        out[tsid] = {
            'tsid': tsid,
            'dataSetName': rec.get('dataSetName', '') or '',
            'archive': str(rec.get('archiveType') or 'Other'),
            'variableName': str(rec.get('paleoData_variableName') or ''),
            'lat': lat,
            'lon': lon,
            'time_start_BP': float(ages.min()) if len(ages) else None,
            'time_end_BP': float(ages.max()) if len(ages) else None,
            'n_obs': int(len(ages)),
        }
    if skipped:
        print(f'  {label}: skipped {skipped} records with no TSid')
    print(f'  {label}: {len(out)} unique-TSid records retained')
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Aggregations
# ═══════════════════════════════════════════════════════════════════════════

def archive_breakdown(custom, reference, shared, only_custom, only_ref):
    """Build per-archive shared / only-custom / only-ref counts."""
    arcs = sorted({r['archive'] for r in custom.values()} |
                  {r['archive'] for r in reference.values()})
    rows = []
    for arc in arcs:
        s = sum(1 for t in shared if custom[t]['archive'] == arc)
        oc = sum(1 for t in only_custom if custom[t]['archive'] == arc)
        orf = sum(1 for t in only_ref if reference[t]['archive'] == arc)
        if s + oc + orf == 0:
            continue
        rows.append({'archive': arc, 'shared': s, 'only_custom': oc,
                     'only_reference': orf})
    rows.sort(key=lambda r: r['shared'] + r['only_custom'] + r['only_reference'],
              reverse=True)
    return rows


def side_by_side_stats(custom, reference):
    def stats_for(records):
        if not records:
            return {'records': 0}
        starts = [r['time_start_BP'] for r in records.values()
                  if r['time_start_BP'] is not None]
        ends = [r['time_end_BP'] for r in records.values()
                if r['time_end_BP'] is not None]
        n_obs = [r['n_obs'] for r in records.values() if r['n_obs']]
        archs = {r['archive'] for r in records.values()}
        return {
            'records': len(records),
            'distinct_archives': len(archs),
            'earliest_start_BP': float(max(ends)) if ends else None,
            'latest_end_BP': float(min(starts)) if starts else None,
            'median_record_length_yr': (float(np.median(
                [r['time_end_BP'] - r['time_start_BP']
                 for r in records.values()
                 if r['time_start_BP'] is not None
                 and r['time_end_BP'] is not None])) if records else None),
            'median_n_obs': float(np.median(n_obs)) if n_obs else None,
        }
    return {'custom': stats_for(custom), 'reference': stats_for(reference)}


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_temporal_coverage(custom, reference, shared, only_custom, only_ref,
                            out_path):
    """Stacked bar chart of record-count vs age, partitioned by source."""
    def hist(records, edges):
        counts = np.zeros(len(edges) - 1, dtype=int)
        for r in records:
            ts, te = r.get('time_start_BP'), r.get('time_end_BP')
            if ts is None or te is None:
                continue
            for i in range(len(edges) - 1):
                if ts <= edges[i + 1] and te >= edges[i]:
                    counts[i] += 1
        return counts

    edges = np.linspace(0, 12000, 49)  # 250-yr bins from 0–12 ka BP
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    shared_recs = [custom[t] for t in shared]
    only_c_recs = [custom[t] for t in only_custom]
    only_r_recs = [reference[t] for t in only_ref]

    h_shared = hist(shared_recs, edges)
    h_only_c = hist(only_c_recs, edges)
    h_only_r = hist(only_r_recs, edges)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(centers, h_shared, width=width * 0.95, label='Shared',
           color='#8C8C8C')
    ax.bar(centers, h_only_c, width=width * 0.95, bottom=h_shared,
           label='Custom-only', color='#1f77b4')
    ax.bar(centers, h_only_r, width=width * 0.95,
           bottom=h_shared + h_only_c, label=f'{REFERENCE_LABEL}-only',
           color='#d62728')
    ax.set_xlabel('Age (yr BP)')
    ax.set_ylabel('Records covering bin')
    ax.set_title(f'Temporal coverage: custom vs {REFERENCE_LABEL}')
    ax.set_xlim(12000, 0)  # present-day on right, oldest on left
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_spatial_map(custom, reference, only_custom, only_ref, out_path):
    """Robinson projection: shared / only-custom / only-reference proxies."""
    fig = plt.figure(figsize=(14, 7))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.gridlines(color='k', linewidth=0.3, linestyle=(0, (1, 5)))

    def scatter(records, color, marker, label):
        lats = [r['lat'] for r in records if r['lat'] is not None]
        lons = [r['lon'] for r in records if r['lon'] is not None
                and r['lat'] is not None]
        if not lats:
            return
        # Filter mismatched
        lats, lons = zip(*[(r['lat'], r['lon']) for r in records
                           if r['lat'] is not None and r['lon'] is not None])
        ax.scatter(list(lons), list(lats), c=color, marker=marker, s=24,
                   edgecolors='black', linewidths=0.3, alpha=0.85,
                   label=f'{label} (n={len(lats)})',
                   transform=ccrs.PlateCarree(), zorder=3)

    shared_recs = [custom[t] for t in (set(custom.keys()) & set(reference.keys()))]
    scatter(shared_recs, '#8C8C8C', 'o', 'Shared')
    scatter([custom[t] for t in only_custom], '#1f77b4', '^', 'Custom-only')
    scatter([reference[t] for t in only_ref], '#d62728', 's',
            f'{REFERENCE_LABEL}-only')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.85)
    ax.set_title(f'Proxy locations: custom vs {REFERENCE_LABEL}', fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CSV writers
# ═══════════════════════════════════════════════════════════════════════════

CSV_FIELDS = ['tsid', 'dataSetName', 'archive', 'variableName',
              'lat', 'lon', 'time_start_BP', 'time_end_BP', 'n_obs']


def write_records_csv(path, records_dict, tsid_iter):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for tsid in tsid_iter:
            r = records_dict.get(tsid)
            if r:
                w.writerow({k: r.get(k) for k in CSV_FIELDS})


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--custom-pickle', required=True,
                   help='Path to lipd_legacy.pkl produced by the run')
    p.add_argument('--reference-cache', default='/tmp/Temp12k1_0_2.pkl',
                   help='Local cache path for the reference Temp12k 1_0_2 pickle')
    p.add_argument('--reference-url', default=REFERENCE_URL)
    p.add_argument('--out-dir', required=True,
                   help='Validation output directory (e.g. /validation)')
    args = p.parse_args()

    out_dir = args.out_dir
    downloads_dir = os.path.join(out_dir, 'downloads')
    os.makedirs(downloads_dir, exist_ok=True)

    ref_path = fetch_reference(args.reference_url, args.reference_cache)
    custom = load_records(args.custom_pickle, 'custom run')
    reference = load_records(ref_path, REFERENCE_LABEL)

    custom_tsids = set(custom.keys())
    ref_tsids = set(reference.keys())
    shared = custom_tsids & ref_tsids
    only_custom = custom_tsids - ref_tsids
    only_ref = ref_tsids - custom_tsids

    print()
    print(f'  shared:        {len(shared)}')
    print(f'  custom-only:   {len(only_custom)}')
    print(f'  reference-only:{len(only_ref)}')

    # Plots
    print('\nGenerating plots ...')
    plot_temporal_coverage(custom, reference, shared, only_custom, only_ref,
                           os.path.join(out_dir, 'temporal_coverage_compare.png'))
    plot_spatial_map(custom, reference, only_custom, only_ref,
                     os.path.join(out_dir, 'spatial_map_compare.png'))

    # CSVs
    print('Writing per-set CSVs ...')
    write_records_csv(os.path.join(downloads_dir, 'shared.csv'),
                      custom, sorted(shared))
    write_records_csv(os.path.join(downloads_dir, 'only_custom.csv'),
                      custom, sorted(only_custom))
    write_records_csv(os.path.join(downloads_dir, 'only_reference.csv'),
                      reference, sorted(only_ref))

    # comparison.json
    print('Writing comparison.json ...')
    PREVIEW_N = 30
    comparison = {
        'reference_label': REFERENCE_LABEL,
        'reference_url': args.reference_url,
        'counts': {
            'shared': len(shared),
            'only_custom': len(only_custom),
            'only_reference': len(only_ref),
            'custom_total': len(custom_tsids),
            'reference_total': len(ref_tsids),
        },
        'stats': side_by_side_stats(custom, reference),
        'archive_rows': archive_breakdown(custom, reference, shared,
                                           only_custom, only_ref),
        'only_custom_preview': [custom[t] for t in sorted(only_custom)[:PREVIEW_N]],
        'only_reference_preview': [reference[t] for t in sorted(only_ref)[:PREVIEW_N]],
        'artifacts': {
            'temporal_coverage': 'temporal_coverage_compare.png',
            'spatial_map': 'spatial_map_compare.png',
            'downloads': {
                'shared': 'downloads/shared.csv',
                'only_custom': 'downloads/only_custom.csv',
                'only_reference': 'downloads/only_reference.csv',
            },
        },
    }
    with open(os.path.join(out_dir, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, default=str)

    print('\nComparison complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
