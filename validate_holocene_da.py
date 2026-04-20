"""
Validation for Holocene DA reconstruction results.

Compares reconstruction GMST (in years BP) against published Holocene
reference reconstructions (Kaufman 2020 Temp12k, optional Marcott 2013)
and produces an HTML report modeled after the LMR `validate_recon.py`.

Reference files are discovered from $REFERENCE_DIR — any CSV with columns
(age_BP, median, q05, q95) or (age_BP, anomaly, uncertainty_1sigma) is
loaded automatically. Missing files are skipped with a warning.

Run inside davidedge/lipd_webapps:holocene_da.
"""

import csv
import glob
import json
import os

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil


RECON_DIR     = os.environ.get('RECON_DIR', '/recons')
VALIDATION_DIR = os.environ.get('VALIDATION_DIR', '/validation')
REFERENCE_DIR = os.environ.get('REFERENCE_DIR', '/reference_data')

os.makedirs(VALIDATION_DIR, exist_ok=True)


def pearson_r(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float('nan')
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def coefficient_of_efficiency(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 5:
        return float('nan')
    o, p = obs[mask], pred[mask]
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def align_series(time_a, val_a, time_b, val_b, ymin=None, ymax=None):
    """Align two series onto the coarser series' grid over their overlap in BP.

    For Holocene data, sampling grids usually don't share exact integer ages
    (e.g. reconstruction at 49, 149, ... vs reference at 0, 100, ...), so we
    interpolate the finer series onto the coarser series' ages within the
    shared range. Returns (common_ages, values_from_a, values_from_b).
    """
    ages_a = np.asarray(time_a, dtype=float)
    ages_b = np.asarray(time_b, dtype=float)
    val_a = np.asarray(val_a, dtype=float)
    val_b = np.asarray(val_b, dtype=float)

    # Drop NaNs and sort each
    mask_a = np.isfinite(ages_a) & np.isfinite(val_a)
    mask_b = np.isfinite(ages_b) & np.isfinite(val_b)
    ages_a, val_a = ages_a[mask_a], val_a[mask_a]
    ages_b, val_b = ages_b[mask_b], val_b[mask_b]
    if len(ages_a) == 0 or len(ages_b) == 0:
        return np.array([]), np.array([]), np.array([])
    oa = np.argsort(ages_a); ages_a, val_a = ages_a[oa], val_a[oa]
    ob = np.argsort(ages_b); ages_b, val_b = ages_b[ob], val_b[ob]

    lo = max(ages_a.min(), ages_b.min())
    hi = min(ages_a.max(), ages_b.max())
    if ymin is not None:
        lo = max(lo, ymin)
    if ymax is not None:
        hi = min(hi, ymax)
    if hi <= lo:
        return np.array([]), np.array([]), np.array([])

    # Pick the coarser-sampled series as the reference grid to avoid manufacturing information.
    dt_a = np.median(np.diff(ages_a)) if len(ages_a) > 1 else np.inf
    dt_b = np.median(np.diff(ages_b)) if len(ages_b) > 1 else np.inf
    if dt_a >= dt_b:
        common = ages_a[(ages_a >= lo) & (ages_a <= hi)]
        out_a = val_a[(ages_a >= lo) & (ages_a <= hi)]
        out_b = np.interp(common, ages_b, val_b)
    else:
        common = ages_b[(ages_b >= lo) & (ages_b <= hi)]
        out_b = val_b[(ages_b >= lo) & (ages_b <= hi)]
        out_a = np.interp(common, ages_a, val_a)

    return common, out_a, out_b


def load_reference(csv_path):
    """Load a reference reconstruction CSV.

    Supported schemas (auto-detected from the header):
      - age_BP, median, q05, q95       (e.g. Kaufman 2020)
      - age_BP, anomaly, uncertainty_1sigma  (e.g. Marcott 2013)

    Returns dict with keys: name, ages, median, lower, upper (lower/upper
    may be None if the source only has 1-sigma uncertainty; in that case
    lower=median-sigma, upper=median+sigma).
    """
    name = os.path.splitext(os.path.basename(csv_path))[0]
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        rows = [row for row in reader if row and row[0].strip() and not row[0].lstrip().startswith('#')]

    if not rows:
        print(f'  WARNING: {csv_path} has no data rows, skipping')
        return None

    try:
        if 'median' in header and 'q05' in header and 'q95' in header:
            i_age = header.index('age_bp')
            i_med = header.index('median')
            i_q05 = header.index('q05')
            i_q95 = header.index('q95')
            ages = np.array([int(round(float(r[i_age]))) for r in rows])
            median = np.array([float(r[i_med]) for r in rows])
            lower = np.array([float(r[i_q05]) for r in rows])
            upper = np.array([float(r[i_q95]) for r in rows])
        elif 'anomaly' in header and 'uncertainty_1sigma' in header:
            i_age = header.index('age_bp')
            i_anom = header.index('anomaly')
            i_sig = header.index('uncertainty_1sigma')
            ages = np.array([int(round(float(r[i_age]))) for r in rows])
            median = np.array([float(r[i_anom]) for r in rows])
            sigma = np.array([float(r[i_sig]) for r in rows])
            lower = median - sigma
            upper = median + sigma
        else:
            print(f'  WARNING: {csv_path} header {header} does not match a known schema, skipping')
            return None
    except (ValueError, IndexError) as e:
        print(f'  WARNING: failed to parse {csv_path}: {e}')
        return None

    return {'name': name, 'ages': ages, 'median': median,
            'lower': lower, 'upper': upper}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load reconstruction
# ═══════════════════════════════════════════════════════════════════════════
nc_candidates = sorted(glob.glob(os.path.join(RECON_DIR, '*.nc')))
if not nc_candidates:
    raise FileNotFoundError(f'No .nc file found in {RECON_DIR}')

nc_path = nc_candidates[0]
print(f'Loading reconstruction from {nc_path} ...')
ds = xr.open_dataset(nc_path)

ages = ds['ages'].values.astype(int)
lat = ds['lat'].values
lon = ds['lon'].values

# GMST across ALL ensemble members: recon_tas_global_mean has shape (ages, ens)
gmst_all = ds['recon_tas_global_mean'].values
n_ens = gmst_all.shape[1]
gmst_median = np.nanmedian(gmst_all, axis=1)
gmst_q05 = np.nanquantile(gmst_all, 0.05, axis=1)
gmst_q95 = np.nanquantile(gmst_all, 0.95, axis=1)

# Spatial field for 6ka map
tas_mean = ds['recon_tas_mean'].values  # (ages, lat, lon)

print(f'  Reconstruction: {len(ages)} age steps ({ages.min()}–{ages.max()} BP), '
      f'{n_ens} ensemble members')

# ═══════════════════════════════════════════════════════════════════════════
# 2. Load references from REFERENCE_DIR
# ═══════════════════════════════════════════════════════════════════════════
print(f'\nScanning references in {REFERENCE_DIR} ...')
references = []
for path in sorted(glob.glob(os.path.join(REFERENCE_DIR, '*.csv'))):
    ref = load_reference(path)
    if ref is not None:
        references.append(ref)
        print(f'  Loaded {ref["name"]}: {len(ref["ages"])} points '
              f'({ref["ages"].min()}–{ref["ages"].max()} BP)')

if not references:
    print('  No reference reconstructions found — metrics will be empty.')

# ═══════════════════════════════════════════════════════════════════════════
# 3. Compute metrics
# ═══════════════════════════════════════════════════════════════════════════
print('\nComputing GMST validation metrics ...')
gmst_results = {}
for ref in references:
    common, r_val, ref_val = align_series(ages, gmst_median, ref['ages'], ref['median'])
    if len(common) < 5:
        print(f'  {ref["name"]}: insufficient overlap, skipping')
        continue
    r = pearson_r(r_val, ref_val)
    ce = coefficient_of_efficiency(ref_val, r_val)
    overlap = f'{common.min()}–{common.max()} BP ({len(common)} pts)'
    print(f'  {ref["name"]}: R={r:.4f}, CE={ce:.4f}  [{overlap}]')
    gmst_results[ref['name']] = {'R': r, 'CE': ce, 'overlap': overlap,
                                  'n_points': int(len(common))}

# ═══════════════════════════════════════════════════════════════════════════
# 4. Plots
# ═══════════════════════════════════════════════════════════════════════════
plt.style.use('default')
PALETTE = ['#4682b4', '#ff8c00', '#2ca02c', '#d62728', '#9467bd']

# GMST time series
print('Generating GMST time series plot ...')
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(ages, gmst_q05, gmst_q95, alpha=0.25, color=PALETTE[0],
                label='Custom Holocene DA (5–95%)')
ax.plot(ages, gmst_median, color=PALETTE[0], lw=2, label='Custom Holocene DA (median)')
for i, ref in enumerate(references):
    color = PALETTE[(i + 1) % len(PALETTE)]
    ax.fill_between(ref['ages'], ref['lower'], ref['upper'], alpha=0.2, color=color,
                    label=f'{ref["name"]} (spread)')
    ax.plot(ref['ages'], ref['median'], color=color, lw=1.8, label=f'{ref["name"]} (median)')
ax.set_xlim(max(ages), min(ages))  # inverted: 12 ka left, 0 right
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlabel('Age (yr BP)')
ax.set_ylabel('Temperature anomaly (\u00b0C)')
ax.set_title('Global Mean Surface Temperature — Holocene')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(VALIDATION_DIR, 'gmst_timeseries.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# GMST ensemble members
print('Generating GMST ensemble members plot ...')
max_lines = 200
plot_idx = range(n_ens) if n_ens <= max_lines \
    else np.linspace(0, n_ens - 1, max_lines, dtype=int)

fig, ax = plt.subplots(figsize=(14, 6))
for i in plot_idx:
    ax.plot(ages, gmst_all[:, i], color=PALETTE[0],
            alpha=max(0.03, 3.0 / n_ens), lw=0.4)
ax.fill_between(ages, gmst_q05, gmst_q95, alpha=0.15, color='navy', label='5–95% range')
ax.plot(ages, gmst_median, color='navy', lw=2, label='Ensemble median')
for i, ref in enumerate(references):
    color = PALETTE[(i + 1) % len(PALETTE)]
    ax.plot(ref['ages'], ref['median'], color=color, lw=1.8,
            label=f'{ref["name"]} (median)')
ax.set_xlim(max(ages), min(ages))
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlabel('Age (yr BP)')
ax.set_ylabel('Temperature anomaly (\u00b0C)')
ax.set_title(f'GMST: All {n_ens} Ensemble Members')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(VALIDATION_DIR, 'gmst_ensemble_members.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Difference plot (recon - first reference, if any)
if references:
    primary_ref = references[0]
    print(f'Generating GMST difference plot (vs {primary_ref["name"]}) ...')
    common, recon_a, ref_a = align_series(ages, gmst_median,
                                          primary_ref['ages'], primary_ref['median'])
    if len(common) >= 5:
        diff = recon_a - ref_a
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(common, 0, diff, where=diff >= 0,
                        color='firebrick', alpha=0.5, interpolate=True)
        ax.fill_between(common, 0, diff, where=diff < 0,
                        color='steelblue', alpha=0.5, interpolate=True)
        ax.plot(common, diff, color='black', lw=0.5, alpha=0.7)
        ax.axhline(0, color='k', ls='--', alpha=0.5)
        ax.set_xlim(max(common), min(common))
        ax.set_xlabel('Age (yr BP)')
        ax.set_ylabel('Difference (\u00b0C)')
        ax.set_title(f'GMST Difference: Custom Holocene DA − {primary_ref["name"]}\n'
                     '(Red = warmer than reference, Blue = cooler)')
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(VALIDATION_DIR, 'gmst_difference.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

# 6ka spatial anomaly map (same style as make_basic_figures.py)
print('Generating 6ka spatial anomaly map ...')
ages_anom = [5500, 6500]
ages_ref = [0, 1000]
ind_anom = np.where((ages >= ages_anom[0]) & (ages <= ages_anom[1]))[0]
ind_ref = np.where((ages >= ages_ref[0]) & (ages <= ages_ref[1]))[0]

if len(ind_anom) > 0 and len(ind_ref) > 0:
    tas_change = np.nanmean(tas_mean[ind_anom, :, :], axis=0) \
        - np.nanmean(tas_mean[ind_ref, :, :], axis=0)

    # Area-weighted geo mean
    wgts = np.cos(np.deg2rad(lat))[:, np.newaxis]
    geo_mean = float(np.nansum(tas_change * wgts) / np.nansum(wgts * np.isfinite(tas_change)))

    tas_cyclic, lon_cyclic = cutil.add_cyclic_point(tas_change, coord=lon)
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    cf = ax.contourf(lon_cyclic, lat, tas_cyclic, np.arange(-1, 1.1, 0.1),
                     extend='both', cmap='bwr', transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.6)
    ax.gridlines(color='k', linewidth=0.5, linestyle=(0, (1, 5)))
    cb = plt.colorbar(cf, ax=ax, orientation='horizontal',
                      fraction=0.07, pad=0.04)
    cb.set_label('\u0394T (\u00b0C)', fontsize=12)
    ax.set_title(f'6 ka Anomaly: mean of {ages_anom[0]}\u2013{ages_anom[1]} BP '
                 f'minus {ages_ref[0]}\u2013{ages_ref[1]} BP\n'
                 f'Area-weighted global mean \u0394T = {geo_mean:.3f}\u00b0C',
                 fontsize=12)
    fig.savefig(os.path.join(VALIDATION_DIR, 'spatial_anomaly_6ka.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
else:
    print('  WARNING: reconstruction does not cover 6ka baseline windows, skipping map')
    geo_mean = float('nan')

# ═══════════════════════════════════════════════════════════════════════════
# 5. Save metrics CSV + JSON
# ═══════════════════════════════════════════════════════════════════════════
metrics_path = os.path.join(VALIDATION_DIR, 'validation_metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    w.writerow(['spatial_anomaly_6ka_geo_mean', f'{geo_mean:.4f}'])
    for ref_name, stats in gmst_results.items():
        w.writerow([f'{ref_name}_R', f'{stats["R"]:.4f}'])
        w.writerow([f'{ref_name}_CE', f'{stats["CE"]:.4f}'])
        w.writerow([f'{ref_name}_overlap', stats['overlap']])
        w.writerow([f'{ref_name}_n_points', stats['n_points']])
    w.writerow(['n_ensemble_members', int(n_ens)])
    w.writerow(['age_range_BP', f'{int(ages.min())}-{int(ages.max())}'])

json_metrics = {
    'spatial': {'anomaly_6ka_geo_mean': geo_mean,
                'anom_window_BP': ages_anom, 'ref_window_BP': ages_ref},
    'gmst': gmst_results,
    'config': {'n_ensemble_members': int(n_ens),
               'age_range_BP': [int(ages.min()), int(ages.max())]},
}
with open(os.path.join(VALIDATION_DIR, 'validation_metrics.json'), 'w') as f:
    json.dump(json_metrics, f, indent=2, default=str)

# ═══════════════════════════════════════════════════════════════════════════
# 6. HTML report
# ═══════════════════════════════════════════════════════════════════════════
print('Generating HTML report ...')

metric_cards = [
    f'''<div class="metric-card">
      <div class="value">{geo_mean:.3f}</div>
      <div class="label">6 ka \u0394T (\u00b0C, area-weighted)</div>
    </div>''']
for ref_name, stats in gmst_results.items():
    metric_cards.append(f'''<div class="metric-card">
      <div class="value">{stats["R"]:.3f}</div>
      <div class="label">GMST R vs {ref_name}</div>
    </div>''')
    metric_cards.append(f'''<div class="metric-card">
      <div class="value">{stats["CE"]:.3f}</div>
      <div class="label">GMST CE vs {ref_name}</div>
    </div>''')

table_rows = ''
for ref_name, stats in gmst_results.items():
    ce = stats['CE']
    ce_color = '#16a34a' if ce > 0.5 else ('#d97706' if ce > 0 else '#dc2626')
    table_rows += f'''    <tr>
      <td>{ref_name}</td>
      <td>{stats["overlap"]}</td>
      <td>{stats["R"]:.4f}</td>
      <td style="color: {ce_color}; font-weight: 600;">{ce:.4f}</td>
    </tr>\n'''

if not table_rows:
    table_rows = '    <tr><td colspan="4" style="text-align:center; color:#6b7280;">No reference reconstructions available. Drop a CSV into <code>reference_data/</code> to enable metrics.</td></tr>'

has_diff_plot = bool(references) and os.path.exists(
    os.path.join(VALIDATION_DIR, 'gmst_difference.png'))
primary_ref_name = references[0]['name'] if references else ''

html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Holocene DA Validation</title>
  <style>
    :root {{ --accent: #4682b4; --bg: #f7f8fa; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 1100px; margin: 0 auto; padding: 24px; color: #1a1a1a;
           background: var(--bg); }}
    h1 {{ border-bottom: 3px solid var(--accent); padding-bottom: 12px; font-size: 1.8rem; }}
    h2 {{ color: #374151; margin-top: 36px; font-size: 1.3rem;
          border-left: 4px solid var(--accent); padding-left: 12px; }}
    p {{ line-height: 1.6; color: #4b5563; }}
    table {{ border-collapse: collapse; margin: 16px 0; width: 100%;
             background: white; border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    th, td {{ border: 1px solid #e5e7eb; padding: 10px 16px; text-align: left; }}
    th {{ background: #f3f4f6; font-weight: 600; font-size: 0.9rem;
          text-transform: uppercase; letter-spacing: 0.03em; color: #6b7280; }}
    img {{ max-width: 100%; margin: 12px 0; border: 1px solid #e5e7eb;
           border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px; margin: 16px 0; }}
    .metric-card {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
    .metric-card .value {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
    .metric-card .label {{ font-size: 0.85rem; color: #6b7280; margin-top: 4px; }}
    .back {{ margin-top: 32px; }}
    code {{ background: #eef1f4; padding: 1px 6px; border-radius: 3px;
            font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Holocene DA Validation Report</h1>
  <p>Validation of the custom Holocene Data Assimilation reconstruction against
     published Holocene global temperature reconstructions. Reference datasets
     are discovered at runtime from <code>reference_data/</code> — any CSV
     with columns <code>age_BP, median, q05, q95</code> or
     <code>age_BP, anomaly, uncertainty_1sigma</code> will be used.</p>

  <div class="metric-grid">
{chr(10).join('    ' + c for c in metric_cards)}
  </div>

  <h2>GMST Validation Metrics</h2>
  <p>Pearson correlation (R) and Nash–Sutcliffe coefficient of efficiency (CE) of
     the ensemble-median GMST against each reference over their common age
     range. CE = 1 is perfect; CE = 0 equals climatology; CE &lt; 0 is worse
     than climatology.</p>
  <table>
    <tr><th>Reference</th><th>Overlap</th><th>R</th><th>CE</th></tr>
{table_rows}  </table>

  <h2>Spatial Anomaly at 6 ka</h2>
  <p>Mean temperature anomaly for {ages_anom[0]}–{ages_anom[1]} BP relative to
     {ages_ref[0]}–{ages_ref[1]} BP baseline. Robinson projection.</p>
  <img src="spatial_anomaly_6ka.png" alt="6 ka spatial anomaly map">

  <h2>GMST Time Series</h2>
  <p>Custom reconstruction ensemble spread alongside reference medians. X-axis
     runs from the oldest age (left) to present (right).</p>
  <img src="gmst_timeseries.png" alt="GMST time series">

  <h2>GMST Ensemble Members ({n_ens} total)</h2>
  <p>Every ensemble member plotted individually (subsampled to 200 for
     readability), showing the full reconstruction spread.</p>
  <img src="gmst_ensemble_members.png" alt="GMST ensemble members">

  {'<h2>GMST Difference (Custom − ' + primary_ref_name + ')</h2>' if has_diff_plot else ''}
  {'<p>Year-by-year difference between the custom reconstruction median and the primary reference. Red = warmer, Blue = cooler.</p>' if has_diff_plot else ''}
  {'<img src="gmst_difference.png" alt="GMST difference plot">' if has_diff_plot else ''}

  <p class="back"><a href="../index.html">&larr; Back to results</a></p>
</body>
</html>"""

with open(os.path.join(VALIDATION_DIR, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(html)

print(f'\nValidation complete. Outputs in {VALIDATION_DIR}/')
print('  Plots: spatial_anomaly_6ka.png, gmst_timeseries.png, gmst_ensemble_members.png'
      + (', gmst_difference.png' if has_diff_plot else ''))
print('  Data:  validation_metrics.csv, validation_metrics.json')
print('  HTML:  index.html')
