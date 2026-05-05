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
COMPARISON_JSON = os.environ.get('COMPARISON_JSON',
                                  os.path.join(VALIDATION_DIR, 'comparison.json'))

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
    # Proxy datasets have their own handler below; don't treat as GMST references.
    if path.endswith('_proxies.csv'):
        continue
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

tas_change = None
geo_mean = float('nan')
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


# ═══════════════════════════════════════════════════════════════════════════
# 4b. Spatial reference comparison (*_6ka_anomaly.nc in REFERENCE_DIR)
# ═══════════════════════════════════════════════════════════════════════════
spatial_results = {}
if tas_change is not None:
    sp_paths = sorted(glob.glob(os.path.join(REFERENCE_DIR, '*_6ka_anomaly.nc')))
    if sp_paths:
        print('\nScanning spatial references in {} ...'.format(REFERENCE_DIR))
    for sp_path in sp_paths:
        sp_name = os.path.splitext(os.path.basename(sp_path))[0].replace('_6ka_anomaly', '')
        try:
            sp_ds = xr.open_dataset(sp_path)
        except Exception as e:
            print(f'  WARN: could not open {sp_path}: {e}')
            continue
        anom_var = list(sp_ds.data_vars)[0]
        ref_field = sp_ds[anom_var].values
        ref_lat = sp_ds['lat'].values
        ref_lon = sp_ds['lon'].values

        # Align longitude convention (recon may be 0-360, ref may be -180..180 or vice versa)
        recon_lon360 = np.mod(lon, 360)
        ref_lon360 = np.mod(ref_lon, 360)

        # Nearest-neighbor regrid of reference onto recon grid
        # (vectorized for speed)
        lat_idx = np.argmin(np.abs(ref_lat[:, None] - lat[None, :]), axis=0)
        lon_idx = np.argmin(np.abs(ref_lon360[:, None] - recon_lon360[None, :]), axis=0)
        ref_on_recon = ref_field[lat_idx[:, None], lon_idx[None, :]]

        # Area-weighted pattern R and RMSE
        wmat = np.cos(np.deg2rad(lat))[:, np.newaxis] * np.ones_like(tas_change)
        mask_sp = np.isfinite(tas_change) & np.isfinite(ref_on_recon)
        if mask_sp.sum() < 100:
            print(f'  {sp_name}: too few valid grid cells, skipping')
            sp_ds.close()
            continue
        ww = wmat[mask_sp]; ww /= ww.sum()
        aa = tas_change[mask_sp]; bb = ref_on_recon[mask_sp]
        a_m = float((aa * ww).sum()); b_m = float((bb * ww).sum())
        cov = float((ww * (aa - a_m) * (bb - b_m)).sum())
        va = float((ww * (aa - a_m) ** 2).sum()); vb = float((ww * (bb - b_m) ** 2).sum())
        sp_r = cov / np.sqrt(va * vb) if va > 0 and vb > 0 else float('nan')
        sp_rmse = float(np.sqrt((ww * (aa - bb) ** 2).sum()))
        spatial_results[sp_name] = {'R': float(sp_r), 'RMSE': sp_rmse,
                                    'recon_geo_mean': geo_mean,
                                    'ref_geo_mean': b_m}
        print(f'  {sp_name}: spatial R={sp_r:.4f}, RMSE={sp_rmse:.4f} \u00b0C')

        # 3-panel figure
        def _panel(ax, data, title, levels, cmap):
            d_cyc, lon_cyc = cutil.add_cyclic_point(data, coord=lon)
            cf = ax.contourf(lon_cyc, lat, d_cyc, levels, extend='both', cmap=cmap,
                             transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.5)
            ax.gridlines(color='k', linewidth=0.3, linestyle=(0, (1, 5)))
            ax.set_title(title, fontsize=10)
            return cf

        diff = tas_change - ref_on_recon
        lvls_main = np.arange(-1, 1.1, 0.1)
        d_abs = float(np.nanmax(np.abs(diff)))
        d_abs = max(d_abs, 0.5)
        lvls_diff = np.linspace(-d_abs, d_abs, 21)

        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(1, 3, 1, projection=ccrs.Robinson()); ax1.set_global()
        cf1 = _panel(ax1, tas_change, 'Custom Holocene DA\n6 ka anomaly',
                     lvls_main, 'bwr')
        ax2 = plt.subplot(1, 3, 2, projection=ccrs.Robinson()); ax2.set_global()
        _panel(ax2, ref_on_recon, f'{sp_name}\n6 ka anomaly', lvls_main, 'bwr')
        ax3 = plt.subplot(1, 3, 3, projection=ccrs.Robinson()); ax3.set_global()
        cf3 = _panel(ax3, diff, f'Custom \u2212 {sp_name}', lvls_diff, 'RdBu_r')

        cb1 = fig.colorbar(cf1, ax=[ax1, ax2], orientation='horizontal',
                           fraction=0.05, pad=0.05, aspect=40)
        cb1.set_label('\u0394T (\u00b0C)', fontsize=10)
        cb3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal',
                           fraction=0.05, pad=0.05, aspect=20)
        cb3.set_label('Difference (\u00b0C)', fontsize=10)
        fig.suptitle(f'6 ka Spatial Anomaly: Custom Holocene DA vs {sp_name}   '
                     f'(pattern R={sp_r:.3f}, RMSE={sp_rmse:.3f} \u00b0C)', fontsize=12)
        fig.savefig(os.path.join(VALIDATION_DIR, f'spatial_anomaly_6ka_vs_{sp_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        sp_ds.close()


# ═══════════════════════════════════════════════════════════════════════════
# 4c. Proxy-site comparison (*_proxies.csv in REFERENCE_DIR)
# ═══════════════════════════════════════════════════════════════════════════
proxy_results = {}
proxy_plots = {}
if tas_change is not None:
    pp_paths = sorted(glob.glob(os.path.join(REFERENCE_DIR, '*_proxies.csv')))
    if pp_paths:
        print('\nScanning proxy datasets in {} ...'.format(REFERENCE_DIR))
    for pp_path in pp_paths:
        pp_name = os.path.splitext(os.path.basename(pp_path))[0]
        with open(pp_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print(f'  {pp_name}: empty, skipping')
            continue
        lats_p, lons_p, vals_p, archives_p = [], [], [], []
        for row in rows:
            try:
                la = float(row.get('lat', 'nan'))
                lo = float(row.get('lon', 'nan'))
                vv = float(row.get('value_6ka', 'nan'))
            except (ValueError, TypeError):
                continue
            if not (np.isfinite(la) and np.isfinite(lo) and np.isfinite(vv)):
                continue
            lats_p.append(la); lons_p.append(lo); vals_p.append(vv)
            archives_p.append((row.get('archive_type') or '').strip() or 'unknown')
        if len(vals_p) < 5:
            print(f'  {pp_name}: fewer than 5 usable proxies, skipping')
            continue
        lats_p = np.array(lats_p); lons_p = np.array(lons_p)
        vals_p = np.array(vals_p); archives_p = np.array(archives_p)

        # Sample recon 6 ka anomaly at each proxy (nearest neighbor)
        recon_lon_is_360 = float(np.nanmax(lon)) > 180
        plot_lons = lons_p.copy()
        match_lons = np.mod(lons_p, 360) if recon_lon_is_360 else lons_p
        recon_at_proxy = np.full(len(vals_p), np.nan)
        for k in range(len(vals_p)):
            ilat = int(np.argmin(np.abs(lat - lats_p[k])))
            ilon = int(np.argmin(np.abs(lon - match_lons[k])))
            recon_at_proxy[k] = tas_change[ilat, ilon]

        mask_p = np.isfinite(recon_at_proxy) & np.isfinite(vals_p)
        if mask_p.sum() < 5:
            print(f'  {pp_name}: no matched proxies, skipping')
            continue
        pr_r = pearson_r(recon_at_proxy, vals_p)
        pr_rmse = float(np.sqrt(np.mean((recon_at_proxy[mask_p] - vals_p[mask_p]) ** 2)))
        pr_bias = float(np.mean(recon_at_proxy[mask_p] - vals_p[mask_p]))
        proxy_results[pp_name] = {'N': int(mask_p.sum()), 'R': float(pr_r),
                                  'RMSE': pr_rmse, 'bias': pr_bias}
        print(f'  {pp_name}: N={mask_p.sum()}, R={pr_r:.3f}, '
              f'RMSE={pr_rmse:.3f}, bias={pr_bias:+.3f}')

        # Figure: map with proxy points + scatter
        fig = plt.figure(figsize=(18, 6))
        ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson()); ax1.set_global()
        tas_cyc_p, lon_cyc_p = cutil.add_cyclic_point(tas_change, coord=lon)
        cf = ax1.contourf(lon_cyc_p, lat, tas_cyc_p, np.arange(-1, 1.1, 0.1),
                          extend='both', cmap='bwr', alpha=0.55,
                          transform=ccrs.PlateCarree())
        ax1.coastlines(linewidth=0.5)
        ax1.scatter(plot_lons, lats_p, c=vals_p, vmin=-1, vmax=1, cmap='bwr',
                    s=28, edgecolors='black', linewidths=0.4,
                    transform=ccrs.PlateCarree())
        cb = fig.colorbar(cf, ax=ax1, orientation='horizontal', fraction=0.05,
                          pad=0.05, aspect=40)
        cb.set_label('6 ka \u0394T (\u00b0C)', fontsize=10)
        ax1.set_title(f'Recon 6 ka anomaly with {pp_name} sites overlaid '
                      f'(N={int(mask_p.sum())})', fontsize=10)

        ax2 = plt.subplot(1, 2, 2)
        ax2.scatter(vals_p[mask_p], recon_at_proxy[mask_p], s=20, alpha=0.6,
                    c='steelblue', edgecolors='black', linewidths=0.2)
        lim = float(max(np.nanmax(np.abs(vals_p)),
                        np.nanmax(np.abs(recon_at_proxy[mask_p]))))
        lim = max(lim, 1.0)
        ax2.plot([-lim, lim], [-lim, lim], '--', color='gray', alpha=0.7, label='1:1')
        ax2.axhline(0, color='k', lw=0.4, alpha=0.5)
        ax2.axvline(0, color='k', lw=0.4, alpha=0.5)
        ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
        ax2.set_aspect('equal')
        ax2.set_xlabel('Proxy 6 ka \u0394T (\u00b0C)')
        ax2.set_ylabel('Recon 6 ka \u0394T at proxy site (\u00b0C)')
        ax2.set_title(f'Proxy vs Recon   R={pr_r:.3f}, RMSE={pr_rmse:.3f}, '
                      f'bias={pr_bias:+.3f}')
        ax2.grid(True, alpha=0.3); ax2.legend(loc='best')
        fig.savefig(os.path.join(VALIDATION_DIR, f'proxy_comparison_{pp_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        proxy_plots[pp_name] = f'proxy_comparison_{pp_name}.png'

# ═══════════════════════════════════════════════════════════════════════════
# 5. Save metrics CSV + JSON
# ═══════════════════════════════════════════════════════════════════════════
metrics_path = os.path.join(VALIDATION_DIR, 'validation_metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    w.writerow(['spatial_anomaly_6ka_geo_mean', f'{geo_mean:.4f}'])
    for ref_name, stats in gmst_results.items():
        w.writerow([f'gmst_{ref_name}_R', f'{stats["R"]:.4f}'])
        w.writerow([f'gmst_{ref_name}_CE', f'{stats["CE"]:.4f}'])
        w.writerow([f'gmst_{ref_name}_overlap', stats['overlap']])
        w.writerow([f'gmst_{ref_name}_n_points', stats['n_points']])
    for sp_name, stats in spatial_results.items():
        w.writerow([f'spatial_{sp_name}_R', f'{stats["R"]:.4f}'])
        w.writerow([f'spatial_{sp_name}_RMSE', f'{stats["RMSE"]:.4f}'])
        w.writerow([f'spatial_{sp_name}_ref_geo_mean', f'{stats["ref_geo_mean"]:.4f}'])
    for pp_name, stats in proxy_results.items():
        w.writerow([f'proxy_{pp_name}_N', int(stats['N'])])
        w.writerow([f'proxy_{pp_name}_R', f'{stats["R"]:.4f}'])
        w.writerow([f'proxy_{pp_name}_RMSE', f'{stats["RMSE"]:.4f}'])
        w.writerow([f'proxy_{pp_name}_bias', f'{stats["bias"]:.4f}'])
    w.writerow(['n_ensemble_members', int(n_ens)])
    w.writerow(['age_range_BP', f'{int(ages.min())}-{int(ages.max())}'])

json_metrics = {
    'spatial': {'anomaly_6ka_geo_mean': geo_mean,
                'anom_window_BP': ages_anom, 'ref_window_BP': ages_ref,
                'comparisons': spatial_results},
    'gmst': gmst_results,
    'proxy': proxy_results,
    'config': {'n_ensemble_members': int(n_ens),
               'age_range_BP': [int(ages.min()), int(ages.max())]},
}
with open(os.path.join(VALIDATION_DIR, 'validation_metrics.json'), 'w') as f:
    json.dump(json_metrics, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# 5b. Load proxy-database comparison vs published Holocene DA (Temp12k 1.0.2)
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_int(v, dash='—'):
    if v is None or v == '':
        return dash
    try:
        return f'{int(round(float(v))):,}'
    except (TypeError, ValueError):
        return str(v)


def _fmt_float(v, fmt='{:.1f}', dash='—'):
    if v is None or v == '':
        return dash
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def _build_comparison_html(c):
    counts = c['counts']
    stats  = c['stats']
    ref_label = c.get('reference_label', 'Reference')
    arts  = c.get('artifacts', {}) or {}
    dls   = arts.get('downloads', {}) or {}

    cust_total = counts.get('custom_total', counts['shared'] + counts['only_custom'])
    ref_total  = counts.get('reference_total', counts['shared'] + counts['only_reference'])

    # Funnel-style summary chips
    chips = []
    for label, val in [
        (f'{ref_label} records',      ref_total),
        ('Custom records',            cust_total),
        ('Shared',                    counts['shared']),
        ('Custom-only',               counts['only_custom']),
        (f'{ref_label}-only',         counts['only_reference']),
    ]:
        chips.append(f'''
      <div class="metric-card">
        <div class="value">{_fmt_int(val)}</div>
        <div class="label">{label}</div>
      </div>''')

    # Side-by-side stats
    cs, rs = stats['custom'], stats['reference']
    stats_rows = []
    rows_def = [
        ('Records',                'records',                 _fmt_int),
        ('Distinct archive types', 'distinct_archives',       _fmt_int),
        ('Earliest record start (yr BP)',  'earliest_start_BP', _fmt_int),
        ('Latest record end (yr BP)',      'latest_end_BP',     _fmt_int),
        ('Median record length (yr)',      'median_record_length_yr',
         lambda v: _fmt_float(v, '{:.0f}')),
        ('Median observations per record', 'median_n_obs',
         lambda v: _fmt_float(v, '{:.0f}')),
    ]
    for label, key, fmt in rows_def:
        stats_rows.append(
            f'<tr><td>{label}</td>'
            f'<td>{fmt(cs.get(key))}</td>'
            f'<td>{fmt(rs.get(key))}</td></tr>')

    # Archive breakdown
    arch_rows = []
    totals = {'shared': 0, 'only_custom': 0, 'only_reference': 0}
    for r in c.get('archive_rows', []):
        totals['shared'] += r['shared']
        totals['only_custom'] += r['only_custom']
        totals['only_reference'] += r['only_reference']
        arch_rows.append(
            f'<tr><td>{r["archive"]}</td>'
            f'<td>{r["shared"]}</td>'
            f'<td>{r["only_custom"]}</td>'
            f'<td>{r["only_reference"]}</td>'
            f'<td>{r["shared"] + r["only_custom"] + r["only_reference"]}</td></tr>')
    arch_rows.append(
        f'<tr style="font-weight:bold"><td>Total</td>'
        f'<td>{totals["shared"]}</td>'
        f'<td>{totals["only_custom"]}</td>'
        f'<td>{totals["only_reference"]}</td>'
        f'<td>{sum(totals.values())}</td></tr>')

    def _preview_table(records, csv_link, total):
        if not records:
            return '<p><em>None.</em></p>'
        body = []
        for r in records:
            body.append(
                f'<tr><td><code>{r.get("tsid","")}</code></td>'
                f'<td>{r.get("archive","")}</td>'
                f'<td>{r.get("dataSetName","")}</td>'
                f'<td>{_fmt_int(r.get("time_start_BP"))}–{_fmt_int(r.get("time_end_BP"))}</td>'
                f'<td>{_fmt_int(r.get("n_obs"))}</td></tr>')
        table = ('<table><tr><th>TSID</th><th>Archive</th>'
                 '<th>Dataset</th><th>Age range (BP)</th><th>n_obs</th></tr>'
                 + ''.join(body) + '</table>')
        if csv_link and total and total > len(records):
            table += (f'<p><a href="{csv_link}" download>'
                      f'⬇ Download full CSV ({total} rows)</a></p>')
        return table

    preview_custom = _preview_table(c.get('only_custom_preview', []),
                                     dls.get('only_custom'),
                                     counts['only_custom'])
    preview_ref    = _preview_table(c.get('only_reference_preview', []),
                                     dls.get('only_reference'),
                                     counts['only_reference'])

    spatial_img = (f'<img src="{arts["spatial_map"]}" alt="Proxy spatial comparison">'
                   if arts.get('spatial_map') else '')
    temporal_img = (f'<img src="{arts["temporal_coverage"]}" alt="Temporal coverage comparison">'
                    if arts.get('temporal_coverage') else '')

    return f'''
  <details class="section">
    <summary style="font-size: 1.3rem; font-weight: 600; cursor: pointer;
                    color: #374151; padding: 8px 0;">
      Proxy Database Comparison vs {ref_label}
      <span style="font-weight: 400; color: #6b7280; font-size: 0.95rem;">
        (shared {counts["shared"]}, custom-only {counts["only_custom"]},
         reference-only {counts["only_reference"]})
      </span>
    </summary>

    <div style="padding-top: 16px;">
      <p>Comparison of the proxy records in this run’s
         <code>lipd_legacy.pkl</code> against
         <strong>{ref_label}</strong>, the version used by the published
         Erb et al. 2022 reconstruction. Records are matched on
         <code>paleoData_TSid</code> after the same
         <code>paleoData_inCompilation == Temp12k</code> +
         <code>paleoData_units == degC</code> filter that
         <code>da_load_proxies.py</code> applies at runtime.</p>

      <div class="metric-grid">{''.join(chips)}</div>

      <h3>Side-by-side statistics</h3>
      <table><tr><th>Statistic</th><th>Custom</th><th>{ref_label}</th></tr>
        {''.join(stats_rows)}
      </table>

      <h3>Records by archive type</h3>
      <table><tr><th>Archive</th><th>Shared</th><th>Custom-only</th>
        <th>{ref_label}-only</th><th>Total</th></tr>
        {''.join(arch_rows)}
      </table>

      <h3>Spatial distribution</h3>
      {spatial_img}

      <h3>Temporal coverage</h3>
      <p>Records covering each 250-yr age bin, partitioned by which
         database they belong to.</p>
      {temporal_img}

      <h3>Records exclusive to the custom run <small>({counts["only_custom"]})</small></h3>
      <p>Records present in this reconstruction’s proxy database but
         absent from {ref_label} — typically records added in later
         Temp12k versions or pulled from filtered queries.</p>
      {preview_custom}

      <h3>Records exclusive to {ref_label} <small>({counts["only_reference"]})</small></h3>
      <p>Records used by the published reconstruction but missing from this
         custom run — typically records dropped by the user’s
         filter or removed from later Temp12k versions.</p>
      {preview_ref}
    </div>
  </details>
'''


comparison_html = ''
if os.path.exists(COMPARISON_JSON):
    print(f'\nLoading proxy comparison from {COMPARISON_JSON} ...')
    try:
        with open(COMPARISON_JSON) as f:
            comparison_data = json.load(f)
        comparison_html = _build_comparison_html(comparison_data)
        print('  Comparison HTML section built.')
    except Exception as exc:
        print(f'  WARN: failed to render comparison.json: {exc}')


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
for sp_name, stats in spatial_results.items():
    metric_cards.append(f'''<div class="metric-card">
      <div class="value">{stats["R"]:.3f}</div>
      <div class="label">Spatial R vs {sp_name}</div>
    </div>''')
for pp_name, stats in proxy_results.items():
    metric_cards.append(f'''<div class="metric-card">
      <div class="value">{stats["R"]:.3f}</div>
      <div class="label">Proxy R ({pp_name}, N={stats["N"]})</div>
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

# Spatial comparison section (one block per reference spatial field)
spatial_section = ''
if spatial_results:
    sp_items = []
    sp_items.append('  <h2>Spatial Comparison at 6 ka</h2>')
    sp_items.append('  <p>Side-by-side 6 ka temperature anomaly for the custom '
                    'reconstruction and each spatial reference, plus their difference. '
                    'Pattern correlation (R) and area-weighted RMSE are computed after '
                    'nearest-neighbor regridding of the reference onto the reconstruction '
                    'grid.</p>')
    sp_items.append('  <table><tr><th>Reference</th><th>Pattern R</th><th>RMSE (°C)</th>'
                    '<th>Recon geo mean</th><th>Ref geo mean</th></tr>')
    for sp_name, stats in spatial_results.items():
        sp_items.append(f'    <tr><td>{sp_name}</td>'
                        f'<td>{stats["R"]:.4f}</td>'
                        f'<td>{stats["RMSE"]:.4f}</td>'
                        f'<td>{stats["recon_geo_mean"]:+.3f} °C</td>'
                        f'<td>{stats["ref_geo_mean"]:+.3f} °C</td></tr>')
    sp_items.append('  </table>')
    for sp_name in spatial_results:
        sp_items.append(f'  <img src="spatial_anomaly_6ka_vs_{sp_name}.png" '
                        f'alt="6 ka spatial comparison vs {sp_name}">')
    spatial_section = '\n'.join(sp_items)

# Proxy comparison section
proxy_section = ''
if proxy_results:
    pp_items = []
    pp_items.append('  <h2>Proxy-Site Comparison (6 ka)</h2>')
    pp_items.append('  <p>For each proxy site, the reconstruction\u2019s 6 ka anomaly is '
                    'sampled via nearest-neighbor on the model grid, then compared to the '
                    'proxy\u2019s own 6 ka anomaly (5500\u20136500 BP relative to 0\u20131000 BP '
                    'baseline, falling back to the record mean if the modern baseline is '
                    'absent). R, RMSE, and bias (recon \u2212 proxy) summarize the match.</p>')
    pp_items.append('  <table><tr><th>Dataset</th><th>N</th><th>R</th><th>RMSE (°C)</th>'
                    '<th>Bias (°C)</th></tr>')
    for pp_name, stats in proxy_results.items():
        pp_items.append(f'    <tr><td>{pp_name}</td>'
                        f'<td>{stats["N"]}</td>'
                        f'<td>{stats["R"]:.4f}</td>'
                        f'<td>{stats["RMSE"]:.4f}</td>'
                        f'<td>{stats["bias"]:+.4f}</td></tr>')
    pp_items.append('  </table>')
    for pp_name, plot_file in proxy_plots.items():
        pp_items.append(f'  <img src="{plot_file}" alt="Proxy comparison ({pp_name})">')
    proxy_section = '\n'.join(pp_items)

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

{spatial_section}

{proxy_section}

{comparison_html}

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
plot_list = ['spatial_anomaly_6ka.png', 'gmst_timeseries.png', 'gmst_ensemble_members.png']
if has_diff_plot:
    plot_list.append('gmst_difference.png')
for sp_name in spatial_results:
    plot_list.append(f'spatial_anomaly_6ka_vs_{sp_name}.png')
for pp_name in proxy_plots:
    plot_list.append(proxy_plots[pp_name])
print('  Plots: ' + ', '.join(plot_list))
print('  Data:  validation_metrics.csv, validation_metrics.json')
print('  HTML:  index.html')
