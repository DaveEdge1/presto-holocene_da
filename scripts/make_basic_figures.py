import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import xarray as xr
import os


# Canonical "Holocene thermal maximum minus modern" windows (yr BP).
CANONICAL_ANOM = [5500, 6500]
CANONICAL_REF  = [0, 1000]


def _indices_in(ages, lo, hi):
    return np.where((ages >= lo) & (ages <= hi))[0]


def select_anomaly_windows(ages):
    """Choose the (anomaly, reference) age windows for the spatial map.

    The default Holocene DA figure shows the 6-0.5 ka anomaly: the mean of
    5500-6500 BP minus 0-1000 BP. When a user reduces age_range_to_reconstruct
    so the reconstruction no longer spans those windows, the canonical index
    sets come back empty and `np.mean(tas[empty], axis=0)` yields an all-NaN
    field — contourf then draws nothing and a *blank* map saves with only a
    RuntimeWarning. (This is the silent failure mode that motivated the patch.)

    To keep the map meaningful we fall back to the oldest-available portion of
    the reconstructed range (anomaly) minus the youngest-available portion
    (reference), preserving the "old minus recent" sense. Returns
    (ages_anom, ages_ref, ind_anom, ind_ref, adaptive) where `adaptive` is True
    when the canonical windows were not covered.
    """
    ages = np.asarray(ages, dtype=float)
    ind_anom = _indices_in(ages, *CANONICAL_ANOM)
    ind_ref  = _indices_in(ages, *CANONICAL_REF)
    if len(ind_anom) > 0 and len(ind_ref) > 0:
        return CANONICAL_ANOM, CANONICAL_REF, ind_anom, ind_ref, False

    # Adaptive fallback derived from the actual reconstructed range.
    a_min, a_max = float(np.min(ages)), float(np.max(ages))
    span = a_max - a_min
    res = float(np.min(np.diff(np.sort(ages)))) if len(ages) > 1 else span
    # ~15% of the span per window, at least one full time bin, but never more
    # than 40% of the span so the old/recent windows stay disjoint.
    width = min(max(span * 0.15, res), span * 0.4) if span > 0 else 0.0
    ages_ref  = [a_min, a_min + width]   # most-recent portion
    ages_anom = [a_max - width, a_max]   # oldest portion
    ind_ref  = _indices_in(ages, *ages_ref)
    ind_anom = _indices_in(ages, *ages_anom)
    # Guarantee non-empty even for a single-bin / zero-span reconstruction.
    if len(ind_ref)  == 0: ind_ref  = np.array([int(np.argmin(ages))])
    if len(ind_anom) == 0: ind_anom = np.array([int(np.argmax(ages))])
    print('WARNING: reconstructed ages {:.0f}-{:.0f} BP do not cover the '
          'canonical 6-0.5 ka anomaly windows ({}-{} BP minus {}-{} BP); '
          'using adaptive windows {:.0f}-{:.0f} BP minus {:.0f}-{:.0f} BP '
          'instead.'.format(a_min, a_max, CANONICAL_ANOM[0], CANONICAL_ANOM[1],
                            CANONICAL_REF[0], CANONICAL_REF[1],
                            ages_anom[0], ages_anom[1], ages_ref[0], ages_ref[1]))
    return ages_anom, ages_ref, ind_anom, ind_ref, True


def _window_label(ages_anom, ages_ref, adaptive):
    if not adaptive:
        return '6-0.5 ka'
    # Express each window compactly in ka.
    return ('{:.2f}-{:.2f} ka minus {:.2f}-{:.2f} ka'
            .format(ages_anom[0] / 1000.0, ages_anom[1] / 1000.0,
                    ages_ref[0] / 1000.0, ages_ref[1] / 1000.0))


def make_figs(results_dir=None):

    #%% LOAD DATA

    dir1 = results_dir if results_dir else "/results/"

    for file in os.listdir(dir1):
        if file.endswith(".nc"):
            data_filename = os.path.join(dir1, file)

    output_dir = dir1

    # Load the Holocene Reconstruction
    data_xarray = xr.open_dataset(data_filename)
    tas_mean = data_xarray['recon_tas_mean'].values
    tas_ens  = data_xarray['recon_tas_ens'].values
    ages     = data_xarray['ages'].values
    lat      = data_xarray['lat'].values
    lon      = data_xarray['lon'].values
    exp_name = 'Holocene_DA'


    #%% CALCULATIONS

    # Compute the spatial anomaly. Prefer the canonical 6-0.5 ka windows, but
    # fall back to adaptive windows when the reconstructed range doesn't cover
    # them, so the map never silently renders blank.
    ages_anom, ages_ref, ind_anom, ind_ref, adaptive = select_anomaly_windows(ages)
    tas_mean_change = np.mean(tas_mean[ind_anom,:,:],axis=0) - np.mean(tas_mean[ind_ref,:,:], axis=0)
    window_label = _window_label(ages_anom, ages_ref, adaptive)

    # Compute global means
    lat_weights = np.cos(np.radians(lat))
    tas_mean_zonal = np.mean(tas_mean,axis=2)
    tas_ens_zonal  = np.mean(tas_ens, axis=3)
    tas_mean_global = np.average(tas_mean_zonal,axis=1,weights=lat_weights)
    tas_ens_global  = np.average(tas_ens_zonal, axis=2,weights=lat_weights)


    #%% FIGURES

    plt.style.use('ggplot')

    # Make a map
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.Robinson()); ax1.set_global()
    tas_change_cyclic,lon_cyclic = cutil.add_cyclic_point(tas_mean_change,coord=lon)
    map1 = ax1.contourf(lon_cyclic,lat,tas_change_cyclic,np.arange(-1,1.1,.1),extend='both',cmap='bwr',transform=ccrs.PlateCarree())
    colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    colorbar1.set_label('$\Delta$T ($^\circ$C)',fontsize=16)
    colorbar1.ax.set_facecolor('none')
    ax1.set_title('Mean $\Delta$T ($^\circ$C) at '+window_label+' for the Holocene Reconstruction\nexp_name: '+exp_name,loc='center',fontsize=16)
    ax1.coastlines()
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    plt.savefig(output_dir+'reconstruction_map_6ka_'+exp_name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.close()

    # Make a time series
    f,ax1 = plt.subplots(1,1,figsize=(12,6))
    ax1.plot(ages,tas_mean_global,linewidth=3)
    ax1.fill_between(ages,np.percentile(tas_ens_global,2.5,axis=1),np.percentile(tas_ens_global,97.5,axis=1),alpha=0.2)
    ax1.set_xlim(max(ages),min(ages))
    ax1.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=16)
    ax1.set_xlabel('Age (yr BP)',fontsize=16)
    ax1.set_title('Global mean $\Delta$T ($^\circ$C) for for the Holocene Reconstruction\nexp_name: '+exp_name,fontsize=18,loc='center')
    plt.savefig(output_dir+'reconstruction_ts_gmt_'+exp_name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.close()

    return 1
