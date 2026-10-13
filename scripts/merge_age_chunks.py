"""Concatenate Holocene DA outputs run over contiguous age ranges.

    python3 merge_age_chunks.py <out_dir> chunk1.nc chunk2.nc ...

holocene_da.yml splits a long reconstruction into age chunks
(presto_age_chunks). Each age step of the DA is independent, so the chunks
together equal one run over the full range, provided they share the prior
ensemble (checked: same ens dimension) and the proxy list (same proxy
dimension and uncertainties). Variables with an `ages` dimension are
concatenated in age order; proxies_selected is OR-ed, since a chunk selects
only the proxies with values inside its range; the rest are copied from the
first chunk. Written one variable and one chunk at a time, so memory stays
at about one chunk's largest variable.
"""
import os
import sys

import netCDF4
import numpy as np


def main():
    out_dir, paths = sys.argv[1], sys.argv[2:]
    srcs = [netCDF4.Dataset(p) for p in paths]
    srcs.sort(key=lambda d: float(d['ages'][0]))
    first = srcs[0]
    for d in srcs[1:]:
        for dim in ('ens', 'ens_selected', 'proxy', 'lat', 'lon'):
            if d.dimensions[dim].size != first.dimensions[dim].size:
                sys.exit(f'chunks differ in {dim}: {first.dimensions[dim].size} vs {d.dimensions[dim].size}')
        if not np.array_equal(d['proxy_uncertainty'][:], first['proxy_uncertainty'][:], equal_nan=True):
            sys.exit('chunks differ in proxy_uncertainty: not the same proxy list')
    ages = np.concatenate([d['ages'][:] for d in srcs])
    if np.any(np.diff(ages) <= 0):
        sys.exit('chunk ages overlap or are out of order')

    name = os.path.basename(first.filepath())
    name = name[:-3] + '_merged.nc' if name.endswith('.nc') else name + '_merged.nc'
    out = netCDF4.Dataset(os.path.join(out_dir, name), 'w')
    for dim, v in first.dimensions.items():
        out.createDimension(dim, len(ages) if dim == 'ages' else v.size)
    for var in first.variables.values():
        o = out.createVariable(var.name, var.dtype, var.dimensions)
        if 'ages' in var.dimensions:
            ax = var.dimensions.index('ages')
            start = 0
            for d in srcs:
                n = d.dimensions['ages'].size
                sl = [slice(None)] * len(var.dimensions)
                sl[ax] = slice(start, start + n)
                o[tuple(sl)] = d[var.name][:]
                start += n
        elif var.name == 'proxies_selected':
            o[:] = np.any([d[var.name][:] for d in srcs], axis=0).astype(var.dtype)
        else:
            o[:] = var[:]
    out.title = first.title
    out.merged_from = ', '.join(os.path.basename(d.filepath()) for d in srcs)
    out.close()
    print(f'merged {len(srcs)} chunks, {len(ages)} ages -> {name}')


if __name__ == '__main__':
    main()
