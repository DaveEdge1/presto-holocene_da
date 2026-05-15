"""Strip non-requested TSIDs from a legacy LiPD dict pickle.

lipdGenerator's filtered pathway returns full parent LiPD records for any
dataset containing at least one requested TSID — so sibling paleoData
columns (alternate chronology flavors, sampleCount, EPS, etc.) survive
into the pickle. Without this pass the downstream Holocene DA
reconstruction assimilates every temperature TSID in those datasets,
not the user's selection.

Output stays a legacy LiPD dict (same shape as the input) because
da_load_proxies.py inside davidedge/lipd_webapps:holocene_da expects
that format. Pattern mirrors LMR2's scripts/lipd_to_pdb.py, but emits
a dict pickle instead of a flat cfr.ProxyDatabase DataFrame.
"""
import json
import pickle
import sys


# In nested LiPD dicts the column key is `TSid`; tolerate the other
# casings just in case a future lipdGenerator build changes it.
TSID_KEYS = ("TSid", "TSID", "tsid", "paleoData_TSid")


def get_tsid(col):
    for k in TSID_KEYS:
        if isinstance(col, dict) and k in col:
            return col[k]
    return None


def filter_table(table, wanted):
    cols = table.get("columns") if isinstance(table, dict) else None
    if not isinstance(cols, list):
        return table, 0
    kept = [c for c in cols if get_tsid(c) in wanted]
    table["columns"] = kept
    return table, len(kept)


def filter_paleo_entry(entry, wanted):
    if not isinstance(entry, dict):
        return entry, 0
    # measurementTable is a list in LiPD ≥1.3, dict in older variants.
    key = "measurementTable" if "measurementTable" in entry else "paleoMeasurementTable"
    tables = entry.get(key) or []
    if isinstance(tables, dict):
        tables = [tables]
    new_tables = []
    n_cols = 0
    for t in tables:
        t, n = filter_table(t, wanted)
        if n:
            new_tables.append(t)
            n_cols += n
    entry[key] = new_tables
    return entry, n_cols


def filter_dataset(ds, wanted):
    paleo = ds.get("paleoData") if isinstance(ds, dict) else None
    if not isinstance(paleo, list):
        return ds, 0
    new_paleo = []
    n_cols = 0
    for entry in paleo:
        entry, n = filter_paleo_entry(entry, wanted)
        if n:
            new_paleo.append(entry)
            n_cols += n
    ds["paleoData"] = new_paleo
    return ds, n_cols


def main(in_pkl, out_pkl, query_json):
    with open(query_json) as f:
        q = json.load(f)
    wanted = set(q.get("tsids") or [])

    with open(in_pkl, "rb") as f:
        data = pickle.load(f)

    # compare_to_temp12k_v102.py:86 handles both shapes; mirror that.
    wrapped = isinstance(data, dict) and "D" in data and isinstance(data["D"], dict)
    D = data["D"] if wrapped else data

    if not wanted:
        print("query_params.json has no tsids; pass-through (no filtering)")
        with open(out_pkl, "wb") as f:
            pickle.dump(data, f)
        return

    n_ds_in = len(D)
    new_D = {}
    n_cols_total = 0
    for name, ds in D.items():
        ds, n = filter_dataset(ds, wanted)
        if n:
            new_D[name] = ds
            n_cols_total += n

    if wrapped:
        data["D"] = new_D
        out = data
    else:
        out = new_D

    survivors = set()
    for ds in new_D.values():
        for entry in ds.get("paleoData", []):
            tables = entry.get("measurementTable") or entry.get("paleoMeasurementTable") or []
            if isinstance(tables, dict):
                tables = [tables]
            for t in tables:
                for c in t.get("columns", []):
                    tsid = get_tsid(c)
                    if tsid:
                        survivors.add(tsid)

    print(f"Datasets: {n_ds_in} -> {len(new_D)}")
    print(f"Columns kept: {n_cols_total}")
    print(f"TSIDs surviving filter: {len(survivors)} (requested {len(wanted)})")
    missing = wanted - survivors
    if missing:
        print(f"WARNING: {len(missing)} requested TSIDs not present in source pickle")

    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: filter_tsids.py IN_PKL OUT_PKL QUERY_JSON")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
