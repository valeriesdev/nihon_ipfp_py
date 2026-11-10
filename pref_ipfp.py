from get_prefdata import load_db, process_db
from get_munidata import load_url, process_url
from gpu_ipfp import gpu_ipfp
from kmeans import gpu_kmeans

import cupy as cp

def pref_ipfp(
        app_id, force_refresh, year,
        prefs,
        tabs_preflevel, tabs_munilevel, tabs_muni_order,
        clusters, kmeans_it, kmeans_redos, kmeans_tol,
        ipfp_it, ipfp_tol,
        tabs_munilevel_chop, tabs_munilevel_remrows,
        tabs_munilevel_remcols, tabs_munilevel_mergecols,
        tabs_munilevel_headers, tabs_munilevel_indices,
        tabs_preflevel_chop, tabs_preflevel_reps
):
    print("Starting data loading")
    prefdata = {}
    for i in tabs_preflevel:
        prefdata[i] = load_db(app_id, tabs_preflevel[i], year, prefs["YMNK"], f"cache/YMNK_{i}.json", force_refresh)
        prefdata[i] = process_db(prefdata[i], tabs_preflevel, tabs_preflevel_chop, tabs_preflevel_reps)

    muniurls = {}
    munidata = {}
    for i in tabs_munilevel:
        muniurls[i] = load_url(app_id, 0, year, tabs_munilevel[i], prefs["YMNK"], f"cache/YMNK_{i}.json", force_refresh)
        munidata[i] = process_url(muniurls[i], 
                                tabs_munilevel_chop[i], 
                                tabs_munilevel_remrows[i], 
                                tabs_munilevel_remcols[i], 
                                tabs_munilevel_mergecols[i], 
                                tabs_munilevel_headers[i],
                                tabs_munilevel_indices[i],
                                f"cache/YMNK_{i}.csv", force_refresh)

    # Columns to exclude
    exclude = {"temp_level", "秘匿処理", "秘匿先情報", "合算地域", "総数"}

    # Extract unique dimension labels
    labels = [None] * (len(munidata) + 2)
    indexs = [None] * (len(munidata) + 2)
    tables = [None] * len(munidata)

    labels[0] = sorted({k[0] for k in munidata[list(munidata.keys())[0]]})
    labels[1] = sorted({k[1] for k in munidata[list(munidata.keys())[0]]})
    indexs[0] = {v: i for i, v in enumerate(labels[0])}
    indexs[1] = {v: i for i, v in enumerate(labels[1])}
    for i, label in enumerate(munidata):
        labels[i + 2] = [a for a in next(iter(munidata[label].values())) if a not in exclude]
        indexs[i + 2] = {v: i for i, v in enumerate(labels[i + 2])}
        tables[i] = cp.full((len(labels[0]), len(labels[1]), len(labels[i + 2])),cp.nan)

    for i, key in enumerate(munidata.keys()):
        for (a,b), data in munidata[key].items():
            for idx in indexs[i+2]:
                value = data.get(idx, cp.nan)
                tables[i][indexs[0][a], indexs[1][b], indexs[i+2][idx]
                    ] = float(value) if value != "-" else 0.0
        tables[i] = tables[i] / tables[i].sum()

    prefdata[list(prefdata.keys())[0]] = {
        k[1:-2]: v for k, v in prefdata[list(prefdata.keys())[0]].items() }

    X_pref = cp.full(tuple([len(j) for i, j in enumerate(labels) if i != 1]), cp.nan)
    for (z), data in prefdata[list(prefdata.keys())[0]].items():
        X_pref[*[indexs[i][lbl] if i == 0 else indexs[i+1][lbl] for i, lbl in enumerate(z)]] = float(data) if data != "-" and data is not None else 0.0
    X_pref = (X_pref / X_pref.sum())

    def to_proportions(arr):
        totals = arr.sum(axis=2, keepdims=True)
        return arr / cp.where(totals == 0, 1, totals)

    X_locals = cp.concatenate([to_proportions(i) for i in tables], axis=2)
    X_locals = X_locals.transpose(1, 0, 2)
    X_locals = X_locals.reshape(X_locals.shape[0], -1)
    X_locals = cp.ascontiguousarray(X_locals, dtype=cp.float32)

    assert bool(cp.isfinite(X_locals).all())

    n, d = X_locals.shape
    if clusters == -1:
        intertias = [0]*10
        for i in range(1,11):
            intertias[i-1], trash1, trash2 = gpu_kmeans(n, d, i, kmeans_it, kmeans_redos, kmeans_tol, X_locals)
            print(i, "inertia is", intertias[i-1])
        clusters = int(input("select cluster"))
    best_inertia, best_labels, best_centroids = gpu_kmeans(n, d, clusters, kmeans_it, kmeans_redos, kmeans_tol, X_locals)
    print(clusters, " Best inertia:", best_inertia)
    print(clusters, " Cluster sizes:", cp.bincount(best_labels, minlength=clusters).get())

    for i, table in enumerate(tables):
        table = table.transpose(1, 0, 2).reshape(table.shape[1], -1)

    tables_cluster = [
        [[] for _ in range(clusters)]
        for _ in range(len(tables))
    ]
    for i, table in enumerate(tables):
        for j, zone in enumerate(labels[1]):
            tables_cluster[i][int(best_labels[j])].append(table[:, j, :])

    local_features = []
    local_margins = [cp.full((clusters, 2, i.shape[2]), cp.nan) for i in tables]
    for i in range(0,clusters):
        local_features.append([])
        for j, table in enumerate(tables):
            local_features[i].append(cp.array(tables_cluster[j][i]))#.reshape(-1, 2, table.shape[1]//2).transpose(1, 0, 2))
            local_margins[j][i,:,:] = local_features[i][j].sum(axis=0)

    aggregates = [*[local_margin for local_margin in local_margins], X_pref]
    dimensions = tabs_muni_order

    N = len(local_margins[0])
    epsilon = 0 # 1e-8
    X = cp.broadcast_to(
        X_pref[None, :, :, :] + epsilon,
        (N, *(X_pref.shape))
    ).copy()

    clustered_table, converged, history = gpu_ipfp(
        original=X,
        aggregates=aggregates,
        dimensions=dimensions,
        max_iter=ipfp_it,
        tol=ipfp_tol,
        check_every=500,
        dtype=cp.float64,
        return_history=True,
        verbose=False
    )

    target = local_margins[0].sum(axis=2)  # (cluster, sex)
    current = clustered_table.sum(axis=(2, 3))  # (cluster, sex)
    scale = target / cp.where(current == 0, 1, current)
    clustered_table *= scale[:, :, None, None]

    results = []
    for i in range(0,clusters):
        print(f"\nippf on cluster {i}")
        features = local_features[i]
        joint = clustered_table[i].copy()

        # Construct support mask
        support = features[0] > 0

        for f in features[1:]:
            support = support[..., None] & (
                (f > 0).reshape(
                    *f.shape[:2],
                    *([1] * (support.ndim - 2)),
                    f.shape[-1]
                )
            )

        # Remove municipality dimension, preserving sex
        support = support.any(axis=0)

        # Remove impossible combinations
        joint = cp.where(support, joint, 0)

        # Rescale separately for each sex
        target = features[0].sum(axis=(0, 2))
        current = joint.sum(axis=tuple(range(1, joint.ndim)))

        scale = target / cp.where(current == 0, 1, current)
        joint *= scale.reshape(-1, *([1] * (joint.ndim - 1)))

        # Prepare IPFP constraints
        aggregates = [*features, joint]

        dimensions = tabs_muni_order

        D = clustered_table[i].shape
        N = local_features[i][0].shape[0]
        epsilon = 1e-12
        X = cp.broadcast_to(
            clustered_table[i][None, ::],
            (N, *D)
        ).copy()
        X = cp.maximum(X, epsilon)
        results.append([])
        results[i], converged, history = gpu_ipfp(
            original=X,
            aggregates=aggregates,
            dimensions=dimensions,
            max_iter=ipfp_it,
            tol=ipfp_tol,
            check_every=1000,
            dtype=cp.float64,
            return_history=True,
            verbose=True
        )
    
    return results, labels[1], best_labels, local_features