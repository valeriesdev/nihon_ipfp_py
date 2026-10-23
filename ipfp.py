from get_prefdata import load_db, process_db
from get_munidata import load_url, process_url
from tabs import *
from gpu_ipfp import gpu_ipfp
from kmeans import gpu_kmeans

import cupy as cp
import time

force_refresh = False

app_id = "8c1856560dde0c0a6cb33e7b1a67fb72bfa85ec0"
year = "2020"
prefs = {"YMNK" : "19"}
tabs_preflevel = {"男女_年齢_職業_産業":"0003450560"}
tabs_munilevel = {
    "男女_年齢_人口":"男女，年齢（5歳階級）別人口，平均年齢及び総年齢－町丁・字等", # age
    "男女_産業_職業者数":"男女，産業（大分類）別就業者数（15歳以上）－町丁・字等", # industry
    "男女_職業_職業者数":"男女，職業（大分類）別就業者数（15歳以上）－町丁・字等" # occupaton
}

print("Starting data loading")
total_start_time = time.perf_counter()
load_start_time = time.perf_counter()

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

end_time = time.perf_counter()
load_elapsed_time = end_time - load_start_time
print(f"\nData load executed in: {load_elapsed_time:.6f} seconds")
print("Starting data processing for IPFP")
proc_start_time = time.perf_counter()

# Columns to exclude
exclude = {"temp_level", "秘匿処理", "秘匿先情報", "合算地域", "総数"}

# Extract unique dimension labels
sexes = sorted({k[0] for k in munidata["男女_年齢_人口"]})
municipalities = sorted({k[1] for k in munidata["男女_年齢_人口"]})
ages = [a for a in next(iter(munidata["男女_年齢_人口"].values())) if a not in exclude]
inds = [a for a in next(iter(munidata["男女_産業_職業者数"].values())) if a not in exclude]
occs = [a for a in next(iter(munidata["男女_職業_職業者数"].values())) if a not in exclude]

# Create lookup dictionaries
sex_idx = {v: i for i, v in enumerate(sexes)}
muni_idx = {v: i for i, v in enumerate(municipalities)}
age_idx = {v: i for i, v in enumerate(ages)}
ind_idx = {v: i for i, v in enumerate(inds)}
occ_idx = {v: i for i, v in enumerate(occs)}

X_ages = cp.full((len(sexes), len(municipalities), len(ages)),cp.nan)
X_inds = cp.full((len(sexes), len(municipalities), len(inds)),cp.nan)
X_occs = cp.full((len(sexes), len(municipalities), len(occs)),cp.nan)

# Populate the array
for (sex, muni), data in munidata["男女_年齢_人口"].items():
    for age in ages:
        value = data.get(age, cp.nan)
        X_ages[sex_idx[sex], muni_idx[muni], age_idx[age]
               ] = float(value) if value != "-" else 0.0

for (sex, muni), data in munidata["男女_産業_職業者数"].items():
    for ind in inds:
        value = data.get(ind, cp.nan)
        X_inds[sex_idx[sex], muni_idx[muni], ind_idx[ind]
               ] = float(value) if value != "-" else 0.0

for (sex, muni), data in munidata["男女_職業_職業者数"].items():
    for occ in occs:
        value = data.get(occ, cp.nan)
        X_occs[sex_idx[sex], muni_idx[muni], occ_idx[occ]
               ] = float(value) if value != "-" else 0.0

X_pref = cp.full((len(sexes), len(ages), len(inds), len(occs)), cp.nan)
for (ign1, sex, age, ind, occ, ign2, ign3), data in prefdata['男女_年齢_職業_産業'].items():
    X_pref[sex_idx[sex], age_idx[age], ind_idx[ind], occ_idx[occ]
           ] = float(data) if data != "-" and data is not None else 0.0

X_occs = (X_occs / X_occs.sum())
X_inds = (X_inds / X_inds.sum())
#X_ages = (X_ages / X_ages.sum())
X_pref = (X_pref / X_pref.sum())

end_time = time.perf_counter()
proc_elapsed_time = end_time - proc_start_time
print(f"\nData process executed in: {load_elapsed_time:.6f} seconds")
print(f"Total execution time in: {(end_time - total_start_time):.6f} seconds")
kmeans_start_time = time.perf_counter()

def to_proportions(arr):
    totals = arr.sum(axis=2, keepdims=True)
    return arr / cp.where(totals == 0, 1, totals)

X_locals = cp.concatenate([to_proportions(X_occs), to_proportions(X_inds)], axis=2)
X_locals = X_locals.transpose(1, 0, 2)
X_locals = X_locals.reshape(X_locals.shape[0], -1)
X_locals = cp.ascontiguousarray(X_locals, dtype=cp.float32)

assert bool(cp.isfinite(X_locals).all())

n, d = X_locals.shape
k = 5
niter = 100
nredo = 5
tol = 1e-5
best_inertia, best_labels, best_centroids = gpu_kmeans(n, d, k, niter, nredo, tol, X_locals)
print(k, " Best inertia:", best_inertia)
print(k, " Cluster sizes:", cp.bincount(best_labels, minlength=k).get())
#for k in range(1,11):
#    best_inertia, best_labels, best_centroids = gpu_kmeans(n, d, k, niter, nredo, tol, X_locals)
#    print(k, " Best inertia:", best_inertia)
#    print(k, " Cluster sizes:", cp.bincount(best_labels, minlength=k).get())
X_occs = X_occs.transpose(1, 0, 2).reshape(X_occs.shape[1], -1)
X_inds = X_inds.transpose(1, 0, 2).reshape(X_inds.shape[1], -1)
X_occs_cluster = [[],[],[],[],[]]
X_inds_cluster = [[],[],[],[],[]]

for i, zone in enumerate(X_occs):
    X_occs_cluster[int(best_labels[i])].append(X_occs[i])
    X_inds_cluster[int(best_labels[i])].append(X_inds[i])

X_pref_cluster = cp.array(5)


end_time = time.perf_counter()
kmeans_elapsed_time = end_time - kmeans_start_time
print(f"\nKmeans process executed in: {kmeans_elapsed_time:.6f} seconds")
print(f"Total execution time in: {(end_time - total_start_time):.6f} seconds")

print("Starting IPFP!")
ipfp_start_time = time.perf_counter()

aggregates = [#X_ages,
    X_inds, X_occs, X_pref]

dimensions = [
    #(0, 1, 2),     # Sex x Neighborhood × Age
    (0, 1, 3),      # Sex x Neighborhood × Industry
    (0, 1, 4),      # Sex x Neighborhood × Occupation
    (0, 2, 3, 4)    # Sex × Age × Industry × Occupation
]

S, A, I, O = X_pref.shape
N = len(municipalities)
epsilon = 0 # 1e-8
X = cp.broadcast_to(
     X_pref[:, None, :, :, :] + epsilon,
     (S, N, A, I, O)
).copy()


result, converged, history = gpu_ipfp(
    original=X,
    aggregates=aggregates,
    dimensions=dimensions,
    max_iter=5000,
    tol=1e-10,
    check_every=500,
    dtype=cp.float64,
    return_history=True,
    verbose=True
)

end_time = time.perf_counter()
ipfp_elapsed_time = end_time - ipfp_start_time
print(f"\IPFP process executed in: {ipfp_elapsed_time:.6f} seconds")
print(f"Total execution time in: {(end_time - total_start_time):.6f} seconds")

breakpoint