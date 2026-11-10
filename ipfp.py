from tabs import *
from pref_ipfp import pref_ipfp
import os
from dotenv import load_dotenv

import pandas as pd
import cupy as cp  

force_refresh = False

load_dotenv("api_key.env") 
app_id = os.getenv('estat_API')
year = "2020"
prefs = {"YMNK" : "19"}
tabs_preflevel = {"男女_年齢_職業_産業":"0003450560"}
tabs_munilevel = {
    "男女_産業_職業者数":"男女，産業（大分類）別就業者数（15歳以上）－町丁・字等", # industry
    "男女_職業_職業者数":"男女，職業（大分類）別就業者数（15歳以上）－町丁・字等" # occupaton
}

results, munis, labels, features = pref_ipfp(
        app_id, force_refresh, year,
        prefs,
        tabs_preflevel, tabs_munilevel, [(0,1,2), (0,1,3),(1,2,3)],
        -1, 500, 5, 1e-5,
        5000, 1e-10,
        tabs_munilevel_chop, tabs_munilevel_remrows,
        tabs_munilevel_remcols, tabs_munilevel_mergecols,
        tabs_munilevel_headers, tabs_munilevel_indices,
        tabs_preflevel_chop, tabs_preflevel_reps
)
breakpoint

# error_tbl = [[],[]]
# curr_muni = [0,0,0,0,0]
# share = [[],[]]
# for i,lbl in enumerate(labels):
#     a_tbl = results[int(lbl)].sum(axis=(2,3))[:,curr_muni[int(lbl)],:]
#     b_tbl = results[int(lbl)].sum(axis=(2,4))[:,curr_muni[int(lbl)],:]
#     error_tbl[0].append(
#         float(cp.sqrt(cp.mean((a_tbl - features[int(lbl)][0][:,curr_muni[int(lbl)],:]) ** 2))))
#     error_tbl[1].append(
#         float(cp.sqrt(cp.mean((b_tbl - features[int(lbl)][1][:,curr_muni[int(lbl)],:]) ** 2))))
#     share[0].append(features[int(lbl)][0][:,curr_muni[int(lbl)],:].sum())
#     share[1].append(features[int(lbl)][1][:,curr_muni[int(lbl)],:].sum())
#     curr_muni[int(lbl)] += 1

# error_totals = pd.DataFrame({
#     'dist': munis,
#     'clus': labels.get(),
#     'epop1':share[0],
#     'epop2':share[1],
#     'erra': error_tbl[0],
#     'errb': error_tbl[1],
# })

# error_totals.to_csv('errors_c.csv', index=False)