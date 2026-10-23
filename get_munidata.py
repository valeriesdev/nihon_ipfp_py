import requests
import json
import pandas as pd
from pathlib import Path

def load_csv(url, cache, force_refresh=False, encoding="cp932"):
    cache = Path(cache)

    if not cache.is_file() or force_refresh:
        print(f"Downloading CSV: {url}")
        res = requests.get(url, timeout=120)
        res.raise_for_status()

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(res.content)
    else:
        print(f"Loading from cache: {cache}")

    return pd.read_csv(cache, encoding=encoding)

def load_url(app_id, stats_code, survey_year, tab, pref, cache, force_refresh):
    url = "https://api.e-stat.go.jp/rest/3.0/app/json/getDataCatalog"
    cache = Path(cache)

    if cache.is_file() and not force_refresh:
        print(f"Loading url from cache: {cache}")
        with cache.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"Downloading url for {tab} {pref}")

    params = {
        "appId": app_id,
        "statsCode": "00200521",
        "surveyYears": survey_year,
        "searchKind": 2,
        "searchWord": f"{tab} {pref}",
        "limit": 100
    }

    # Request data catalog
    res = requests.get(url, params=params, timeout=60)
    res.raise_for_status()

    txt = res.text
    lst = json.loads(txt)
    if lst["GET_DATA_CATALOG"]["RESULT"]["STATUS"] == 0:
        print(f"Got URL {tab} {pref}")
    else:
        print(f"Failed URL {tab} {pref}")
        print(f"With a err {lst['GET_DATA_CATALOG']['RESULT']['ERROR_MSG']}")
        return None
    
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        f.write(res.content)
    print(f"Saved to cache: {cache}")
    return lst

def process_url(lst, tabs_munilevel_chop, remrows, remcols, mergecols, headers, twist_indices, cache, force_refresh):
    for sublevel in tabs_munilevel_chop:
        lst = lst[sublevel]

    lst = load_csv(lst, cache, force_refresh)

    # Merge columns selected to be merged
    for new_col, indices in mergecols.items():
        cols = lst.iloc[:, list(indices)]
        lst[new_col] = cols.apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1, min_count=1)

    # Remove specified rows by position
    lst = lst.drop(index=lst.index[list(remrows)])

    # Remove specified columns
    lst = lst.drop(columns=lst.columns[list(remcols)])

    # Reset row indices
    lst = lst.reset_index(drop=True)
    lst.columns = headers
    lst = lst.set_index(lst.columns[:twist_indices].tolist())
    lst = lst.to_dict(orient="index")

    # Remove elements with unnneded values 
    lst = {k: v for k, v in lst.items() if k[0] != "総数"}
    lst = {k: v for k, v in lst.items() if k[2] != "-"}

    # Remove higher level classifications
    lst_t = list(lst.items())
    lst_t.append(lst_t[0])
    lst = {
        k1: v1 for (k1, v1), (k2, v2) in zip(lst_t, lst_t[1:])
        if int(v1["temp_level"]) >= int(v2["temp_level"]) or not k2[2].startswith(k1[2])
    }

    # - to 0
    lst = {
        k: {col: 0 if x == "-" or x is None or x != x else x for col, x in v.items()}
        for k, v in lst.items()
        if "X" not in v.values()
    }

    # merge muni and neighborhood
    lst = {(k[0], k[1] + k[2]): v for k, v in lst.items()}

    return lst