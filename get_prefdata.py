import requests
import json
from pathlib import Path

def load_db(app_id, data_id, survey_year, pref_code, cache, force_refresh=False):
    cache = Path(cache)

    if cache.is_file() and not force_refresh:
        print(f"Loading from cache: {cache}")
        with cache.open("r", encoding="utf-8") as f:
            return json.load(f)
        
    print(f"Downloading from e-Stat: {data_id}")
    url = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
    params = {
        "appId": app_id,
        "statsDataId": data_id,
        "surveyYears": survey_year,
        "cdAreaFrom": f"{pref_code}000",
        "cdAreaTo": f"{pref_code}999",
        "metaGetFlg": "Y",
        "cntGetFlg": "N",
        "explanationGetFlg": "Y",
        "annotationGetFlg": "Y",
        "sectionHeaderFlg": "1",
        "replaceSpChars": "0"
    }

    res = requests.get(url, params=params, timeout=60)
    res.raise_for_status()
    txt = res.text
    lst = json.loads(txt)


    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        f.write(res.content)
    print(f"Saved to cache: {cache}")
    
    return lst

def process_db(lst, tabs_preflevel, tabs_preflevel_chop, tabs_preflevel_reps):
    for table_name in tabs_preflevel:    
        print(f"Getting DB {tabs_preflevel[table_name]} for {table_name}")
        lst_processed = lst
        if lst["GET_STATS_DATA"]["RESULT"]["STATUS"] == 0:
            print(f"Got DB {tabs_preflevel[table_name]} for {table_name}")
            for sublevel in tabs_preflevel_chop[table_name]:
                lst_processed = lst_processed[sublevel]
        else:
            print(f"Failed DB {tabs_preflevel[table_name]} for {table_name}")
            print(f"With a err {lst['GET_STATS_DATA']['RESULT']['ERROR_MSG']}")
            continue

        reps = tabs_preflevel_reps[table_name]
        reps = {
            tag: (
                {k: v for d in mapping for k, v in d.items()}
                if isinstance(mapping, tuple) else mapping
            )
            for tag, mapping in reps.items()
        }

        lst_processed = [
            {
                tag: reps[tag].get(value, value) if tag in reps else value
                for tag, value in row.items()
            }
            for row in lst_processed
            if all(
                value in reps[tag]
                for tag, value in row.items()
                if tag in reps
            )
        ]

        index_cols = []
        for i in tabs_preflevel_reps[table_name]:
            index_cols.append(i)

        lst_processed = {
            tuple(row[col] for col in index_cols):
                int(row["$"]) if row["$"].isdigit() else 0
            for row in lst_processed
        }
    return lst_processed