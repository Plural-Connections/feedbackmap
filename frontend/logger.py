import datetime as dt
import json
import os
import streamlit as st

LOG_DIR = "logs"


def log(action, dashboard_state, query_params=None):
    referral_code = (
        st.session_state.get("referral_code")
        or query_params.get("referral_code", [None])[0]
        or "None"
    )
    now = dt.datetime.now()
    time_txt = now.strftime("%Y-%m-%d %H:%M:%S")
    time_sec = int(now.strftime("%s"))
    x = {
        "time": time_txt,
        "time_sec": time_sec,
        "referral_code": referral_code,
        "dashboard_state": dashboard_state,
        "action": action,
        "query_params": query_params,
    }
    with open(os.path.join(LOG_DIR, referral_code + ".jsonl"), "a") as fs_out:
        print(json.dumps(x), file=fs_out)


def init():
    pass
