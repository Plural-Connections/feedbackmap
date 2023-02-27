import datetime as dt
import json
import os
import streamlit as st
import hashlib
import time

LOG_DIR = "logs"


def get_session_id():
    if "session_id" in st.session_state:
        return st.session_state["session_id"]
    else:
        return None


def log(action=None, extra_data=None):
    now = dt.datetime.now()
    time_txt = now.strftime("%Y-%m-%d %H:%M:%S")
    time_sec = int(now.strftime("%s"))
    session_id = get_session_id()
    x = {
        "time": time_txt,
        "time_sec": time_sec,
        "session_id": session_id,
        "action": action,
        "extra_data": extra_data,
    }
    with open(os.path.join(LOG_DIR, session_id + ".jsonl"), "a") as fs_out:
        print(json.dumps(x), file=fs_out)


def init():
    if not "session_id" in st.session_state:

        # Generate a session ID based on the hash of the time the user access the app
        # Obvi this can cause some issues but given our expected volume, prob not an issue
        session_id = hashlib.md5(str(time.time()).encode("utf-8")).hexdigest()[:10]
        st.session_state["session_id"] = session_id
