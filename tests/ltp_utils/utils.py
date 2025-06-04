import os

from datetime import datetime

def get_env(key):
    val = os.getenv(key)
    assert val is not None and len(value.strip()), f'Env {key} is empty'
    return val

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
