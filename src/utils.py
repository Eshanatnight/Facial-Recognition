import os

def make_dirs_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)