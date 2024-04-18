import json
import os
import warnings


def create_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"{dir_path} : has been created successfully")
    else:
        print(f"{dir_path} : has already exist")


def jload(path, lines=True):
    if not os.path.exists(path):
        raise ValueError("path not exist")
    if lines is True:
        data = [json.loads(_) for _ in open(path, "r") if _ != ""]
    else:
        with open(path, "r") as f:
            data = json.load(f)
    return data


def jdump(obj, path, mode="w", lines=True, indent=4):
    assert type(obj) is list or type(obj) is dict
    if mode != "w" and mode != "a":
        raise ValueError("error mode")
    if not os.path.exists(os.path.dirname(path)):
        create_dir(path)
    if type(obj) is dict:
        obj = [obj]

    if lines is True:
        with open(path, mode) as f:
            for item in obj:
                f.write(json.dumps(item) + "\n")
    else:
        if mode == "w":
            with open(path, "w") as f:
                json.dump(obj, f, indent=indent)
        elif mode == "a":
            with open(path, "a+") as f:
                data = json.load(path)
                json.dump(data + obj, f, indent=indent)
