import json

def read_json_file(json_file):
    with open(json_file) as f:
        data = json.load(f)
        return data