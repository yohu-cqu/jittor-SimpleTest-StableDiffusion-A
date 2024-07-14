import json
import sys

def get_value_from_json(file_path, key):
    with open(file_path, 'r') as f:
        data = json.load(f)
        if key in data:
            return data[key]
        else:
            return None

if __name__ == "__main__":
    file_path = sys.argv[1]
    key = sys.argv[2]
    value = get_value_from_json(file_path, key)
    if value is not None:
        print(value)
    else:
        sys.exit(1)
