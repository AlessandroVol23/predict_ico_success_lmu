import json

def read_feature_meta():
    with open("data/features/feature_set_meta/feature_set_meta.json") as f:
        return (json.load(f))
