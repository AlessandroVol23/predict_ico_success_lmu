import json

def read_feature_meta(random = False):
    if random: 
        with open("data/features/feature_set_meta/random_feature_sets_meta.json") as f:
            return (json.load(f))
    else:
        with open("data/features/feature_set_meta/feature_set_meta.json") as f:
            return (json.load(f))
