
import numpy as np
import click
from src.features.feature_class import FeatureEngineering
import logging
import json
from tqdm import tqdm
from random import seed
from random import randint
from time import time

seed(123)

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

def _read_feature_set():
    with open("data/features/featuer_list.json") as f:
        return (json.load(f))

def _select_random_features(feature_list, amount):
    """Selects a given amount of random features from the feature list"""
    set_size = len(feature_list) -1
    random_features = []
    for i in range(amount):
        while(True):
            random_feature = feature_list[randint(0, set_size)]
            if(random_feature in random_features):
                continue
            else:
                random_features.append(random_feature)
                break
    return random_features

def _select_random_meta_values(feature_list, meta, meta_name):
    random_features =[]
    for feature in feature_list:
        if((meta) not in feature):
            random_features.append(feature)
            continue

        meta_list = feature[meta]
        meta_list_size = len(meta_list) -1

        generated_feature = {x: feature[x] for x in feature if x != meta}
        generated_feature[meta_name] = meta_list[randint(0, meta_list_size)]

        random_features.append(generated_feature)

    return random_features


def _write_random_feature_sets(feature_sets):
    with open("data/features/feature_set_meta/random_feature_sets_meta.json", "w") as f:
        json.dump(feature_sets, f)

@click.command()
@click.argument('max')
def main(max = 10):
    feature_list = _read_feature_set()
    random_feature_sets = {}
    timestamp = str(int(time()))
    for i in tqdm(range(int(max))):
        tqdm.write("building random feature_set {} ".format(i))
        feature_amount = randint(1,60)
        random_feature_set = _select_random_features(feature_list,feature_amount)
        random_feature_set = _select_random_meta_values(random_feature_set, "na_strategies", "na_strategy")
        random_feature_set = _select_random_meta_values(random_feature_set, "encoders", "encoder")
        random_feature_sets[timestamp+ "_"+ str(i +1)] = random_feature_set
    _write_random_feature_sets(random_feature_sets)
    

if __name__ == "__main__":
    main()
