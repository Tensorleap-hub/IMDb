import json
from keras.preprocessing.text import tokenizer_from_json
import pandas as pd
from os.path import join

from imdb.config import CONFIG
from imdb.gcs_utils import _download
from transformers import AutoTokenizer

MODEL_TYPE = CONFIG['MODEL_TYPES'][1]  #TODO: change between bert to dense


def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer


def download_load_assets():
    cloud_path = join("assets", "imdb.csv")
    local_path = _download(cloud_path)
    df = pd.read_csv(local_path)
    if MODEL_TYPE == 'dense':
        cloud_path = join("assets", "tokenizer_v2.json")
        local_path = _download(cloud_path)
        tokenizer = load_tokenizer(local_path)  # if dense model
    else:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'], skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False, use_fast=False)  # if bert model
    return tokenizer, df
