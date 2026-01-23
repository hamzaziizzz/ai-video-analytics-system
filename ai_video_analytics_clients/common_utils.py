import base64
from distutils import util
from itertools import chain, islice
from typing import List, Union


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def read_image(path):
    with open(path, mode='rb') as fl:
        data = fl.read()
        return data


def b64_encode_data(data: List[Union[bytes, str]]) -> List[str]:
    data_encoded = []
    for item in data:
        if isinstance(item, bytes):
            item = base64.b64encode(item).decode('ascii')
        data_encoded.append(item)
    return data_encoded


def decode_people_data(response: dict):
    data = response.get('data', [])
    if data:
        for item in data:
            entries = item.get('people') or []
            for entry in entries:
                blob = entry.get("persondata")
                if blob and isinstance(blob, str):
                    entry["persondata"] = base64.b64decode(blob)
    return response




def to_bool(value):
    try:
        return bool(util.strtobool(value))
    except:
        return False
