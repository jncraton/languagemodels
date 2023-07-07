import re
from languagemodels.models import convert_to_gb, models

model_names = {m['name']: m for m in models}

def validate_model(model_name):
    return model_names[model_name]['name']

schema = {
    "max_ram": convert_to_gb,
    "license_filter": re.compile,
    "instruct_model": validate_model
}

def convert_to_gb(space):
    """Convert max RAM string to int

    Output will be in gigabytes

    If not specified, input is assumed to be in gigabytes

    >>> convert_to_gb("512")
    512.0

    >>> convert_to_gb(".5")
    0.5

    >>> convert_to_gb("4G")
    4.0

    >>> convert_to_gb("256mb")
    0.25

    >>> convert_to_gb("256M")
    0.25
    """

    if isinstance(space, int) or isinstance(space, float):
        return float(space)

    multipliers = {
        "g": 1.0,
        "m": 2 ** -10,
    }

    space = space.lower()
    space = space.rstrip("b")

    if space[-1] in multipliers:
        return float(space[:-1]) * multipliers[space[-1]]
    else:
        return float(space)

class Config(dict):
    """
    >>> c = Config({'max_ram': '4gb'})
    >>> c
    {'max_ram': 4.0}

    >>> c = Config({'instruct_model': 'bad_model'})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_model'

    >>> c = Config({'bad_value': 1})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_value'
    """
    def __init__(self, config={}):
        for key in config.keys():
            self[key] = schema[key](config[key])
