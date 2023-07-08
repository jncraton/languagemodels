import re
import os
from languagemodels.models import models


class Config(dict):
    """
    Store configuration information for the package.

    This is a dictionary that provides data basic data validation.

    Only appropriate keys and values are allowed to be set.

    >>> c = Config({'max_ram': '4gb'})
    >>> c
    {'max_ram': 4.0}

    >>> c = Config({'instruct_model': 'flan-t5-small-ct2-int8'})
    >>> c
    {'instruct_model': 'flan-t5-small-ct2-int8'}

    >>> c = Config({'license_filter': 'apache.*|mit'})
    >>> c
    {'license_filter': re.compile('apache.*|mit')}

    >>> c = Config({'instruct_model': 'flan-t5-bad-ct2-int8'})
    Traceback (most recent call last):
      ...
    KeyError: 'flan-t5-bad-ct2-int8'

    >>> c = Config({'bad_value': 1})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_value'

    >>> c = Config()
    >>> c.update({'bad_value': 1})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_value'

    """

    model_names = {m["name"]: m for m in models}

    def __init__(self, config={}):
        # Environment variables are loaded first
        for key in Config.schema:
            value = os.environ.get(f"LANGUAGEMODELS_{key.upper()}")
            if value:
                self[key] = value

        # Any values passed in the config dict override environment vars
        for key in config.keys():
            self[key] = config[key]

    def __setitem__(self, key, value):
        super().__setitem__(key, Config.schema[key](value))

    def update(self, other):
        for key in other:
            self[key] = other[key]

    @staticmethod
    def validate_model(model_name):
        return Config.model_names[model_name]["name"]

    @staticmethod
    def convert_to_gb(space):
        """Convert max RAM string to int

        Output will be in gigabytes

        If not specified, input is assumed to be in gigabytes

        >>> Config.convert_to_gb("512")
        512.0

        >>> Config.convert_to_gb(".5")
        0.5

        >>> Config.convert_to_gb("4G")
        4.0

        >>> Config.convert_to_gb("256mb")
        0.25

        >>> Config.convert_to_gb("256M")
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


Config.schema = {
    "max_ram": Config.convert_to_gb,
    "license_filter": re.compile,
    "instruct_model": Config.validate_model,
}
