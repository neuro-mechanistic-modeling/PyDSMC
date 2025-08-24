# pydsmc/__init__.py
from importlib.metadata import version

from pydsmc.evaluator import Evaluator
from pydsmc.json_parser import jsons_to_df
from pydsmc.property import (
    Property,
    create_custom_property,
    create_predefined_property,
    get_predefined_properties,
)
from pydsmc.utils import create_eval_envs

__version__ = version(__name__)
__all__ = [
    "Evaluator",
    "Property",
    "create_custom_property",
    "create_eval_envs",
    "create_predefined_property",
    "get_predefined_properties",
    "jsons_to_df",
]
