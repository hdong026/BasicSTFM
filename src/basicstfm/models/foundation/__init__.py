"""Built-in spatio-temporal foundation model families."""

from basicstfm.models.foundation.factost import FactoSTFoundationModel
from basicstfm.models.foundation.opencity import OpenCityFoundationModel
from basicstfm.models.foundation.unist import UniSTFoundationModel

__all__ = [
    "FactoSTFoundationModel",
    "OpenCityFoundationModel",
    "UniSTFoundationModel",
]
