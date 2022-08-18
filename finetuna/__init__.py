import finetuna

with open(finetuna.__path__[0] + "/__version__") as v:
    __version__ = v.read()
from .main import *
