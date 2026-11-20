import importlib
import json
import platform
import sys
import sympy

case=sys.argv[1]
if case not in ('green','exterior','wave','analytic'):
    raise ValueError(case)
print(json.dumps(dict(case=case,python=platform.python_version(),sympy=sympy.__version__,
                      result=importlib.import_module(case).run()),indent=2,sort_keys=True))
