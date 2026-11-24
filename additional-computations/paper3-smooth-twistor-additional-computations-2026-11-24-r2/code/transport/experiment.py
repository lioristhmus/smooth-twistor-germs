import json,sys
from pathlib import Path
if sys.argv[1]=='cr':
    from continuum import extract
    data=extract()
elif sys.argv[1]=='readback':
    from endpoint import run
    data=run()
else:
    from transport import run
    data=run(sys.argv[1])
Path('results.json').write_text(json.dumps(data,indent=2,allow_nan=False))
