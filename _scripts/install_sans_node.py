"""Install / sync the AI Toolkit environment WITHOUT Node.js or npm.

Reuses the repo's own manager (torch pin per driver, triton-windows,
flash-attn, FFmpeg, migrations...) but turns its two Node steps into no-ops:
the official Next.js UI is never built, the Gradio UI in _scripts/h3_ui.py
is used instead.

Usage (from the ai-toolkit folder):  python install_sans_node.py install|sync
"""

import os
import re
import sys


def _m(key, default):
    """Message dans la langue choisie au lancement du .bat (variables M_*)."""
    return os.environ.get("M_" + key) or default


aitk = os.getcwd()
if not os.path.isfile(os.path.join(aitk, "run.py")):
    sys.exit(_m("PY_WRONGDIR", "run this script from the ai-toolkit folder"))
sys.path.insert(0, aitk)

from manager import nodejs  # noqa: E402


def _skip(*_args, **_kwargs):
    return None


# garde-fou : si le depot renomme ces fonctions, on s'arrete au lieu de laisser
# le gestionnaire installer Node en silence
import inspect  # noqa: E402
from manager import env as _env  # noqa: E402

_src = inspect.getsource(_env.sync)
for _name in ("ensure_node", "ensure_ui_deps"):
    if not hasattr(nodejs, _name) or ("nodejs.%s(" % _name) not in _src:
        sys.exit(_m("PY_CHANGED", "The ai-toolkit manager has changed - this script must be adapted "
                    "before continuing:") + " nodejs.%s" % _name)
_others = set(re.findall(r"nodejs\.(\w+)\(", _src)) - {"ensure_node", "ensure_ui_deps"}
if _others:
    sys.exit(_m("PY_NEWSTEP", "New Node step in the ai-toolkit manager - script to adapt:")
             + " " + ", ".join(sorted(_others)))

nodejs.ensure_node = _skip
nodejs.ensure_ui_deps = _skip

from manager.__main__ import main  # noqa: E402

command = sys.argv[1] if len(sys.argv) > 1 else "install"
if command not in ("install", "sync"):
    sys.exit(_m("PY_CMD", "expected command: install or sync"))
print(_m("PY_NONODE", ">> Installation without Node.js / npm - Gradio UI in Python"))
sys.exit(main([command]) or 0)
