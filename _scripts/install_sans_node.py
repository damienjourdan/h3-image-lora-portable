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

aitk = os.getcwd()
if not os.path.isfile(os.path.join(aitk, "run.py")):
    sys.exit("lancez ce script depuis le dossier ai-toolkit")
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
        sys.exit("Le gestionnaire d'ai-toolkit a change (nodejs.%s introuvable) : "
                 "ce script doit etre adapte avant de continuer." % _name)
_others = set(re.findall(r"nodejs\.(\w+)\(", _src)) - {"ensure_node", "ensure_ui_deps"}
if _others:
    sys.exit("Nouvelle etape Node dans le gestionnaire (%s) : script a adapter."
             % ", ".join(sorted(_others)))

nodejs.ensure_node = _skip
nodejs.ensure_ui_deps = _skip

from manager.__main__ import main  # noqa: E402

command = sys.argv[1] if len(sys.argv) > 1 else "install"
if command not in ("install", "sync"):
    sys.exit("commande attendue : install ou sync")
print(">> Installation sans Node.js / npm (UI Gradio en Python)")
sys.exit(main([command]) or 0)
