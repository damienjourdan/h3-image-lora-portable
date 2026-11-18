"""Traduction de l'interface (7 langues) sans toucher a la logique.

Le code ecrit ses textes en francais. Ce module les traduit a l'affichage :
- tables i18n/<langue>.json : texte source -> traduction ;
- les textes a trous (f-strings) sont reconnus par motif, et leurs morceaux
  variables sont traduits a leur tour (ex. un nom d'etape dans une phrase) ;
- le HTML est traduit noeud de texte par noeud de texte.
La langue choisie est enregistree dans language.txt a la racine du dossier
(partagee avec les fichiers .bat).
"""

import contextvars
import functools
import inspect
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LANG_FILE = os.path.join(ROOT, "language.txt")

LANGS = [("en", "English"), ("fr", "Français"), ("de", "Deutsch"), ("es", "Español"),
         ("pt", "Português"), ("ja", "日本語"), ("zh", "中文")]
CODES = [c for c, _ in LANGS]
DEFAULT = "en"

current_lang = contextvars.ContextVar("h3_lang", default=None)

TAG = r"</?(?:div|span|b|i|br|svg|text|line|polyline|style|a|p|small|code|table|tr|th|td)\b[^>]*>"
SPLIT = re.compile("(" + TAG + r"|(?:&nbsp;|\s)*(?:&nbsp;)+(?:\s|&nbsp;|\|)*|&#\d+;)")
STYLE_BLOCK = re.compile(r"(<style>.*?</style>)", re.S)
UNITS = re.compile(r"(\d\s?)(Go|Mo)\b")
HAS_LETTER = re.compile(r"[^\W\d_]")


def saved_lang():
    try:
        with open(LANG_FILE, encoding="utf-8") as f:
            code = f.read().strip().lower()
        if code in CODES:
            return code
    except OSError:
        pass
    env = (os.environ.get("H3_LANG") or "").strip().lower()
    return env if env in CODES else DEFAULT


def save_lang(code):
    if code in CODES and code != saved_lang():
        try:
            with open(LANG_FILE, "w", encoding="utf-8") as f:
                f.write(code)
        except OSError:
            pass


def lang_now():
    return current_lang.get() or saved_lang()


# ------------------------------------------------------------------ tables
_exact, _templates, _cache = {}, {}, {}


def _load(code):
    if code in _exact:
        return
    table = {}
    try:
        with open(os.path.join(HERE, "i18n", code + ".json"), encoding="utf-8") as f:
            table = json.load(f)
    except (OSError, ValueError):
        pass
    _exact[code] = {k.strip(): v for k, v in table.items()}
    temps = []
    for src, dst in table.items():
        src = src.strip()
        if not re.search(r"\{\d+\}", src):
            continue
        parts = re.split(r"(\{\d+\})", src)
        literal = "".join(p for p in parts if not re.fullmatch(r"\{\d+\}", p))
        if len(HAS_LETTER.findall(literal)) < 2:
            continue
        order = [int(p[1:-1]) for p in parts if re.fullmatch(r"\{\d+\}", p)]
        rx = "".join("(.*?)" if re.fullmatch(r"\{\d+\}", p) else re.escape(p).replace("\\ ", "\\s*")
                     for p in parts)
        longest = max((p for p in parts if not re.fullmatch(r"\{\d+\}", p)), key=len)
        temps.append((len(literal), re.compile(rx + r"\Z", re.S), order, dst, longest))
    temps.sort(key=lambda t: -t[0])
    _templates[code] = temps


def _line(s, code, depth=0):
    """Traduit une ligne (sans balises) : exacte, puis par motif, puis par morceaux."""
    core = s.strip()
    if not core or not HAS_LETTER.search(core):
        return s
    lead, trail = s[:len(s) - len(s.lstrip())], s[len(s.rstrip()):]
    table = _exact[code]
    hit = table.get(core)
    if hit is not None:
        return lead + hit + trail
    if depth < 3:
        for _n, rx, order, dst, longest in _templates[code]:
            if longest not in core:
                continue
            m = rx.match(core)
            if not m:
                continue
            vals = {n: _line(g, code, depth + 1) for n, g in zip(order, m.groups())}
            out = re.sub(r"\{(\d+)\}", lambda mm: vals.get(int(mm.group(1)), mm.group(0)), dst)
            return lead + out + trail
        if " : " in core:  # "Etape : explication" composee a l'execution
            a, b = core.split(" : ", 1)
            ta, tb = _line(a, code, depth + 1), _line(b, code, depth + 1)
            if ta != a or tb != b:
                sep = "：" if code in ("ja", "zh") else (" : " if code == "fr" else ": ")
                return lead + ta + sep + tb + trail
    return s


def tr(text, code=None):
    """Traduit un texte libre (texte simple, Markdown ou HTML)."""
    if not isinstance(text, str) or not text.strip():
        return text
    code = code or lang_now()
    _load(code)
    key = (code, text)
    if key in _cache:
        return _cache[key]
    whole = _exact[code].get(text.strip())
    if whole is not None:
        out = text[:len(text) - len(text.lstrip())] + whole + text[len(text.rstrip()):]
    elif re.search(TAG, text) or "&nbsp;" in text:
        chunks = STYLE_BLOCK.split(text)
        for i, chunk in enumerate(chunks):
            if chunk.startswith("<style>"):
                continue
            pieces = SPLIT.split(chunk)
            for j in range(0, len(pieces), 2):
                pieces[j] = "\n".join(_line(x, code) for x in pieces[j].split("\n"))
            chunks[i] = "".join(pieces)
        out = "".join(chunks)
    else:
        out = "\n".join(_line(x, code) for x in text.split("\n"))
    if code != "fr":  # unites : Go / Mo -> GB / MB
        out = UNITS.sub(lambda m: m.group(1) + ("GB" if m.group(2) == "Go" else "MB"), out)
    if len(_cache) > 20000:
        _cache.clear()
    if len(text) < 6000:
        _cache[key] = out
    return out


def tr_exact(text, code=None):
    """Traduit seulement un texte connu tel quel (valeurs par defaut des champs)."""
    if not isinstance(text, str):
        return text
    code = code or lang_now()
    _load(code)
    hit = _exact[code].get(text.strip())
    return text if hit is None else hit


# ------------------------------------------------------------------ Gradio
def _translate_choices(choices, code):
    out = []
    for c in choices or []:
        if isinstance(c, (list, tuple)) and len(c) == 2:
            out.append((tr(c[0], code) if isinstance(c[0], str) else c[0], c[1]))
        elif isinstance(c, str):
            t = tr(c, code)
            out.append((t, c) if t != c else c)
        else:
            out.append(c)
    return out


def localize_blocks(demo, code, keep_values=()):
    """Traduit libelles, aides, boutons, onglets et textes fixes d'une interface construite."""
    import gradio as gr

    keep = {id(c) for c in keep_values}
    _load(code)
    for block in list(demo.blocks.values()):
        for attr in ("label", "info", "placeholder"):
            val = getattr(block, attr, None)
            if isinstance(val, str) and val:
                setattr(block, attr, tr(val, code))
        if isinstance(block, (gr.Markdown, gr.HTML, gr.Button)):
            if isinstance(getattr(block, "value", None), str):
                block.value = tr(block.value, code)
        elif isinstance(block, gr.Textbox):  # valeurs par defaut connues seulement
            if isinstance(getattr(block, "value", None), str):
                block.value = tr_exact(block.value, code)
        if isinstance(block, gr.Dataframe) and isinstance(getattr(block, "headers", None), list):
            block.headers = [tr(h, code) if isinstance(h, str) else h for h in block.headers]
            if isinstance(getattr(block, "value", None), dict) and isinstance(block.value.get("headers"), list):
                block.value["headers"] = list(block.headers)
        if getattr(block, "choices", None):
            try:
                block.choices = _translate_choices(block.choices, code)
            except Exception:
                pass
    demo.title = tr(demo.title, code)


def _localize_value(value, comp, code, keep):
    import gradio as gr

    if isinstance(value, dict) and value.get("__type__") == "update":
        value = dict(value)
        for attr in ("label", "info", "placeholder"):
            if isinstance(value.get(attr), str):
                value[attr] = tr(value[attr], code)
        if "choices" in value:
            value["choices"] = _translate_choices(value["choices"], code)
        if "value" in value:
            value["value"] = _localize_value(value["value"], comp, code, keep)
        return value
    if not isinstance(value, str) or id(comp) in keep:
        return value
    if isinstance(comp, (gr.Markdown, gr.HTML, gr.Textbox)):
        return tr(value, code)
    return value


def wrap_events(demo, code, keep_values=()):
    """Chaque evenement de cette interface s'execute 'dans' sa langue et voit ses
    sorties traduites (messages, rapports, panneaux de suivi)."""
    keep = {id(c) for c in keep_values}
    fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
    for bf in fns:
        fn = getattr(bf, "fn", None)
        if fn is None or getattr(fn, "_h3_wrapped", False):
            continue
        if inspect.iscoroutinefunction(fn) or inspect.isgeneratorfunction(fn):
            continue
        outputs = list(getattr(bf, "outputs", []) or [])

        def make(fn=fn, outputs=outputs):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                token = current_lang.set(code)
                try:
                    res = fn(*args, **kwargs)
                finally:
                    current_lang.reset(token)
                if not outputs:
                    return res
                if len(outputs) == 1:
                    return _localize_value(res, outputs[0], code, keep)
                if isinstance(res, dict):
                    return {k: _localize_value(v, k, code, keep) for k, v in res.items()}
                if isinstance(res, (list, tuple)) and len(res) == len(outputs):
                    out = [_localize_value(v, c, code, keep) for v, c in zip(res, outputs)]
                    return tuple(out) if isinstance(res, tuple) else out
                return res
            wrapped._h3_wrapped = True
            return wrapped

        bf.fn = make()


SELECTOR_JS = """
(code) => {
  try {
    const tabs = Array.from(document.querySelectorAll('button[role="tab"]'));
    const i = tabs.findIndex(b => b.getAttribute('aria-selected') === 'true');
    sessionStorage.setItem('h3tab', String(Math.max(0, i)));
  } catch (e) {}
  window.location.href = '/' + code + '/';
  return code;
}
"""

# a l'ouverture d'une page : revenir a l'onglet ou l'on etait avant de changer de langue
RESTORE_TAB_JS = """
() => {
  let i = null;
  try { i = sessionStorage.getItem('h3tab'); sessionStorage.removeItem('h3tab'); } catch (e) {}
  if (i === null) return;
  let tries = 0;
  const t = setInterval(() => {
    const tabs = document.querySelectorAll('button[role="tab"]');
    if (tabs.length > Number(i)) { tabs[Number(i)].click(); clearInterval(t); }
    if (++tries > 40) clearInterval(t);
  }, 150);
}
"""
