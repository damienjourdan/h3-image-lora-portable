"""Logique (sans interface) de l'UI portable MiniMax-H3-Image / ai-toolkit.

Tout ce qui est ici est testable sans Gradio : lancement et pilotage des jobs,
pause / reprise, file d'attente, calibrage, moniteur memoire, anti-veille,
notifications, courbe de perte, legendage, test de LoRA, reglages.
"""

import codecs
import collections
import contextlib
import csv
import datetime
import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AITK = os.path.join(ROOT, "ai-toolkit")
OUTPUT = os.path.join(ROOT, "output")
DATASETS = os.path.join(ROOT, "datasets")
CONFIGS = os.path.join(ROOT, "configs")
CACHE = os.path.join(ROOT, "cache")
MODELS = os.environ.get("MODELS_PATH", os.path.join(ROOT, "models"))
DB_PATH = os.path.join(ROOT, "h3_ui.db")
SETTINGS_PATH = os.path.join(ROOT, "reglages_ui.json")
PIN_FILE = os.path.join(ROOT, "_scripts", "version_figee.txt")
RUN_JOB = os.path.join(ROOT, "_scripts", "run_job.py")
PY = sys.executable
IS_WIN = os.name == "nt"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

for _d in (OUTPUT, DATASETS, CONFIGS, CACHE, MODELS):
    os.makedirs(_d, exist_ok=True)

H3_REPO = "Comfy-Org/MiniMax-H3-Image"
ADAPTER = "ostris/minimax_h3_training_adapter/minimax_h3_image_training_adapter_v1.safetensors"
FILES_COMMON = [
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_image_vae_fp16.safetensors",
]
FILE_W4A8 = "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors"
FILE_INT8 = "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"

PRESET_INT8 = "int8 + offload RAM - defaut, chemin sur pour la 3090"
PRESET_W4A8 = "w4a8 - EXPERIMENTAL sur 3090 (plus leger, sans offload)"

CALIB_NAME = "_calibrage"
CALIB_STEPS = 20
TEST_PREFIX = "_test_"
REG_PREFIX = "_reg_"

# ordre des champs de l'onglet Entrainement (UI <-> dictionnaire de parametres)
PARAM_KEYS = ["name", "folder", "ctrl", "preset", "offload", "resolutions", "steps", "lr",
              "rank", "trigger", "save_every", "sample_every", "sample_size", "no_sample",
              "prompts", "w4a8_fix",
              "process", "optimizer", "ema", "adapter", "likeness", "sampler", "control",
              "dop", "dop_class", "reg", "reg_weight"]

# ---- procedes d'entrainement
PROC_STANDARD = "Standard ai-toolkit"
PROC_FAST = "Rapide 512 (reglages Fizgig, composants ai-toolkit)"
PROC_FIZGIG = "Recette Fizgig complete (experimental)"
PROCESSES = [PROC_STANDARD, PROC_FAST, PROC_FIZGIG]

OPT_ADAMW = "adamw8bit (taux fixe)"
OPT_AUTO4 = "automagic4 (taux auto-regle, plafond 2,5e-4)"
OPT_AUTO3 = "automagic3 (taux auto-regle)"
OPTIMIZERS = {OPT_ADAMW: "adamw8bit", OPT_AUTO4: "automagic4", OPT_AUTO3: "automagic3"}

EMA_OFF = "Aucune"
EMA_098 = "0,98 (reglage Fizgig)"
EMA_CHOICES = {EMA_OFF: 0.0, EMA_098: 0.98}

ADAPTER_OSTRIS = "ostris (celui d'ai-toolkit)"
ADAPTER_FIZGIG = "Fizgig (telecharge et converti, experimental)"
ADAPTERS = [ADAPTER_OSTRIS, ADAPTER_FIZGIG]
FIZGIG_ADAPTER_PATH = os.path.join(MODELS, "loras", "training_adapters",
                                   "fizgig_h3_image_training_adapter_aitk.safetensors")

SAMPLER_EULER = "Euler (celui du modele)"
SAMPLER_LCM = "LCM (peau plus naturelle selon Fizgig, experimental)"
SAMPLERS = [SAMPLER_EULER, SAMPLER_LCM]


def count_images(folder):
    try:
        return sum(1 for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
    except OSError:
        return 0


def process_values(process, folder=""):
    """Valeurs que le procede choisi impose aux champs de l'onglet Entrainement,
    ou None pour le procede Standard (rien n'est modifie)."""
    if process == PROC_STANDARD:
        return None
    n = count_images(folder)
    steps = max(300, 30 * n) if n else 600        # 30 passages sur le dataset
    every = max(50, int(round(steps / 10 / 10.0)) * 10)
    full = process == PROC_FIZGIG
    return {"resolutions": ["512"], "rank": 32, "lr": 2.5e-4, "steps": steps,
            "save_every": every, "sample_every": every,
            "optimizer": OPT_AUTO4, "ema": EMA_098,
            "adapter": ADAPTER_FIZGIG if full else ADAPTER_OSTRIS,
            "likeness": full, "sampler": SAMPLER_LCM if full else SAMPLER_EULER}

CAPTION_PROMPT = (
    "Caption this image as if you were going to try to generate it with an image generator. "
    "Be thorough and describe everything in the image. Be decisive by stating things as they "
    "are. Do not say things like \"It appears that\" Or \"possibly\". Start out with things "
    "like \"A person on the beach\" or \"A black dragon\". No preamble. Just get to the point."
)
LORA_SUBJECT = "Personnage / sujet"
LORA_STYLE = "Style"
LORA_GENERAL = "General (tout decrire)"
LORA_TYPES = [LORA_SUBJECT, LORA_STYLE, LORA_GENERAL]

# mots qui decrivent un trait permanent d'un personnage : a bannir des legendes
# d'un LoRA de sujet, sinon le trait s'attache a ces mots et non au mot declencheur
TRAIT_WORDS = ["hair", "haired", "hairstyle", "blond", "blonde", "brunette", "redhead", "bald",
               "balding", "mohawk", "beard", "bearded", "goatee", "mustache", "moustache",
               "stubble", "muscular", "muscled", "muscles", "physique", "scar", "scars", "scarred",
               "wrinkle", "wrinkles", "wrinkled", "tattoo", "tattoos", "tattooed", "freckles",
               "skin", "tanned", "eyes", "eyebrows", "jawline", "middle-aged",
               "elderly", "veins", "veined"]
MULTI_HINTS = ["two men", "two women", "two people", "two figures", "three views", "two views",
               "both men", "both women", "character sheet", "multiple views", "side by side",
               "second man", "second woman", "front and back"]


def caption_prompt(lora_type, trigger="", extra_traits=""):
    """Consigne donnee au modele de vision, selon le type de LoRA."""
    trigger = (trigger or "").strip()
    extra = ", ".join(t.strip() for t in re.split(r"[,;\n]", extra_traits or "") if t.strip())
    tail = (' Be factual and decisive: no preamble, no "appears to" or "possibly". '
            "Write one short paragraph in English.")
    if lora_type == LORA_SUBJECT:
        start = (f'Start the caption with "{trigger}, " and refer to the main subject only as '
                 f'"{trigger}", "he" or "she".' if trigger else
                 'Refer to the main subject only with a bare generic noun such as "a man", '
                 '"a woman" or "a creature".')
        return ("Write a training caption for a character LoRA. " + start + " Never describe the "
                "subject's permanent physical traits: no hair, hairstyle, hair color, facial hair, "
                "face, eyes, age, skin, body build, muscles, scars, wrinkles or tattoos"
                + (f", and never mention: {extra}" if extra else "") + ". Describe only what "
                "changes from one image to another, in this order: image medium (photo, 3D render, "
                "illustration), framing (close-up portrait, upper body, full body), viewing angle, "
                "pose and facial expression, clothing and accessories, background, lighting. "
                "If the image shows several views or several people, say so explicitly." + tail)
    if lora_type == LORA_STYLE:
        return ("Write a training caption for a style LoRA. Describe the content of the image "
                "thoroughly: subjects and their appearance, actions, setting, composition and "
                "framing. Do not describe the artistic style, medium, rendering technique, color "
                "grading or mood: no words such as painting, illustration, render, cinematic or "
                "stylized." + tail)
    return CAPTION_PROMPT


def caption_check(folder, lora_type, trigger="", extra_traits=""):
    """Controle des legendes existantes ; texte vide s'il n'y a rien a signaler."""
    if not folder or not os.path.isdir(folder):
        return ""
    trigger = (trigger or "").strip()
    extras = [t.strip().lower() for t in re.split(r"[,;\n]", extra_traits or "") if t.strip()]
    caps = {}
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".txt") and any(
                os.path.isfile(os.path.join(folder, os.path.splitext(f)[0] + e)) for e in IMG_EXT):
            try:
                with open(os.path.join(folder, f), encoding="utf-8", errors="replace") as fh:
                    caps[f] = fh.read().strip()
            except OSError:
                pass
    if not caps:
        return ""
    lines = ["", f"Legendes ({lora_type}) :"]
    several = re.compile(r"\b(two|three|both|several) (\w+ ){0,2}(men|women|people|persons|figures|"
                         r"characters|views)\b", re.I)
    multi = [f for f, c in caps.items()
             if several.search(c) or any(h in c.lower() for h in MULTI_HINTS)]
    if multi:
        lines.append(f"- {len(multi)} image(s) semblent contenir plusieurs vues ou plusieurs personnes : "
                     "decoupez-les en une image par vue, ou retirez-les : " + ", ".join(multi[:8]))
    if lora_type == LORA_SUBJECT:
        words = TRAIT_WORDS + extras
        pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
        flagged = {}
        for f, c in caps.items():
            hits = sorted({m.group(1).lower() for m in pattern.finditer(c)})
            if hits:
                flagged[f] = hits
        if flagged:
            lines.append(f"- {len(flagged)}/{len(caps)} legende(s) decrivent des traits permanents du sujet. "
                         "Ces traits s'attacheront a ces mots et non au mot declencheur : retirez-les, "
                         "ou relancez le legendage en mode Personnage avec 'Refaire les legendes'.")
            for f, hits in list(flagged.items())[:10]:
                lines.append(f"    {f} : {', '.join(hits[:8])}")
            if len(flagged) > 10:
                lines.append(f"    ... et {len(flagged) - 10} autre(s)")
        else:
            lines.append("- Aucune legende ne decrit de trait permanent du sujet.")
        if trigger:
            missing = [f for f, c in caps.items() if trigger.lower() not in c.lower()]
            if missing:
                lines.append(f"- {len(missing)} legende(s) sans le mot declencheur '{trigger}' : ai-toolkit "
                             "l'ajoutera au debut, mais une legende qui le nomme comme sujet apprend mieux.")
            if re.fullmatch(r"[A-Za-z]+", trigger) and len(trigger) >= 3:
                lines.append(f"- Le mot declencheur '{trigger}' ressemble a un vrai mot ou a un nom que le modele "
                             "peut deja connaitre : preferez un mot invente avec des chiffres (ex. m41ne).")
        else:
            lines.append("- Aucun mot declencheur renseigne (onglet Entrainement) : conseille pour un sujet.")
    elif lora_type == LORA_STYLE:
        style_words = re.compile(r"\b(painting|illustration|render|rendered|cinematic|stylized|"
                                 r"watercolor|anime|comic|photorealistic|aesthetic)\b", re.I)
        flagged = [f for f, c in caps.items() if style_words.search(c)]
        if flagged:
            lines.append(f"- {len(flagged)}/{len(caps)} legende(s) nomment le style ou la technique : "
                         "pour un LoRA de style, mieux vaut ne decrire que le contenu.")
    return "\n".join(lines) if len(lines) > 2 else ""


CAPTION_MODELS = ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"]


def safe_name(name, default="h3img_lora"):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", (name or default).strip()) or default


# =========================================================== reglages globaux
DEFAULT_SETTINGS = {"sound": True, "ntfy_topic": "", "comfy_loras_dir": "", "keep_awake": True,
                    # temperature GPU : alerte, puis pause automatique (0 = desactive)
                    "temp_alert": 78, "temp_pause": 88,
                    # limite de puissance appliquee pendant les jobs, en watts (0 = ne pas toucher)
                    "power_limit_w": 0,
                    # thermostat : ajuste la limite de puissance pour tenir une temperature cible
                    "thermo_on": False, "thermo_target": 72, "thermo_min_w": 150,
                    # cout de l'electricite consommee par le GPU (0 = kWh seulement ; "" = devise selon la langue)
                    "kwh_price": 0.0, "currency": "",
                    # ressemblance / fidelite au prompt des apercus (processeur, modeles ~700 Mo)
                    "likeness_on": True,
                    # pause automatique quand le personnage apparait aussi sans mot declencheur
                    "leak_pause": False,
                    # balayage de puissance au debut de l'entrainement (droits administrateur)
                    "sweep_on": False, "sweep_min_w": 200, "sweep_step_w": 30,
                    # generateur MiniMaxH3-Portable (dossier ; vide = detection automatique)
                    "gen_dir": ""}

# devises des langues de l'interface : code -> (symbole, decimales, symbole devant)
CURRENCIES = {"EUR": ("€", 2, False), "USD": ("$", 2, True), "GBP": ("£", 2, True),
              "BRL": ("R$", 2, True), "JPY": ("¥", 0, True), "CNY": ("¥", 2, True)}
LANG_CURRENCY = {"en": "USD", "fr": "EUR", "de": "EUR", "es": "EUR", "pt": "EUR", "ja": "JPY", "zh": "CNY"}


def currency_for(lang, cfg=None):
    cur = ((cfg or load_settings()).get("currency") or "").upper()
    return cur if cur in CURRENCIES else LANG_CURRENCY.get(lang, "EUR")


def fmt_money(amount, cur):
    sym, dec, before = CURRENCIES.get(cur, CURRENCIES["EUR"])
    if dec and amount < 0.1:
        dec = 3   # quelques centimes : un chiffre de plus pour voir le compteur bouger
    txt = f"{amount:.{dec}f}"
    return f"{sym}{txt}" if before else f"{txt} {sym}"


def _hms_seconds(text):
    """'05:12', '1:02:03' ou '1 day, 2:03:04' (format tqdm) -> secondes."""
    try:
        days = 0
        if "day" in text:
            d, text = text.split(",", 1)
            days = int(d.split()[0])
        parts = [float(x) for x in text.strip().split(":")]
        sec = 0.0
        for x in parts:
            sec = sec * 60 + x
        return days * 86400 + sec
    except (ValueError, AttributeError, IndexError):
        return None


def energy_text(eta=None, lang=None):
    """Consommation du GPU depuis le debut du job, son cout, et le cout projete a la fin."""
    wh = MON.job_wh
    if wh <= 0:
        return ""
    cfg = load_settings()
    price = float(cfg.get("kwh_price") or 0)
    kwh = wh / 1000
    if price <= 0:
        return f"GPU {kwh:.2f} kWh"
    cur = currency_for(lang, cfg)
    spent = fmt_money(kwh * price, cur)
    left = _hms_seconds(eta) if eta else None
    if left and MON.avg_w:
        proj = fmt_money((wh + MON.avg_w * left / 3600) / 1000 * price, cur)
        return f"GPU {kwh:.2f} kWh = {spent}, ~{proj} a la fin"
    return f"GPU {kwh:.2f} kWh = {spent}"


def energy_summary(wh, lang=None):
    if not wh:
        return ""
    cfg = load_settings()
    price = float(cfg.get("kwh_price") or 0)
    txt = f"GPU {wh / 1000:.2f} kWh"
    if price > 0:
        txt += f" = {fmt_money(wh / 1000 * price, currency_for(lang, cfg))}"
    return txt
THERMO_PERIOD = 45     # secondes entre deux ajustements (inertie thermique de la carte)
THERMO_STEP = 10       # watts par ajustement
THERMO_BAND_HIGH = 2   # degres au-dessus de la cible avant de baisser la puissance
THERMO_BAND_LOW = 3    # degres en dessous de la cible avant de la remonter
TEMP_PAUSE_DELAY = 60  # secondes consecutives au-dessus du seuil avant la pause


def load_settings():
    data = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            data.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return data


def save_settings(**changes):
    data = load_settings()
    data.update(changes)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return data


def job_settings_path(name):
    return os.path.join(CONFIGS, f"{safe_name(name)}.reglages.json")


def save_job_settings(params):
    with open(job_settings_path(params["name"]), "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, ensure_ascii=False)


def list_job_settings():
    return sorted(os.path.basename(p)[: -len(".reglages.json")]
                  for p in glob.glob(os.path.join(CONFIGS, "*.reglages.json")))


def load_job_settings(name):
    try:
        with open(job_settings_path(name), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# =========================================================== base de controle
# ai-toolkit (trainer "diffusion_trainer") lit une table SQLite "Job" a la fin
# de chaque pas : save_now=1 -> sauvegarde immediate, sample_now=1 -> apercus,
# stop=1 -> arret propre. C'est le mecanisme de son UI officielle.
def _db():
    con = sqlite3.connect(DB_PATH, timeout=30.0)
    con.isolation_level = None
    return contextlib.closing(con)


def db_init_job(job_id):
    with _db() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS Job (id TEXT PRIMARY KEY, name TEXT, gpu_ids TEXT DEFAULT '0',"
            " job_config TEXT DEFAULT '', status TEXT DEFAULT 'stopped', stop INTEGER DEFAULT 0,"
            " return_to_queue INTEGER DEFAULT 0, step INTEGER DEFAULT 0, total_steps INTEGER,"
            " info TEXT DEFAULT '', speed_string TEXT DEFAULT '', queue_position INTEGER DEFAULT 0,"
            " pid INTEGER, job_type TEXT DEFAULT 'train', job_ref TEXT,"
            " save_now INTEGER DEFAULT 0, sample_now INTEGER DEFAULT 0)")
        con.execute(
            "INSERT OR REPLACE INTO Job (id, name, status, info, stop, return_to_queue, save_now, sample_now)"
            " VALUES (?, ?, 'running', 'Starting', 0, 0, 0, 0)", (job_id, job_id))


def db_set(job_id, key, value):
    assert key in ("stop", "save_now", "sample_now")
    try:
        with _db() as con:
            con.execute(f"UPDATE Job SET {key} = ? WHERE id = ?", (int(value), job_id))
        return True
    except sqlite3.Error:
        return False


def db_get(job_id):
    try:
        with _db() as con:
            row = con.execute(
                "SELECT status, info, step, speed_string, save_now, sample_now FROM Job WHERE id = ?",
                (job_id,)).fetchone()
        if row:
            return dict(zip(("status", "info", "step", "speed", "save_now", "sample_now"), row))
    except sqlite3.Error:
        pass
    return None


# =========================================================== moniteur memoire
def _ram_gb():
    """(utilisee, totale) en Go, sans dependance obligatoire."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 1e9, vm.total / 1e9
    except Exception:
        pass
    try:
        if IS_WIN:
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return (st.ullTotalPhys - st.ullAvailPhys) / 1e9, st.ullTotalPhys / 1e9
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return (info["MemTotal"] - info["MemAvailable"]) / 1e9, info["MemTotal"] / 1e9
    except Exception:
        return None, None


def _num(text):
    """'123.4' -> 123.4 ; '[N/A]' ou vide -> None."""
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def nvidia_smi(*args, timeout=8):
    """(code de retour, sortie) ; (1, message) si nvidia-smi est indisponible."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                           timeout=timeout, creationflags=flags)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as exc:
        return 1, str(exc)


def is_admin():
    try:
        if IS_WIN:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return False


# =========================================================== limite de puissance
def power_info():
    """Limites de puissance du GPU en watts : dict min / max / default / current."""
    code, out = nvidia_smi("--query-gpu=power.min_limit,power.max_limit,power.default_limit,power.limit",
                           "--format=csv,noheader,nounits")
    if code != 0:
        return None
    vals = [_num(x) for x in out.strip().splitlines()[0].split(",")]
    if len(vals) < 4 or any(v is None for v in vals[:3]):
        return None
    return dict(zip(("min", "max", "default", "current"), vals))


def power_text():
    info = power_info()
    if not info:
        return "Limites de puissance non lisibles sur cette carte (nvidia-smi)."
    cur = f"{info['current']:.0f} W" if info["current"] is not None else "?"
    want = int(load_settings().get("power_limit_w") or 0)
    plan = (f"Reglage enregistre : {want} W, applique automatiquement au lancement de chaque job "
            "(ou tout de suite avec le bouton 'Appliquer maintenant').\n" if want else
            "Aucune limite enregistree : la puissance n'est pas modifiee pendant les jobs.\n")
    return (plan + f"Limite actuelle : {cur}. Plage autorisee : {info['min']:.0f} a {info['max']:.0f} W "
            f"(defaut {info['default']:.0f} W). "
            + ("UI lancee en administrateur : aucune confirmation ne sera demandee."
               if is_admin() else
               "UI sans droits administrateur : Windows demandera une confirmation a chaque "
               "changement (sauf jobs lances par la file d'attente, ou la limite n'est pas modifiee)."))


def _nvidia_smi_elevated(*args, timeout=90):
    """Windows, UI sans droits administrateur : lance nvidia-smi avec elevation.
    Windows affiche sa fenetre de confirmation (UAC) ; rien d'autre n'est eleve,
    et l'entrainement en cours n'est pas interrompu. Retourne True si accepte."""
    if not IS_WIN:
        return False
    arglist = ",".join("'%s'" % a for a in args)
    script = ("try { $p = Start-Process -FilePath 'nvidia-smi' -ArgumentList %s -Verb RunAs "
              "-WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode } catch { exit 1223 }" % arglist)
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True,
                           timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
        return r.returncode == 0
    except Exception:
        return False


def _pl(watts, allow_prompt):
    """Execute nvidia-smi -pl ; avec allow_prompt, demande l'elevation si necessaire.
    Le resultat est verifie en relisant la limite reellement en vigueur."""
    code, out = nvidia_smi("-pl", str(watts))
    if code != 0 and allow_prompt and not is_admin():
        _nvidia_smi_elevated("-pl", str(watts))
    now = power_info()
    if now and now["current"] is not None and abs(now["current"] - watts) <= 1:
        return True, ""
    if not is_admin():
        return False, ("confirmation Windows refusee ou absente" if allow_prompt and IS_WIN
                       else "droits administrateur requis : cliquez 'Appliquer maintenant' "
                            "et acceptez la demande de Windows")
    return False, (out.strip().splitlines() or ["refus de nvidia-smi"])[-1]


def set_power_limit(watts, allow_prompt=False):
    """Applique une limite de puissance. Retourne (ok, message)."""
    info = power_info()
    if not info:
        return False, "limite de puissance non prise en charge par cette carte"
    watts = int(max(info["min"], min(info["max"], watts)))
    if info["current"] is not None and abs(info["current"] - watts) <= 1:
        return True, f"limite de puissance {watts} W deja en place"
    ok, why = _pl(watts, allow_prompt)
    if not ok:
        return False, f"limite de puissance non appliquee : {why}"
    return True, f"limite de puissance {watts} W appliquee"


def apply_power_now():
    watts = int(load_settings().get("power_limit_w") or 0)
    if watts <= 0:
        return "Saisissez d'abord une limite en watts, puis Enregistrer."
    ok, text = set_power_limit(watts, allow_prompt=True)
    if ok and RUN.running():
        ORCH._power_changed = True  # changee pendant un job : retablie a sa fin
    ORCH.power_message = text[0].upper() + text[1:] + "."
    return ORCH.power_message


def reset_power_now():
    info = power_info()
    if not info:
        return "Limites de puissance non lisibles sur cette carte."
    ok, why = _pl(int(info["default"]), allow_prompt=True)
    ORCH.power_message = ""
    return (f"Limite retablie a {info['default']:.0f} W." if ok else
            f"Retablissement impossible : {why}.")


def restore_power_limit():
    info = power_info()
    if info and info["current"] is not None and abs(info["current"] - info["default"]) > 1:
        nvidia_smi("-pl", str(int(info["default"])))


class SysMon:
    """Interroge nvidia-smi toutes les 2 s ; garde le pic de VRAM de la phase
    d'entrainement (hors chargement du modele et encodeur de texte)."""

    def __init__(self):
        self.vram_used = self.vram_total = self.gpu_util = self.gpu_temp = None
        self.power_draw = self.power_limit = self.fan = None
        self.ram_used = self.ram_total = None
        self.peak_all = 0.0
        self.peak_train = 0.0
        self.in_train_phase = lambda: False
        self._started = False
        # energie du job en cours : integration de power.draw ; moyenne glissante pour la projection
        self.is_busy = lambda: False
        self.job_wh = 0.0
        self.avg_w = None
        self._t_last = None

    def reset_energy(self):
        self.job_wh, self.avg_w, self._t_last = 0.0, None, None

    def reset_peaks(self):
        self.peak_all = self.peak_train = 0.0

    def start(self):
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._loop, daemon=True).start()

    FIELDS = "memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed"

    def _query(self):
        try:
            out = nvidia_smi("--query-gpu=" + self.FIELDS, "--format=csv,noheader,nounits")[1]
            vals = [_num(x) for x in out.strip().splitlines()[0].split(",")]
            used, total, util, temp, draw, limit, fan = (vals + [None] * 7)[:7]
            self.vram_used = used / 1024 if used is not None else None
            self.vram_total = total / 1024 if total is not None else None
            self.gpu_util = int(util) if util is not None else None
            self.gpu_temp = int(temp) if temp is not None else None
            self.power_draw, self.power_limit = draw, limit
            self.fan = int(fan) if fan is not None else None
        except Exception:
            self.vram_used = None

    def _loop(self):
        while True:
            self._query()
            self.ram_used, self.ram_total = _ram_gb()
            now = time.time()
            if self.is_busy() and self.power_draw:
                if self._t_last is not None:
                    dt = min(10.0, now - self._t_last)
                    self.job_wh += self.power_draw * dt / 3600
                    a = min(1.0, dt / 300)   # moyenne sur ~5 min : suit le regime de l'entrainement
                    self.avg_w = self.power_draw if self.avg_w is None else self.avg_w + a * (self.power_draw - self.avg_w)
                self._t_last = now
            else:
                self._t_last = None
            try:
                PERF.tick()
            except Exception:  # noqa: BLE001 - la mesure ne doit jamais gener le moniteur
                pass
            if self.vram_used is not None:
                self.peak_all = max(self.peak_all, self.vram_used)
                if self.in_train_phase():
                    self.peak_train = max(self.peak_train, self.vram_used)
            time.sleep(2)


# =========================================================== anti-veille
def _keep_awake_loop(is_busy):
    """Tant qu'un job tourne, signale a Windows que le systeme est utilise :
    le PC ne se met pas en veille (l'ecran peut s'eteindre normalement)."""
    if not IS_WIN:
        return
    import ctypes

    ES_SYSTEM_REQUIRED = 0x00000001
    while True:
        try:
            if is_busy() and load_settings().get("keep_awake", True):
                ctypes.windll.kernel32.SetThreadExecutionState(ES_SYSTEM_REQUIRED)
        except Exception:
            pass
        time.sleep(30)


# =========================================================== notifications
def notify(title, text):
    try:  # notifications dans la langue de l'interface
        import h3_i18n
        title, text = h3_i18n.tr(title), h3_i18n.tr(text)
    except Exception:
        pass
    cfg = load_settings()
    if cfg.get("sound") and IS_WIN:
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass
    topic = (cfg.get("ntfy_topic") or "").strip()
    if topic:
        def _send():
            try:
                req = urllib.request.Request(
                    "https://ntfy.sh/" + urllib.request.quote(topic, safe=""),
                    data=text.encode("utf-8"),
                    headers={"Title": title.encode("ascii", "replace").decode()})
                urllib.request.urlopen(req, timeout=10).read()
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()


# =========================================================== courbe de perte
class LossLog:
    """Perte par pas, lue dans la barre de progression d'ai-toolkit et
    conservee dans <job>/pertes.csv (survit aux pauses et reprises)."""

    def __init__(self):
        self.path = None
        self.last_step = -1

    def open(self, job_dir):
        self.path = os.path.join(job_dir, "pertes.csv")
        self.last_step = -1
        for step, _ in self.read(self.path):
            self.last_step = max(self.last_step, step)

    def add(self, step, loss):
        if self.path is None or step <= self.last_step:
            return
        self.last_step = step
        try:
            with open(self.path, "a", newline="") as fh:
                csv.writer(fh).writerow([step, f"{loss:.6g}"])
        except OSError:
            pass

    @staticmethod
    def read(path):
        rows = {}
        try:
            with open(path, newline="") as fh:
                for row in csv.reader(fh):
                    try:
                        rows[int(row[0])] = float(row[1])
                    except (ValueError, IndexError):
                        pass
        except OSError:
            pass
        return sorted(rows.items())


def loss_points(job, alpha=0.04):
    """Liste de dicts {pas, perte, serie} : valeurs brutes + moyenne lissee."""
    rows = LossLog.read(os.path.join(OUTPUT, job, "pertes.csv"))
    if len(rows) > 1500:  # allege l'affichage des longs entrainements
        k = len(rows) // 1500 + 1
        rows = rows[::k]
    out, ema = [], None
    for step, loss in rows:
        ema = loss if ema is None else ema + alpha * (loss - ema)
        out.append({"pas": step, "perte": loss, "serie": "brute"})
        out.append({"pas": step, "perte": ema, "serie": "lissee"})
    return out


# =========================================================== processus
# etapes affichees dans l'onglet de chaque type de job
STAGES = {
    "caption": ["Lancement", "Chargement du modele", "Quantification", "Legendage"],
    "download": ["Lancement", "Telechargement"],
    "train": ["Lancement", "Chargement du modele", "Preparation du dataset", "Entrainement"],
    "calib": ["Lancement", "Chargement du modele", "Preparation du dataset", "Mesure (20 pas)"],
    "test": ["Lancement", "Chargement du modele", "Generation des images"],
}
# ce qu'on sait de la duree de l'etape en cours, affiche tant qu'aucune barre ne la mesure
STAGE_HINTS = {
    "Lancement": "demarrage de Python et import de PyTorch : 20 s a 1 min",
    "Chargement du modele": ("lecture des poids : 1 a 3 min ; au premier usage, telechargement "
                             "du modele avant (barre de progression ci-dessous)"),
    "Quantification": "compression du modele en int8 a chaque lancement : 1 a 3 min",
    "Preparation du dataset": "mise en cache des images et des legendes",
}


class Runner:
    """Un sous-processus a la fois ; stdout/stderr fusionnes lus en direct.

    tqdm redessine sa barre avec '\\r' : un '\\r' remplace la ligne courante au
    lieu d'en ajouter une, le journal reste lisible et la barre analysable."""

    BAR = re.compile(
        r"(?P<desc>[^|\r\n]*?):?\s*(?P<pct>\d{1,3})%\|[^|]*\|\s*"
        r"(?P<n>[\d.]+\s*[kMGT]?i?B?)/(?P<tot>[\d.]+\s*[kMGT]?i?B?)\s*"
        r"\[(?P<el>[^<\]]*)<?(?P<eta>[^,\]]*),?\s*(?P<rate>[^\],]*)"
    )
    LOSS = re.compile(r"\bloss:\s*([0-9.]+(?:e[+-]?\d+)?)", re.I)
    CONTROLLED = ("train", "calib", "test")  # jobs pilotes par la base de controle

    def __init__(self):
        self.proc = None
        self.kind = self.label = self.job_name = self.job_dir = ""
        self.lines = collections.deque(maxlen=400)
        self.partial = ""
        self.bar = self.train_bar = None
        self.aside = 0
        self.phase = "En attente"
        self.started = None
        self.exit_code = None
        self.pausing = False
        self.stop_phase = None
        self.saw_oom = self.saw_compile_error = False
        self.log_file = None
        self.loss = LossLog()
        self.stage = ""
        self.stage_started = None
        self.items_total = 0
        self.history = {}  # type de job -> etat final du dernier job de ce type
        self.on_exit = []  # callbacks(runner)
        self.lock = threading.Lock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd, kind, label, log_path=None, job_name="", job_dir="", extra_env=None):
        if self.running():
            return False
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        env.pop("AITK_JOB_ID", None)
        if kind in self.CONTROLLED:
            env["AITK_JOB_ID"] = job_name
        env.update(extra_env or {})
        MON.reset_energy()
        with self.lock:
            self.lines.clear()
            self.partial = ""
            self.bar = self.train_bar = None
            self.phase = "Lancement : " + STAGE_HINTS["Lancement"]
            self.stage, self.stage_started, self.items_total = "Lancement", time.time(), 0
            self.kind, self.label = kind, label
            self.job_name, self.job_dir = job_name, job_dir
            self.started, self.exit_code = time.time(), None
            self.pausing, self.stop_phase = False, None
            self.saw_oom = self.saw_compile_error = False
            if self.log_file:
                self.log_file.close()
            self.log_file = open(log_path, "a", encoding="utf-8") if log_path else None
            if kind == "train" and job_dir:
                self.loss.open(job_dir)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0
        self.proc = subprocess.Popen(
            cmd, cwd=AITK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags)
        threading.Thread(target=self._read, daemon=True).start()
        return True

    def _push_line(self, text, replace):
        if replace and self.lines:
            self.lines[-1] = text
        else:
            self.lines.append(text)
        self._parse(text)

    def _read(self):
        stream = self.proc.stdout
        # decodeur incremental : un caractere UTF-8 coupe entre deux blocs reste intact
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        replace_next = False
        while True:
            chunk = stream.read1(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            with self.lock:
                if self.log_file:
                    self.log_file.write(text.replace("\r", "\n"))
                    self.log_file.flush()
                for ch in re.split(r"(\r\n|\r|\n)", text):
                    if ch in ("\n", "\r\n"):
                        # "\r" puis "\n" (CRLF coupe entre deux blocs, ou fin d'une
                        # barre tqdm) : la ligne est deja affichee, ne pas l'effacer
                        if self.partial or not replace_next:
                            self._push_line(self.partial, replace_next)
                        self.partial, replace_next = "", False
                    elif ch == "\r":
                        if self.partial:
                            self._push_line(self.partial, replace_next)
                            replace_next = True
                        self.partial = ""
                    else:
                        self.partial += ch
        self.proc.wait()
        with self.lock:
            if self.partial:
                self._push_line(self.partial, replace_next)
            self.exit_code = self.proc.returncode
            if self.stop_phase:
                # arret / pause demandes : le code de sortie non nul est normal
                self.phase = self.stop_phase
            elif self.exit_code == 0:
                self.phase = "Termine"
            elif not self.phase.startswith(("ECHEC", "MEMOIRE")):
                self.phase = f"Erreur (code {self.exit_code}) - voir le journal"
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            lines = list(self.lines)
            self.history[self.kind] = {
                "label": self.label, "phase": self.phase, "exit_code": self.exit_code,
                "ended": time.time(), "duration": time.time() - (self.started or time.time()),
                "log": "\n".join(lines[-120:]), "stopped": bool(self.stop_phase),
                "wh": MON.job_wh if self.kind != "download" else 0}
        for callback in list(self.on_exit):
            try:
                callback(self)
            except Exception as exc:  # un callback ne doit jamais bloquer la suite
                print("callback de fin en erreur :", exc)

    def _set_stage(self, stage):
        if stage != self.stage:
            self.stage, self.stage_started = stage, time.time()
            self.bar = None   # la barre de l'etape precedente n'a plus de sens
            hint = STAGE_HINTS.get(stage)
            self.phase = stage + (" : " + hint if hint else "")

    ASIDE = ("generating samples", "saving at step", "saved checkpoint", "unloading assistant",
             "loading assistant")

    def _parse(self, line):
        low = line.lower()
        self._parse_stage(line, low)
        if any(k in low for k in self.ASIDE):
            self.aside += 1   # apercu ou sauvegarde : ces pas ne servent pas a mesurer la vitesse
        m = self.BAR.search(line)
        if m:
            d = {k: (v or "").strip() for k, v in m.groupdict().items()}
            is_bytes = "b" in d["tot"].lower() or "b/s" in d["rate"].lower()
            # la barre d'entrainement d'ai-toolkit a pour description le nom du job
            if self.kind in self.CONTROLLED and not is_bytes and d["desc"] == self.job_name:
                self.train_bar = d
                self.bar = None  # l'etape annexe (apercus, cache...) est terminee
                self.phase = "Entrainement"
                if self.kind == "train":
                    lm = self.LOSS.search(line)
                    if lm and d["n"].isdigit():
                        try:
                            self.loss.add(int(d["n"]), float(lm.group(1)))
                        except ValueError:
                            pass
            else:
                self.bar = d
                if is_bytes or ".safetensors" in low:
                    self.phase = "Telechargement des modeles (reprise automatique si coupure)"
                elif "latent" in low:
                    self.phase = "Mise en cache des latents"
                elif "embedding" in low or "text" in low:
                    self.phase = "Mise en cache des embeddings de texte"
                elif self.kind == "caption":
                    self.phase = ("Legendage des images" if "captioning" in low else
                                  "Chargement des poids du modele" if "loading" in low else self.phase)
        elif "loading" in low and ("model" in low or "transformer" in low) and self.kind != "caption":
            self.phase = "Chargement du modele"
        elif "generating images" in low or "sampling" in low or "generating baseline" in low:
            self.phase = "Generation des apercus"
        elif "saving at step" in low or ("saved" in low and "safetensors" in low):
            self.phase = "Sauvegarde du checkpoint"
        elif "compilationerror" in low or ("triton" in low and "error" in low):
            self.saw_compile_error = True
            self.phase = ("ECHEC COMPILATION TRITON - preset w4a8 : cochez le correctif w4a8 "
                          "ou repassez au preset int8")
        elif "out of memory" in low:
            self.saw_oom = True
            self.phase = "MEMOIRE GPU INSUFFISANTE - voir l'aide (erreur memoire)"

    def _parse_stage(self, line, low):
        """Etape en cours, d'apres les messages connus d'ai-toolkit et de huggingface_hub."""
        kind = self.kind
        is_download = ("b/s" in low or "fetching" in low or "downloading" in low) and "%|" in line
        if kind == "caption":
            if "captioning files" in low:
                self._set_stage("Legendage")
            elif "quantiz" in low:
                self._set_stage("Quantification")
            elif ("loading" in low or is_download) and self.stage == "Lancement":
                self._set_stage("Chargement du modele")
            m = re.search(r"found (\d+) files to caption", low)
            if m:
                self.items_total = int(m.group(1))
                self._set_stage("Legendage")
            if "captioning complete" in low:
                self.phase = "Legendage termine"
        elif kind == "download":
            if is_download or ".safetensors" in low:
                self._set_stage("Telechargement")
        elif kind in ("train", "calib", "test"):
            if "loading" in low and ("model" in low or "transformer" in low):
                self._set_stage("Chargement du modele")
            elif "caching" in low or "bucket" in low or "preprocessing image" in low:
                self._set_stage("Preparation du dataset")
            elif kind == "test" and ("generating" in low or "sampling" in low):
                self._set_stage("Generation des images")
            elif self.job_name and line.lstrip().startswith(self.job_name + ":") and "%|" in line:
                self._set_stage(STAGES[kind][-1])

    def view(self, kinds):
        """Etat a afficher dans l'onglet d'un type de job : le job en cours s'il est de
        ce type, sinon le dernier termine de ce type, sinon None."""
        with self.lock:
            if self.running() and self.kind in kinds:
                lines = list(self.lines)
                if self.partial:
                    lines.append(self.partial)
                return {"live": True, "kind": self.kind, "label": self.label, "phase": self.phase,
                        "stage": self.stage, "stages": STAGES.get(self.kind, []),
                        "stage_elapsed": time.time() - (self.stage_started or time.time()),
                        "elapsed": time.time() - (self.started or time.time()),
                        "bar": self.bar, "train_bar": self.train_bar, "total": self.items_total,
                        "pausing": self.pausing, "log": "\n".join(lines[-150:])}
            done = [(h["ended"], k, h) for k, h in self.history.items() if k in kinds]
            if not done:
                return None
            _, kind, h = max(done)
            return dict(h, live=False, kind=kind)

    def kill(self):
        if not self.running():
            return
        if IS_WIN:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            self.proc.terminate()

    def stop(self, phase="Arrete par l'utilisateur", grace=20):
        if not self.running():
            return
        self.phase = self.stop_phase = phase
        # arret propre d'abord : ai-toolkit lit stop=1 et sort de lui-meme
        if self.kind in self.CONTROLLED and self.job_name and db_set(self.job_name, "stop", 1):
            deadline = time.time() + grace
            while self.running() and time.time() < deadline:
                time.sleep(0.5)
        self.kill()

    def snapshot(self):
        with self.lock:
            lines = list(self.lines)
            if self.partial:
                lines.append(self.partial)
            return {
                "running": self.running(), "kind": self.kind, "label": self.label,
                "phase": self.phase, "bar": self.bar, "train_bar": self.train_bar,
                "elapsed": time.time() - self.started if self.started else 0,
                "log": "\n".join(lines[-120:]),
            }


RUN = Runner()
MON = SysMon()


class PerfLog:
    """Puissance, temperature et vitesse pendant l'entrainement, par intervalles d'au moins 5 pas
    sans apercu ni sauvegarde : output/<job>/gpu_perf.csv
    (heure, pas, puissance W, temperature C, limite W, s/pas). Sert a trouver la limite de puissance
    la plus basse qui ne ralentit pas l'entrainement."""
    MIN_STEPS = 5
    HEADER = "heure,pas,puissance_w,temperature_c,limite_w,s_par_pas\n"

    def __init__(self):
        self._reset(None, None)

    def _reset(self, job, step):
        self.job, self.step0, self.t0, self.acc, self.aside0 = job, step, None, [], RUN.aside

    def tick(self):
        bar = RUN.train_bar if RUN.running() and RUN.kind == "train" else None
        try:
            step = int(bar["n"]) if bar else None
        except (KeyError, TypeError, ValueError):
            step = None
        if step is None:
            self._reset(None, None)
            return
        now = time.time()
        if RUN.job_name != self.job or RUN.aside != self.aside0:
            self._reset(RUN.job_name, step)   # nouvel intervalle des que les pas reprennent
            return
        if self.t0 is None:
            if step != self.step0:
                self.t0, self.step0, self.acc = now, step, []
            return
        if MON.power_draw:
            self.acc.append((MON.power_draw, MON.gpu_temp, MON.power_limit))
        if step - self.step0 >= self.MIN_STEPS and self.acc:
            n = len(self.acc)
            watts = sum(a[0] for a in self.acc) / n
            temps = [a[1] for a in self.acc if a[1] is not None]
            limit = self.acc[-1][2]
            row = (f"{now:.0f},{step},{watts:.1f},{(sum(temps) / len(temps)) if temps else 0:.1f},"
                   f"{limit or 0:.0f},{(now - self.t0) / (step - self.step0):.3f}\n")
            path = os.path.join(OUTPUT, self.job, "gpu_perf.csv")
            try:
                new = not os.path.isfile(path)
                with open(path, "a", encoding="utf-8") as f:
                    f.write((self.HEADER if new else "") + row)
            except OSError:
                pass
            self.t0, self.step0, self.acc = now, step, []

    @staticmethod
    def read(job):
        rows = []
        try:
            with open(os.path.join(OUTPUT, job, "gpu_perf.csv"), encoding="utf-8") as f:
                for line in f.read().splitlines()[1:]:
                    p = line.split(",")
                    if len(p) == 6:
                        rows.append((float(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
        except (OSError, ValueError):
            pass
        return rows


PERF = PerfLog()


class PowerSweep:
    """Balayage de puissance au debut d'un entrainement : la limite descend par paliers
    (SWEEP_STEPS pas chacun), puis le seuil conseille (le palier le plus bas a moins de 3 % de la
    vitesse maximale, mesuree par PerfLog) est applique et enregistre comme limite de puissance.
    Une fois par job (output/<job>/balayage.json) ; necessite les droits administrateur."""
    STEPS = 30          # pas par palier
    WARMUP = 10         # pas avant le premier palier (fin des premiers apercus, cache chaud)
    TOLERANCE = 0.03

    def __init__(self):
        self.note = ""
        self._reset(None)

    def _reset(self, job):
        self.job, self.levels, self.idx, self.step0, self.t_start, self.done = job, [], -1, None, None, False

    def running(self):
        return bool(self.levels) and not self.done

    def _file(self, job):
        return os.path.join(OUTPUT, job, "balayage.json")

    def tick(self):
        cfg = load_settings()
        bar = RUN.train_bar if RUN.running() and RUN.kind == "train" else None
        if not cfg.get("sweep_on") or bar is None or not str(bar.get("n", "")).isdigit():
            if self.running():
                self.note = ""   # job arrete ou mis en pause : le balayage reprendra du debut
            if not RUN.running():
                self._reset(None)
            return
        step, job = int(bar["n"]), RUN.job_name
        if job != self.job:
            self._reset(job)
            if os.path.isfile(self._file(job)):
                try:
                    with open(self._file(job), encoding="utf-8") as f:
                        pick = json.load(f).get("pick")
                    self.note = f"Balayage de puissance deja fait pour ce job : {pick} W."
                except (OSError, ValueError):
                    self.note = ""
                self.done = True
                return
            if not is_admin():
                self.note = ("Balayage de puissance inactif : droits administrateur requis "
                             "(lancez 2_LANCER_UI_ADMIN.bat).")
                self.done = True
                return
            info = power_info()
            if not info:
                self.note = "Balayage de puissance inactif : limite de puissance illisible sur cette carte."
                self.done = True
                return
            top = int(cfg.get("power_limit_w") or 0) or int(info["default"])
            top = int(max(info["min"], min(info["max"], top)))
            floor = int(max(info["min"], min(top, int(cfg.get("sweep_min_w") or 200))))
            stepw = max(10, int(cfg.get("sweep_step_w") or 30))
            self.levels = list(range(top, floor - 1, -stepw))
            if len(self.levels) < 2:
                self.note = "Balayage de puissance : plage trop etroite (plancher proche de la limite)."
                self.done = True
                return
            self.start_at = step + self.WARMUP
            self.note = f"Balayage de puissance : debut au pas {self.start_at}."
            return
        if self.done:
            return
        if self.idx < 0:
            if step >= self.start_at:
                self.t_start = time.time()
                self._set(0, step)
            return
        if step - self.step0 >= self.STEPS:
            if self.idx + 1 < len(self.levels):
                self._set(self.idx + 1, step)
            else:
                self._finish(job)

    def _set(self, i, step):
        ok, msg = set_power_limit(self.levels[i])
        if not ok:
            self.note = "Balayage de puissance interrompu : " + msg + "."
            self.done = True
            return
        ORCH._power_changed = True
        self.idx, self.step0 = i, step
        self.note = (f"Balayage de puissance : palier {i + 1}/{len(self.levels)} a {self.levels[i]} W "
                     f"({self.STEPS} pas par palier).")

    def _finish(self, job):
        rows = [r for r in PerfLog.read(job) if self.t_start and r[0] >= self.t_start]
        by_level = {}
        for r in rows:
            by_level.setdefault(int(round(r[4])), []).append(r[5])
        stats = {}
        for lvl in self.levels:
            vals = by_level.get(lvl, [])[1:]          # premier intervalle : transition entre paliers
            if len(vals) >= 2:
                vals = sorted(vals)
                stats[lvl] = vals[len(vals) // 2]
        self.done = True
        if len(stats) < 2:
            self.note = "Balayage de puissance : pas assez de mesures pour conclure (apercus trop frequents ?)."
            return
        best = min(stats.values())
        pick = min(lvl for lvl, v in stats.items() if v <= best * (1 + self.TOLERANCE))
        set_power_limit(pick)
        save_settings(power_limit_w=pick)
        try:
            with open(self._file(job), "w", encoding="utf-8") as f:
                json.dump({"levels": stats, "pick": pick, "time": int(time.time())}, f)
        except OSError:
            pass
        slower = (stats[pick] / best - 1) * 100
        self.note = (f"Balayage de puissance termine : {pick} W retenus ({slower:.0f} % plus lent que le plus "
                     f"rapide, {max(stats) - pick} W de moins). Enregistre comme limite de puissance.")
        notify("Balayage de puissance termine", self.note)


SWEEP = PowerSweep()
MON.in_train_phase = lambda: RUN.running() and RUN.train_bar is not None
MON.is_busy = lambda: RUN.running() and RUN.kind != "download"


def rate_to_seconds(rate):
    """'2.50s/it' -> 2.5 ; '4.00it/s' -> 0.25 ; None si illisible."""
    m = re.match(r"\s*([\d.]+)\s*(s/it|it/s)", rate or "")
    if not m or float(m.group(1)) <= 0:
        return None
    value = float(m.group(1))
    return value if m.group(2) == "s/it" else 1.0 / value


# =========================================================== checkpoints
def safetensors_ok(path):
    """Verifie l'en-tete et la taille : un checkpoint tronque (arret brutal
    pendant l'ecriture, ai-toolkit n'ecrit pas de facon atomique) est detecte."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if n <= 0 or n > 100_000_000 or 8 + n > size:
                return False
            header = json.loads(fh.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
                  default=0)
        return size == 8 + n + end
    except Exception:
        return False


def quarantine_bad_checkpoints(job_dir):
    bad = []
    for p in glob.glob(os.path.join(job_dir, "*.safetensors")):
        if not safetensors_ok(p):
            os.replace(p, p + ".corrompu")
            bad.append(os.path.basename(p))
    return bad


def list_jobs(include_internal=False):
    names = sorted(d for d in os.listdir(OUTPUT) if os.path.isdir(os.path.join(OUTPUT, d)))
    return names if include_internal else [n for n in names if not n.startswith("_")]


def job_checkpoints(job):
    files = glob.glob(os.path.join(OUTPUT, job, "*.safetensors"))
    return sorted(files, key=os.path.getmtime, reverse=True)


def job_samples(job, limit=24):
    files = glob.glob(os.path.join(OUTPUT, job, "samples", "*.*"))
    files = [f for f in files if f.lower().endswith(IMG_EXT)]
    return sorted(files, key=os.path.getmtime, reverse=True)[:limit]


# =========================================================== dataset
def dataset_report(folder):
    if not folder or not os.path.isdir(folder):
        return "Dossier introuvable.", []
    imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
    missing = [f for f in imgs
               if not os.path.isfile(os.path.join(folder, os.path.splitext(f)[0] + ".txt"))]
    msg = f"{len(imgs)} image(s), {len(imgs) - len(missing)} legende(s) .txt."
    if missing:
        msg += (f"\nSans legende ({len(missing)}) : " + ", ".join(missing[:15])
                + (" ..." if len(missing) > 15 else ""))
        msg += "\nLe modele lit le prompt tel quel : utilisez le legendage automatique ou ecrivez-les."
    if len(imgs) < 10:
        msg += "\nConseil : 15 a 40 images variees donnent en general de meilleurs resultats."
    return msg, [os.path.join(folder, f) for f in imgs[:60]]


def image_size_report(folder, resolutions):
    """Controle des tailles : ai-toolkit reduit les images a la surface de la
    resolution choisie mais ne les agrandit jamais. Une image plus petite est
    donc entrainee a sa taille d'origine, quelle que soit la resolution cochee."""
    from PIL import Image

    if not folder or not os.path.isdir(folder):
        return ""
    sizes = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(IMG_EXT):
            try:
                with Image.open(os.path.join(folder, f)) as im:
                    sizes.append((f, im.width, im.height))
            except Exception:
                pass
    if not sizes:
        return ""
    res = sorted(int(r) for r in (resolutions or [])) or [1024]
    lines = ["", "Tailles des images :"]
    small = [(f, w, h) for f, w, h in sizes if w * h < 500_000]
    if small:
        lines.append(f"- {len(small)} image(s) de moins de 0,5 megapixel (environ 700x700) : trop peu "
                     "de detail pour ce modele. "
                     "Retirez-les ou remplacez-les par une source plus grande :")
        lines.append("  " + ", ".join(f"{f} ({w}x{h})" for f, w, h in small[:12])
                     + (" ..." if len(small) > 12 else ""))
    else:
        lines.append("- Aucune image de moins de 0,5 megapixel.")
    prev_area = 0
    for r in res:
        area = r * r
        full = sum(1 for _, w, h in sizes if w * h >= 0.95 * area)
        part = sum(1 for _, w, h in sizes if prev_area * 1.05 < w * h < 0.95 * area) if prev_area else 0
        txt = f"- Resolution {r} : {full}/{len(sizes)} image(s) assez grandes"
        if prev_area:
            txt += f", {part} en profitent en partie, {len(sizes) - full - part} pas du tout"
        lines.append(txt + ".")
        prev_area = area
    top = res[-1]
    useful = sum(1 for _, w, h in sizes
                 if w * h > (res[-2] ** 2 * 1.05 if len(res) > 1 else 0.95 * top * top))
    if len(res) > 1 and useful < len(sizes) / 2:
        lines.append(f"- Moins de la moitie du dataset profite de la resolution {top} : "
                     "elle rallonge l'entrainement pour peu de gain. Decochez-la, ou trouvez "
                     "des images plus grandes.")
    elif len(res) == 1 and useful < len(sizes) / 2:
        lines.append(f"- Plus de la moitie des images sont plus petites que la resolution {top} : "
                     "le LoRA sera entraine en dessous de cette resolution.")
    return "\n".join(lines)


def checkpoint_steps(job):
    """Numeros de pas des checkpoints presents (le checkpoint final n'a pas de numero)."""
    steps = []
    for p in job_checkpoints(job):
        m = re.search(r"_(\d{9})\.safetensors$", os.path.basename(p))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def fill_captions(folder, text):
    n = 0
    for f in os.listdir(folder):
        if f.lower().endswith(IMG_EXT):
            p = os.path.join(folder, os.path.splitext(f)[0] + ".txt")
            if not os.path.isfile(p):
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(text.strip())
                n += 1
    return n


# =========================================================== configuration
def build_config(P):
    """Parametres de l'UI -> (nom du job, config ai-toolkit en dict)."""
    name = safe_name(P.get("name"))
    preset = P.get("preset", PRESET_INT8)
    offload = float(P.get("offload") or 0)
    model = {
        "name_or_path": H3_REPO,
        "arch": "minimax_h3_image",
        "quantize": True,
        "qtype": "convrotw4a8" if preset == PRESET_W4A8 else "convrot8",
        "quantize_te": True,
        "qtype_te": "nvfp4",
        "low_vram": True,
        "assistant_lora_path": (FIZGIG_ADAPTER_PATH if P.get("adapter") == ADAPTER_FIZGIG
                                else ADAPTER),
    }
    optimizer = OPTIMIZERS.get(P.get("optimizer"), "adamw8bit")
    lr = float(P.get("lr") or 1e-4)
    # automagic regle lui-meme son taux : le champ est son point de depart, et
    # max_lr son plafond (2,5e-4, le reglage teste par Fizgig sur ce modele)
    opt_params = {"weight_decay": 1e-4} if optimizer == "adamw8bit" else {"max_lr": max(lr, 2.5e-4)}
    ema = float(EMA_CHOICES.get(P.get("ema"), 0.0))
    if preset == PRESET_INT8 or offload > 0:
        model["layer_offloading"] = True
        model["layer_offloading_transformer_percent"] = round(offload, 2)
        model["layer_offloading_text_encoder_percent"] = 1.0
    ds = {
        "folder_path": P.get("folder", ""),
        "caption_ext": "txt",
        "cache_latents_to_disk": True,
        "resolution": sorted(int(r) for r in (P.get("resolutions") or [])) or [1024],
    }
    if (P.get("ctrl") or "").strip():
        ds["multi_control_paths"] = [P["ctrl"].strip()]
    prompts = [p.strip() for p in (P.get("prompts") or "").splitlines() if p.strip()]
    if P.get("control", True):
        ctl, auto = control_prompt(prompts, P.get("trigger"))
        if auto:
            prompts.append(ctl)   # prompt temoin : mesure du debordement (onglet Resultats > Ressemblance)
    size = int(P.get("sample_size") or 2048)
    proc = {
        "type": "diffusion_trainer",
        "sqlite_db_path": DB_PATH,
        "training_folder": OUTPUT,
        "device": "cuda:0",
        "network": {
            "type": "lora", "linear": int(P.get("rank") or 16), "linear_alpha": int(P.get("rank") or 16),
            "network_kwargs": {"ignore_if_contains": ["adaln_proj"]},
        },
        "save": {"dtype": "bf16", "save_every": int(P.get("save_every") or 250),
                 "max_step_saves_to_keep": 10},
        "datasets": [ds],
        "train": {
            "batch_size": 1, "steps": int(P.get("steps") or 2000), "gradient_accumulation": 1,
            "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
            "cache_text_embeddings": True, "noise_scheduler": "flowmatch",
            "timestep_type": "shift", "optimizer": optimizer, "lr": lr,
            "optimizer_params": opt_params, "dtype": "bf16",
            "disable_sampling": bool(P.get("no_sample")),
        },
        "model": model,
        "sample": {
            "sampler": "flowmatch", "sample_every": int(P.get("sample_every") or 250),
            "width": size, "height": size, "prompts": prompts or ["a detailed photo"],
            "neg": "", "seed": 42, "walk_seed": True, "guidance_scale": 1, "sample_steps": 6,
        },
    }
    if ema > 0:
        proc["train"]["ema_config"] = {"use_ema": True, "ema_decay": ema}
    if (P.get("trigger") or "").strip():
        proc["trigger_word"] = P["trigger"].strip()
        if P.get("dop"):
            # preservation (DOP, ai-toolkit) : a chaque pas, le LoRA ne doit pas changer ce que le modele
            # de base produit pour la legende ou le mot declencheur est remplace par la classe
            proc["train"].update({"diff_output_preservation": True,
                                  "diff_output_preservation_class": (P.get("dop_class") or CONTROL_SUBJECT).strip(),
                                  "diff_output_preservation_multiplier": 1.0})
    reg_dir = reg_folder(P.get("folder"))
    if P.get("reg") and count_images(reg_dir):
        # images de regularisation (generees par le modele de base) : pas de mot declencheur ajoute
        proc["datasets"].append({"folder_path": reg_dir, "caption_ext": "txt", "cache_latents_to_disk": True,
                                 "is_reg": True, "resolution": ds["resolution"]})
        proc["train"]["reg_weight"] = float(P.get("reg_weight") or 1.0)
    cfg = {"job": "extension", "config": {"name": name, "process": [proc]},
           "meta": {"name": "[name]", "version": "1.0"}}
    return name, cfg


def dump_yaml(cfg):
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def write_config(name, cfg):
    path = os.path.join(CONFIGS, f"{name}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dump_yaml(cfg))
    return path


def _job_env(cfg, P=None, start_step=0):
    """Variables d'environnement du job, lues par run_job.py : correctif w4a8,
    adaptateur Fizgig, optimisation de ressemblance, apercus LCM."""
    proc = cfg["config"]["process"][0]
    qtype = proc.get("model", {}).get("qtype", "")
    want_fix = True if P is None else bool(P.get("w4a8_fix", True))
    env = {"H3UI_W4A8_FIX": "1"} if "w4a8" in qtype and want_fix else {}
    if proc.get("model", {}).get("assistant_lora_path") == FIZGIG_ADAPTER_PATH:
        env["H3UI_FIZGIG_ADAPTER"] = FIZGIG_ADAPTER_PATH
    P = P or {}
    if P.get("likeness"):
        folder = (proc.get("datasets") or [{}])[0].get("folder_path", "")
        env.update(H3UI_LIKENESS="1", H3UI_EPOCH_STEPS=str(max(1, count_images(folder))),
                   H3UI_START_STEP=str(int(start_step or 0)))
    if P.get("sampler") == SAMPLER_LCM:
        env["H3UI_LCM"] = "1"
    return env


def _prompt_lines(prompts):
    if isinstance(prompts, (list, tuple)):
        return [str(x).strip() for x in prompts if str(x).strip()]
    return [x.strip() for x in str(prompts or "").splitlines() if x.strip()]


def prompt_trigger_issues(trigger, prompts):
    """Ecarts entre le mot declencheur et les prompts : liste de messages (vide = tout va bien).
    - aucun prompt ne contient le mot declencheur ;
    - un prompt contient un mot proche (ancien nom, faute de frappe : Maine_1 pour Maine_2)."""
    trigger = (trigger or "").strip()
    lines = _prompt_lines(prompts)
    if not trigger or not lines:
        return []
    import difflib

    trig = trigger.lower()
    stem = re.sub(r"[\d_\-]+$", "", trig)
    found, near, plain = False, [], []
    for line in lines:
        for word in re.findall(r"[\w\-]+", line):
            tok = word.lower()
            if tok == trig:
                found = True
            elif len(tok) >= 4 and word not in near and (
                    (stem and len(stem) >= 3 and re.sub(r"[\d_\-]+$", "", tok) == stem)
                    or (re.search(r"[\d_]", tok) and difflib.SequenceMatcher(None, tok, trig).ratio() >= 0.75)):
                (near if re.search(r"[\d_\-]", tok) else plain).append(word)
    if not found:   # un mot ordinaire proche (Maine) n'est suspect que si le vrai mot est absent
        near += plain
    issues = []
    if near:
        issues.append(f"les prompts d'apercu contiennent '{', '.join(near)}', proche du mot declencheur "
                      f"'{trigger}' : ancien nom ou faute de frappe ? Le LoRA n'est appele que par son mot exact.")
    if not found:
        issues.append(f"aucun prompt d'apercu ne contient le mot declencheur '{trigger}' : les apercus "
                      "ne montreront pas l'effet du LoRA appele par son mot.")
    return issues


CONTROL_SUBJECT = "a person"


def _has_word(line, word):
    return any(t.lower() == word.lower() for t in re.findall(r"[\w\-]+", line))


def control_prompt(prompts, trigger, subject=CONTROL_SUBJECT):
    """Prompt temoin sans mot declencheur, qui sert a mesurer le debordement (le personnage qui
    apparait meme sans son mot). Rend (texte, ajoute_automatiquement) ou (None, False).
    Un prompt de l'utilisateur sans le mot (et sans variante proche) sert de temoin ; sinon on derive
    le premier prompt en remplacant le mot declencheur par 'a person'."""
    trigger = (trigger or "").strip()
    lines = _prompt_lines(prompts)
    if not trigger or not lines:
        return None, False
    for line in lines:
        if not _has_word(line, trigger) and not prompt_trigger_issues(trigger, [trigger, line]):
            return line, False
    base = next((x for x in lines if _has_word(x, trigger)), lines[0])
    text = re.sub(r"(?<![\w\-])" + re.escape(trigger) + r"(?![\w\-])", subject, base, flags=re.I)
    # variantes proches du mot (ancien nom, faute de frappe) : remplacees aussi
    text = re.sub(r"[\w\-]+", lambda m: subject if (
        m.group(0).lower() != subject and prompt_trigger_issues(trigger, [trigger, m.group(0)])
        and m.group(0).lower() != trigger.lower()) else m.group(0), text)
    text = re.sub(r"\s+", " ", text.replace("[trigger]", subject)).strip()
    parts = []   # "a man, a man, wearing..." : groupe repete retire
    for part in text.split(", "):
        if not parts or part.strip().lower() != parts[-1].strip().lower():
            parts.append(part)
    return ", ".join(parts), True


def prompt_warning(P):
    if P.get("no_sample"):
        return ""
    return "".join("\nATTENTION : " + x for x in prompt_trigger_issues(P.get("trigger"), P.get("prompts")))


def ram_warning(P):
    """Sous Windows, la memoire de la carte est adossee a la RAM : si la RAM
    sature, la carte refuse des allocations meme avec de la place libre. Repere
    de Fizgig : blocs decharges + 14 Go de marge doivent tenir en RAM disponible."""
    used, total = _ram_gb()
    if not total:
        return ""
    available = total - used
    need = 20.0 * float(P.get("offload") or 0) + 14.0
    if available < need:
        return (f" ATTENTION : {available:.0f} Go de RAM disponibles pour environ {need:.0f} Go "
                "necessaires (blocs decharges + marge). Fermez les autres programmes "
                "(generateur, ComfyUI, navigateur) : sinon la carte ralentit ou refuse de la memoire.")
    return ""


# =========================================================== orchestration
BEFORE_GPU_JOB = []   # fonctions appelees avant chaque job GPU (ex. arret du generateur) -> note


class Orchestrator:
    """File d'attente, pause programmee, calibrage, notifications de fin."""

    def __init__(self):
        self.queue = []            # dicts : kind, name, cfg_path, label, env
        self.hold = False          # file suspendue apres un arret manuel
        self.pause_at_step = 0
        self.pause_at_time = None  # datetime
        self.calib_report = ""
        self.calib_params = None
        self.last_message = ""
        self.thermal_message = ""
        self.power_message = ""
        self._alert_armed = True
        self._hot_since = None
        self._power_changed = False
        self.ram_message = ""
        self._ram_alerted = False
        self._thermo_temps = []
        self._thermo_last = 0.0
        self._thermo_note = ""
        self.thermo_log = collections.deque(maxlen=12)
        self._started = False
        RUN.on_exit.append(self._job_finished)

    def start(self):
        if self._started:
            return
        self._started = True
        MON.start()
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=_keep_awake_loop, args=(RUN.running,), daemon=True).start()

    # ---- lancement
    def launch(self, item, interactive=False):
        """interactive=True : lancement par un clic (une confirmation Windows peut
        etre demandee pour la limite de puissance). False : file d'attente."""
        name, kind = item["name"], item["kind"]
        job_dir = os.path.join(OUTPUT, name) if kind in Runner.CONTROLLED else ""
        note = ""
        if job_dir:
            os.makedirs(job_dir, exist_ok=True)
            db_init_job(name)
            bad = quarantine_bad_checkpoints(job_dir)
            if bad:
                note = f" Checkpoint(s) tronque(s) ecarte(s) : {', '.join(bad)}."
        for hook in BEFORE_GPU_JOB:
            try:
                note += hook()
            except Exception as exc:  # noqa: BLE001 - jamais bloquant
                print("avant job :", exc)
        log = os.path.join(job_dir or CACHE, "journal_ui.txt")
        if kind in ("train", "calib"):
            MON.reset_peaks()
        self.power_message = ""
        watts = int(load_settings().get("power_limit_w") or 0)
        if watts > 0:
            done, text = set_power_limit(watts, allow_prompt=interactive)
            self._power_changed = self._power_changed or done
            self.power_message = text[0].upper() + text[1:] + "."
        self._hot_since, self._alert_armed, self.thermal_message = None, True, ""
        ok = RUN.start([PY, RUN_JOB, item["cfg_path"]], kind, item["label"], log,
                       job_name=name, job_dir=job_dir, extra_env=item.get("env"))
        return ok, note

    def enqueue(self, item):
        self.queue.append(item)
        return len(self.queue)

    def queue_text(self):
        if not self.queue:
            return "File d'attente vide."
        head = "File SUSPENDUE (cliquez sur Reprendre la file)\n" if self.hold else ""
        return head + "\n".join(f"{i + 1}. {q['label']}" for i, q in enumerate(self.queue))

    # ---- boucle de fond
    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception as exc:
                print("orchestrateur :", exc)
            time.sleep(1)

    def _thermal_tick(self):
        """Alerte puis pause automatique quand le GPU chauffe trop."""
        cfg = load_settings()
        temp = MON.gpu_temp
        alert, limit = int(cfg.get("temp_alert") or 0), int(cfg.get("temp_pause") or 0)
        if temp is None or not RUN.running():
            self._hot_since = None
            if not RUN.running():
                self.thermal_message = ""
            return
        if alert and temp >= alert:
            self.thermal_message = (f"GPU a {temp} °C (seuil d'alerte {alert} °C) : ameliorez le "
                                    "refroidissement ou baissez la limite de puissance.")
            if self._alert_armed:
                self._alert_armed = False
                notify("GPU chaud", f"{temp} °C pendant {RUN.label}.")
        elif alert and temp <= alert - 5:
            self._alert_armed, self.thermal_message = True, ""
        if limit and temp >= limit:
            self._hot_since = self._hot_since or time.time()
            if time.time() - self._hot_since >= TEMP_PAUSE_DELAY and not RUN.pausing \
                    and not RUN.stop_phase:
                self._hot_since = None
                reason = f"GPU a {temp} °C depuis {TEMP_PAUSE_DELAY} s (seuil {limit} °C)"
                if RUN.kind == "train":
                    pause_job(reason=reason)
                else:
                    if self.queue:
                        self.hold = True
                    threading.Thread(target=RUN.stop, daemon=True,
                                     kwargs={"phase": f"Arrete par securite : {reason}"}).start()
                self.last_message = f"Arret thermique : {reason}."
                notify("Arret thermique", self.last_message)
        else:
            self._hot_since = None

    def _thermostat_tick(self):
        """Regule la limite de puissance pour tenir la temperature cible.

        Toutes les THERMO_PERIOD secondes, compare la temperature moyenne de la
        periode a la cible : trop chaud -> baisse la limite, nettement trop froid
        -> la remonte, entre le plancher et le plafond choisis. Ne fait rien hors
        job, ni sans droits administrateur (pas de fenetre de confirmation)."""
        cfg = load_settings()
        if not cfg.get("thermo_on") or not RUN.running() or RUN.kind not in Runner.CONTROLLED:
            self._thermo_temps, self._thermo_last, self._thermo_note = [], 0.0, ""
            return
        if SWEEP.running():   # le balayage pilote la limite : le thermostat attend la fin
            self._thermo_temps, self._thermo_last = [], 0.0
            self._thermo_note = "Thermostat en attente : balayage de puissance en cours."
            return
        if MON.gpu_temp is None:
            return
        if not is_admin():
            self._thermo_note = ("Thermostat inactif : droits administrateur requis "
                                 "(lancez 2_LANCER_UI_ADMIN.bat).")
            return
        self._thermo_temps.append(MON.gpu_temp)
        now = time.time()
        if not self._thermo_last:
            self._thermo_last = now
            return
        if now - self._thermo_last < THERMO_PERIOD:
            return
        temps, self._thermo_temps, self._thermo_last = self._thermo_temps, [], now
        info = power_info()
        if not info or info["current"] is None or not temps:
            self._thermo_note = "Thermostat inactif : limite de puissance illisible sur cette carte."
            return
        target = int(cfg.get("thermo_target") or 72)
        ceiling = int(cfg.get("power_limit_w") or 0) or int(info["default"])
        ceiling = int(max(info["min"], min(info["max"], ceiling)))
        floor = int(max(info["min"], min(ceiling, int(cfg.get("thermo_min_w") or 150))))
        current = int(round(info["current"]))
        mean = sum(temps) / len(temps)
        error = mean - target
        new = current
        if error > THERMO_BAND_HIGH:
            # plus l'ecart est grand, plus la baisse est franche
            new = current - THERMO_STEP * (3 if error > 10 else 2 if error > 5 else 1)
        elif error < -THERMO_BAND_LOW and (MON.gpu_util or 0) >= 50:
            # ne remonte que si la carte travaille : inutile pendant un chargement
            new = current + THERMO_STEP
        new = max(floor, min(ceiling, new))
        state = f"cible {target} °C, moyenne {mean:.1f} °C, limite {new} W (plage {floor}-{ceiling} W)"
        if new != current:
            code, _ = nvidia_smi("-pl", str(new))
            if code != 0:
                self._thermo_note = "Thermostat : nvidia-smi a refuse le changement de limite."
                return
            self._power_changed = True
            self.thermo_log.append(f"{time.strftime('%H:%M:%S')}  {mean:.1f} °C  {current} -> {new} W")
        elif new == floor and error > THERMO_BAND_HIGH:
            state += " - PLANCHER ATTEINT : la ventilation ne suffit pas"
        self._thermo_note = "Thermostat : " + state + "."

    def thermo_text(self):
        cfg = load_settings()
        if not cfg.get("thermo_on"):
            return (SWEEP.note + "\n" if SWEEP.note else "") + "Thermostat desactive."
        head = self._thermo_note or (
            f"Thermostat actif (cible {cfg.get('thermo_target')} °C) : il regule pendant les jobs.")
        if SWEEP.note:
            head = SWEEP.note + "\n" + head
        return head + ("\n" + "\n".join(self.thermo_log) if self.thermo_log else "")

    def _ram_tick(self):
        """Alerte quand la RAM sature pendant un job (Windows adosse la VRAM a la RAM)."""
        if not RUN.running() or not MON.ram_total or MON.ram_used is None:
            self.ram_message = ""
            return
        free = MON.ram_total - MON.ram_used
        if free < max(4.0, 0.06 * MON.ram_total):
            self.ram_message = (f"RAM presque pleine ({free:.0f} Go libres) : fermez d'autres programmes, "
                                "sinon l'entrainement ralentit ou la carte refuse de la memoire.")
            if not self._ram_alerted:
                self._ram_alerted = True
                notify("RAM presque pleine", self.ram_message)
        elif free > max(8.0, 0.12 * MON.ram_total):
            self.ram_message, self._ram_alerted = "", False

    def _tick(self):
        self._thermal_tick()
        try:
            SWEEP.tick()
        except Exception as exc:  # noqa: BLE001 - jamais bloquant
            SWEEP.note = f"Balayage de puissance interrompu : {exc}"
            SWEEP.done = True
        self._thermostat_tick()
        self._ram_tick()
        if RUN.running():
            if RUN.kind == "train" and not RUN.pausing and RUN.train_bar is not None:
                step = int(RUN.train_bar["n"]) if RUN.train_bar["n"].isdigit() else 0
                due_step = self.pause_at_step and step >= self.pause_at_step
                due_time = self.pause_at_time and datetime.datetime.now() >= self.pause_at_time
                if due_step or due_time:
                    reason = f"pas {self.pause_at_step} atteint" if due_step else "heure programmee"
                    self.pause_at_step, self.pause_at_time = 0, None
                    pause_job(reason=reason)
            return
        if self.queue and not self.hold:
            item = self.queue.pop(0)
            self.launch(item)

    def schedule_pause(self, step, hhmm):
        self.pause_at_step = int(step or 0)
        self.pause_at_time = None
        hhmm = (hhmm or "").strip()
        if hhmm:
            m = re.match(r"^(\d{1,2})[:hH](\d{2})$", hhmm)
            if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
                return "Heure invalide : format attendu HH:MM (ex. 07:30)."
            now = datetime.datetime.now()
            target = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
            if target <= now:
                target += datetime.timedelta(days=1)
            self.pause_at_time = target
        return self.schedule_text()

    def schedule_text(self):
        parts = []
        if self.pause_at_step:
            parts.append(f"au pas {self.pause_at_step}")
        if self.pause_at_time:
            parts.append("le " + self.pause_at_time.strftime("%d/%m a %H:%M"))
        return ("Pause automatique programmee " + " ou ".join(parts) + " (au premier atteint)."
                if parts else "Aucune pause automatique programmee.")

    # ---- fin de job
    def _job_finished(self, run):
        kind, name, code = run.kind, run.job_name, run.exit_code
        # rend au GPU sa limite de puissance d'origine, sauf si un job suit dans la file
        if self._power_changed and not (self.queue and not self.hold):
            self._power_changed = False
            if is_admin():
                restore_power_limit()
            else:
                # pas de fenetre de confirmation surprise en fin de job
                self.power_message = ("La limite de puissance reste en place (bouton 'Retablir la "
                                      "limite d'origine' dans Outils, ou redemarrage du PC).")
        stopped = bool(run.stop_phase)
        if kind == "calib":
            if stopped:
                self.calib_report = "Calibrage interrompu."
                return
            self.calib_report = build_calibration_report(run, self.calib_params or {})
            notify("Calibrage termine", self.calib_report.splitlines()[0])
            return
        if stopped:
            if run.stop_phase.startswith("En pause"):
                notify("Entrainement en pause", f"{name} : {run.stop_phase}")
            return
        if kind == "download":
            return
        if name and name.startswith(REG_PREFIX):
            self.last_message = collect_reg(name, code == 0)
            notify("Images de regularisation", self.last_message)
            return
        label = {"train": "Entrainement", "test": "Test de LoRA", "caption": "Legendage"}.get(kind, kind)
        if code == 0:
            extra = ""
            if kind == "train" and not pinned_commit():
                extra = " Pensez a figer la version d'ai-toolkit (onglet Outils)."
            self.last_message = f"{label} termine : {name or run.label}.{extra}"
            notify(f"{label} termine", self.last_message)
        else:
            self.last_message = f"{label} en ERREUR : {name or run.label} - {run.phase}"
            notify(f"{label} en erreur", self.last_message)
        if not self.queue and kind == "test":
            notify("Tests de LoRA termines", "Toutes les series sont dans l'onglet Tester un LoRA.")


ORCH = Orchestrator()


# =========================================================== actions : entrainement
def start_training(P, queue=False):
    folder = P.get("folder")
    if not folder or not os.path.isdir(folder):
        return "Dossier du dataset introuvable.", None
    name, cfg = build_config(P)
    if name.startswith("_"):
        return "Les noms commencant par _ sont reserves (calibrage, tests).", None
    cfg_path = write_config(name, cfg)
    save_job_settings({**P, "name": name})
    done = checkpoint_steps(name)
    item = {"kind": "train", "name": name, "cfg_path": cfg_path, "label": f"Entrainement {name}",
            "env": _job_env(cfg, P, start_step=done[-1] if done else 0)}
    ram = ram_warning(P).strip()
    note_ram = ("\n" + ram if ram else "") + prompt_warning(P)
    if queue:
        pos = ORCH.enqueue(item)
        return f"'{name}' ajoute a la file d'attente (position {pos}).{note_ram}", None
    if RUN.running():
        return ("Un processus tourne deja : mettez-le en pause, ou utilisez "
                "'Ajouter a la file d'attente'."), None
    ORCH.hold = False
    resumed = bool(job_checkpoints(name))
    ok, note = ORCH.launch(item, interactive=True)
    if not ok:
        return "Impossible de lancer le job.", None
    if resumed:
        return f"Reprise de '{name}' depuis le dernier checkpoint valide.{note}{note_ram}", name
    return f"Entrainement '{name}' lance.{note}{note_ram}", name


def _wait_files_quiet(job_dir, quiet=4, timeout=60):
    """Attend que plus aucun checkpoint du job ne soit en cours d'ecriture."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = glob.glob(os.path.join(job_dir, "*.safetensors")) + \
            glob.glob(os.path.join(job_dir, "*.pt"))
        if not files or time.time() - max(os.path.getmtime(f) for f in files) > quiet:
            return
        time.sleep(1)


def stop_job():
    if not RUN.running():
        return "Rien a arreter."
    if ORCH.queue:
        ORCH.hold = True
    if RUN.kind in Runner.CONTROLLED and RUN.job_dir:
        _wait_files_quiet(RUN.job_dir)
    RUN.stop()
    return ("Arrete SANS sauvegarde : la reprise partira du dernier checkpoint automatique. "
            "Utilisez Pause pour sauvegarder l'etat exact.")


def _pause_worker(job, job_dir, t0, reason):
    try:
        # 1. ai-toolkit prend la demande a la fin du pas en cours
        deadline = time.time() + 900
        saw_saving = False
        while RUN.running() and time.time() < deadline:
            st = db_get(job)
            saw_saving = saw_saving or bool(st and st["info"] == "Saving model")
            if st and not st["save_now"]:
                break
            time.sleep(0.3)
        # 2. attendre un checkpoint complet, ecrit apres la demande. Des que
        #    ai-toolkit repasse de "Saving model" a "Training", on arrete : au
        #    plus un pas est refait a la reprise. Sinon, repli : fichier stable 4 s.
        deadline = time.time() + 600
        saved = False
        while RUN.running() and time.time() < deadline:
            info = (db_get(job) or {}).get("info")
            saw_saving = saw_saving or info == "Saving model"
            new = [f for f in glob.glob(os.path.join(job_dir, "*.safetensors"))
                   if os.path.getmtime(f) >= t0]
            if new and info != "Saving model" and all(safetensors_ok(f) for f in new) \
                    and (saw_saving or time.time() - max(os.path.getmtime(f) for f in new) > 4):
                saved = True
                break
            time.sleep(0.3)
        if not (saved and saw_saving):
            _wait_files_quiet(job_dir, timeout=30)
        step = (db_get(job) or {}).get("step", "?")
        suffix = f" ({reason})" if reason else ""
        RUN.stop(phase=(f"En pause au pas {step} - etat sauvegarde{suffix}" if saved else
                        "En pause - sauvegarde non confirmee, reprise au dernier checkpoint valide"))
    finally:
        RUN.pausing = False


def pause_job(reason=""):
    if not (RUN.running() and RUN.kind == "train"):
        return "Aucun entrainement en cours."
    if RUN.pausing:
        return "Pause deja demandee, sauvegarde en cours..."
    if ORCH.queue:
        ORCH.hold = True
    if RUN.train_bar is None:
        # pas encore dans la boucle d'entrainement : rien a sauvegarder
        RUN.stop(phase="En pause avant le premier pas - rien a sauvegarder")
        return "Arrete avant le debut de l'entrainement (les caches sont conserves)."
    RUN.pausing = True
    RUN.phase = "Pause demandee : sauvegarde a la fin du pas en cours..."
    db_set(RUN.job_name, "save_now", 1)
    threading.Thread(target=_pause_worker,
                     args=(RUN.job_name, RUN.job_dir, time.time() - 1, reason), daemon=True).start()
    return ("Pause demandee : sauvegarde de l'etat (LoRA + optimiseur + numero de pas), puis arret "
            "(au plus un ou deux pas seront refaits a la reprise). "
            "Pour reprendre : meme nom de job, bouton Lancer / Reprendre.")


def sample_now():
    if not (RUN.running() and RUN.kind == "train") or RUN.train_bar is None:
        return "Les apercus a la demande ne sont possibles que pendant l'entrainement."
    db_set(RUN.job_name, "sample_now", 1)
    return "Apercus demandes : generes a la fin du pas en cours (onglet Resultats)."


def resume_queue():
    ORCH.hold = False
    return ORCH.queue_text()


def clear_queue():
    ORCH.queue.clear()
    ORCH.hold = False
    return ORCH.queue_text()


# =========================================================== actions : calibrage
def start_calibration(P):
    folder = P.get("folder")
    if RUN.running():
        return "Un processus tourne deja."
    if not folder or not os.path.isdir(folder):
        return "Dossier du dataset introuvable."
    Q = dict(P, name=CALIB_NAME, steps=CALIB_STEPS, no_sample=True, save_every=100000)
    name, cfg = build_config(Q)
    # toujours repartir de zero : sinon le job, deja "termine", ne ferait aucun pas
    shutil.rmtree(os.path.join(OUTPUT, name), ignore_errors=True)
    cfg_path = write_config(name, cfg)
    ORCH.calib_params = dict(P)
    ORCH.calib_report = "Calibrage en cours : chargement du modele puis 20 pas de mesure..."
    ok, _ = ORCH.launch({"kind": "calib", "name": name, "cfg_path": cfg_path,
                         "label": "Test rapide (calibrage, 20 pas)", "env": _job_env(cfg, P)},
                        interactive=True)
    return "Test rapide lance : resultat dans le cadre 'Calibrage'." if ok else "Lancement impossible."


def build_calibration_report(run, P):
    """Mesures du test rapide -> texte avec recommandations chiffrees."""
    steps = int(P.get("steps") or 2000)
    offload = float(P.get("offload") or 0)
    res = ", ".join(str(r) for r in (P.get("resolutions") or []))
    total = MON.vram_total or 24.0
    lines = []
    if run.exit_code != 0:
        lines.append("Calibrage ECHOUE.")
        if run.saw_compile_error:
            lines.append("- Erreur de compilation triton : cochez 'Correctif w4a8', ou repassez au preset int8.")
        elif run.saw_oom:
            nxt = min(0.8, round(offload + 0.15, 2))
            lines.append(f"- Memoire GPU insuffisante avec part en RAM = {offload:.2f} et resolutions {res}.")
            lines.append(f"- Essayez part en RAM = {nxt:.2f}, puis la resolution 1024 seule, puis rang 8.")
        else:
            lines.append(f"- {run.phase}. Consultez le journal (output\\{CALIB_NAME}\\journal_ui.txt).")
        return "\n".join(lines)
    sec = rate_to_seconds((run.train_bar or {}).get("rate"))
    peak = MON.peak_train
    lines.append("Calibrage reussi (20 pas, mesure indicative : les premiers pas sont plus lents).")
    if sec:
        hours = sec * steps / 3600
        lines.append(f"- Vitesse : {sec:.1f} s par pas -> environ {hours:.1f} h pour {steps} pas, hors apercus.")
    else:
        lines.append("- Vitesse : non lue (voir le journal).")
    if peak:
        margin = total - peak
        lines.append(f"- Pic de VRAM pendant les pas : {peak:.1f} / {total:.1f} Go (marge {margin:.1f} Go).")
        if margin < 1.0:
            lines.append(f"- Marge trop faible : montez la part en RAM a {min(0.8, offload + 0.1):.2f}, "
                         "ou retirez la plus haute resolution.")
        elif margin > 4.0 and offload > 0:
            lines.append(f"- Marge confortable : essayez une part en RAM de {max(0.0, offload - 0.1):.2f} "
                         "(plus rapide), puis relancez le test.")
        else:
            lines.append("- Reglage coherent pour 24 Go : vous pouvez lancer l'entrainement.")
        lines.append("- Les apercus (surtout en 2048) demandent davantage : s'ils saturent, passez-les en 1536.")
        lines.append("- Repere (mesures Fizgig, base int8 sans dechargement) : ~21,0 Go en 512 px, ~21,7 Go "
                     "en 1024 px. La resolution change surtout la duree d'un pas, tres peu la memoire.")
    else:
        lines.append("- Pic de VRAM : non mesure (nvidia-smi indisponible).")
    if MON.ram_used and MON.ram_total:
        lines.append(f"- RAM utilisee : {MON.ram_used:.0f} / {MON.ram_total:.0f} Go.")
    return "\n".join(lines)


# =========================================================== actions : legendage
def start_captioning(folder, model, recaption, prompt, max_tokens=256):
    if RUN.running():
        return f"Un autre job tourne deja ({RUN.label}) : attendez sa fin ou arretez-le."
    if not folder or not os.path.isdir(folder):
        return "Dossier introuvable."
    cfg = {"job": "extension", "config": {"name": "_legendes", "process": [{
        "type": "Qwen3VLCaptioner",
        "device": "cuda",
        "caption": {
            "model_name_or_path": model or CAPTION_MODELS[0],
            "dtype": "bf16", "quantize": True, "qtype": "convrot8", "low_vram": False,
            "extensions": ["jpg", "jpeg", "png", "webp"],
            "path_to_caption": folder, "recaption": bool(recaption),
            # compile=True passerait par torch.compile : inutilement fragile sous Windows
            "compile": False, "caption_extension": "txt",
            "caption_prompt": (prompt or CAPTION_PROMPT).strip(),
            "max_res": 768, "max_new_tokens": int(max_tokens), "batch_size": 1,
        }}]}}
    cfg_path = write_config("_legendes", cfg)
    freed = "".join(hook() for hook in BEFORE_GPU_JOB)
    ok = RUN.start([PY, RUN_JOB, cfg_path], "caption",
                   f"Legendage automatique ({count_images(folder)} images, {(model or CAPTION_MODELS[0]).split('/')[-1]})",
                   os.path.join(CACHE, "journal_legendes.txt"))
    return ("Legendage lance (le modele est telecharge au premier usage). "
            "Relisez ensuite les .txt : une legende fausse s'apprend aussi." + freed
            if ok else "Lancement impossible.")


# =========================================================== actions : test de LoRA
def start_lora_test(job, checkpoints, prompts, size, seed):
    """Un job par checkpoint : on recharge le LoRA comme poids de depart, 0 pas
    d'entrainement, et ai-toolkit genere les apercus avec son propre code
    d'echantillonnage (celui de l'entrainement, connu pour ce modele)."""
    base_path = os.path.join(CONFIGS, f"{job}.yaml")
    if not job or not os.path.isfile(base_path):
        return "Config du job introuvable (configs\\<job>.yaml) : test possible seulement pour un job lance depuis cette UI."
    if not checkpoints:
        return "Selectionnez au moins un checkpoint."
    plist = [p.strip() for p in (prompts or "").splitlines() if p.strip()]
    if not plist:
        return "Saisissez au moins un prompt."
    with open(base_path, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    added = ""
    if (load_job_settings(job) or {}).get("control", True):
        ctl, auto = control_prompt(plist, base["config"]["process"][0].get("trigger_word"))
        if auto:
            plist.append(ctl)
            added = f" Prompt temoin sans mot declencheur ajoute : '{ctl}'."
    count = 0
    for ck in checkpoints:
        path = ck if os.path.isabs(ck) else os.path.join(OUTPUT, job, ck)
        if not safetensors_ok(path):
            continue
        tag = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(path))[0])
        name = f"{TEST_PREFIX}{tag}"
        cfg = json.loads(json.dumps(base))  # copie profonde
        cfg["config"]["name"] = name
        proc = cfg["config"]["process"][0]
        proc["network"]["pretrained_lora_path"] = path
        proc["train"].update({"steps": 0, "skip_first_sample": True, "disable_sampling": False})
        proc["save"]["save_every"] = 100000
        proc["sample"].update({"prompts": plist, "width": int(size), "height": int(size),
                               "seed": int(seed), "walk_seed": False, "sample_every": 100000})
        shutil.rmtree(os.path.join(OUTPUT, name), ignore_errors=True)
        cfg_path = write_config(name, cfg)
        saved = load_job_settings(job) or {}
        ORCH.enqueue({"kind": "test", "name": name, "cfg_path": cfg_path,
                      "label": f"Test {os.path.basename(path)}",
                      "env": _job_env(cfg, {"sampler": saved.get("sampler"),
                                            "w4a8_fix": saved.get("w4a8_fix", True)})})
        count += 1
    if not count:
        return "Aucun checkpoint valide selectionne."
    ORCH.hold = False
    return (f"{count} test(s) ajoute(s) a la file. Chaque test recharge le modele "
            "(plusieurs minutes) : les images arrivent une serie apres l'autre." + ("\n" + added.strip() if added else ""))


def lora_test_results(job):
    """[(chemin image, legende)] pour tous les tests des checkpoints du job."""
    out = []
    for d in sorted(glob.glob(os.path.join(OUTPUT, f"{TEST_PREFIX}{job}*"))):
        ck = os.path.basename(d)[len(TEST_PREFIX):]
        # checkpoints du job : "<job>" (final) ou "<job>_<numero de pas>", pas "<job>2..."
        if not re.fullmatch(r"(_\d+)?", ck[len(job):]):
            continue
        imgs = sorted(f for f in glob.glob(os.path.join(d, "samples", "*.*"))
                      if f.lower().endswith(IMG_EXT))
        for i, img in enumerate(imgs):
            out.append((img, f"{ck} | prompt {i + 1}"))
    return out


# =========================================================== images de regularisation
def reg_folder(folder):
    """Dossier des images de regularisation, a cote du dataset (ai-toolkit lit les sous-dossiers)."""
    return (os.path.normpath(folder) + "_regularisation") if folder else ""


def reg_prompts(folder, trigger, count, subject=CONTROL_SUBJECT):
    """Legendes du dataset, mot declencheur (et variantes proches) remplace par 'a person'."""
    caps = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".txt") and any(
                os.path.isfile(os.path.join(folder, os.path.splitext(f)[0] + e)) for e in IMG_EXT):
            try:
                with open(os.path.join(folder, f), encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
            except OSError:
                continue
            if text:
                ctl, _auto = control_prompt([text], trigger, subject)
                caps.append(ctl or text)
    if not caps:
        return []
    return [caps[i % len(caps)] for i in range(count)]


def start_reg_generation(P, count):
    """Genere les images de regularisation avec le modele de base (job ai-toolkit a 0 pas,
    LoRA neutre), puis les range avec leurs legendes dans <dataset>_regularisation."""
    folder, trigger = P.get("folder"), (P.get("trigger") or "").strip()
    if not folder or not os.path.isdir(folder):
        return "Dossier du dataset introuvable."
    if not trigger:
        return "Renseignez d'abord le mot declencheur (onglet Entrainement) : il est remplace par 'a person'."
    count = int(max(4, min(60, int(count or 0) or count_images(folder))))
    prompts = reg_prompts(folder, trigger, count, (P.get("dop_class") or CONTROL_SUBJECT).strip())
    if not prompts:
        return "Aucune legende dans le dataset : creez-les d'abord (legendage automatique)."
    job = safe_name(P.get("name") or os.path.basename(os.path.normpath(folder)))
    name = f"{REG_PREFIX}{job}"
    q = dict(P)
    q.update(name=name, no_sample=False, control=False, dop=False, reg=False)
    _n, cfg = build_config(q)
    cfg["config"]["name"] = name
    proc = cfg["config"]["process"][0]
    proc["train"].update({"steps": 0, "skip_first_sample": True, "disable_sampling": False})
    proc["train"].pop("ema_config", None)
    proc["save"]["save_every"] = 100000
    proc["sample"].update({"prompts": prompts, "width": 1024, "height": 1024, "seed": 1234,
                           "walk_seed": True, "sample_every": 100000})
    proc.pop("trigger_word", None)
    shutil.rmtree(os.path.join(OUTPUT, name), ignore_errors=True)
    os.makedirs(os.path.join(OUTPUT, name), exist_ok=True)
    with open(os.path.join(OUTPUT, name, "reg_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"folder": reg_folder(folder), "prompts": prompts}, fh)
    cfg_path = write_config(name, cfg)
    ORCH.enqueue({"kind": "test", "name": name, "cfg_path": cfg_path,
                  "label": f"Images de regularisation ({len(prompts)})",
                  "env": _job_env(cfg, {"sampler": SAMPLER_EULER, "w4a8_fix": P.get("w4a8_fix", True)})})
    ORCH.hold = False
    return (f"Generation de {len(prompts)} image(s) de regularisation ajoutee a la file (modele de base, "
            "1024 px). Elles seront rangees dans " + reg_folder(folder) + ".")


def collect_reg(name, ok):
    """Fin de la generation : copie les images et leurs legendes dans le dossier de regularisation."""
    meta_path = os.path.join(OUTPUT, name, "reg_meta.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return "Images de regularisation : informations de generation introuvables."
    imgs = sorted(f for f in glob.glob(os.path.join(OUTPUT, name, "samples", "*.*")) if f.lower().endswith(IMG_EXT))
    if not ok or not imgs:
        return "Images de regularisation : la generation a echoue (voir le journal)."
    dest = meta["folder"]
    os.makedirs(dest, exist_ok=True)
    for f in glob.glob(os.path.join(dest, "reg_*.*")):
        os.remove(f)
    n = 0
    for path in imgs:
        m = re.search(r"_(\d+)\.[a-z]+$", os.path.basename(path), re.I)
        idx = int(m.group(1)) if m else n
        prompt = meta["prompts"][idx] if idx < len(meta["prompts"]) else ""
        base = os.path.join(dest, f"reg_{idx:03d}")
        shutil.copy2(path, base + os.path.splitext(path)[1].lower())
        with open(base + ".txt", "w", encoding="utf-8") as fh:
            fh.write(prompt)
        n += 1
    return f"{n} image(s) de regularisation pretes dans {dest}. Cochez 'Images de regularisation' pour les utiliser."


def reg_status(folder):
    d = reg_folder(folder)
    n = count_images(d) if d and os.path.isdir(d) else 0
    return (f"{n} image(s) de regularisation disponibles dans {d}." if n else
            "Aucune image de regularisation pour ce dataset : generez-les d'abord.")


# =========================================================== version d'ai-toolkit
def _git(*args):
    try:
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        r = subprocess.run(["git", "-C", AITK, *args], capture_output=True, text=True,
                           timeout=20, creationflags=flags)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def current_commit():
    return _git("rev-parse", "HEAD")


def pinned_commit():
    try:
        with open(PIN_FILE, encoding="ascii") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def pin_version():
    commit = current_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return "Commit courant illisible (git indisponible ?)."
    with open(PIN_FILE, "w", encoding="ascii", newline="\n") as fh:
        fh.write(commit + "\n")
    return version_text()


def unpin_version():
    with contextlib.suppress(OSError):
        os.remove(PIN_FILE)
    return version_text()


def version_text():
    cur, pin = current_commit(), pinned_commit()
    date = _git("log", "-1", "--format=%cd", "--date=short")
    lines = [f"Version installee : {cur[:10] or 'inconnue'} ({date or '?'})"]
    lines.append(f"Version FIGEE sur {pin[:10]} : 3_METTRE_A_JOUR.bat ne changera plus le code."
                 if pin else "Version non figee : 3_METTRE_A_JOUR.bat installe la derniere version.")
    return "\n".join(lines)


# =========================================================== ComfyUI
def copy_to_comfy(path):
    target = (load_settings().get("comfy_loras_dir") or "").strip()
    if not target or not os.path.isdir(target):
        return "Renseignez d'abord le dossier 'loras' de ComfyUI dans l'onglet Outils."
    if not path or not safetensors_ok(path):
        return "Checkpoint introuvable ou incomplet."
    dest = os.path.join(target, os.path.basename(path))
    shutil.copy2(path, dest)
    return f"Copie : {dest}"


# =========================================================== telechargements
def start_download(choice):
    if RUN.running():
        return "Un processus tourne deja."
    files = list(FILES_COMMON)
    if choice in ("w4a8", "les deux"):
        files.append(FILE_W4A8)
    if choice in ("int8", "les deux"):
        files.append(FILE_INT8)
    hf = os.path.join(os.path.dirname(PY), "hf.exe" if IS_WIN else "hf")
    repo, _, fname = ADAPTER.rpartition("/")
    adapter_dir = os.path.join(MODELS, "loras", "training_adapters")
    # boucle de reprise : hf reprend les fichiers partiels la ou ils sont
    script = (
        "import subprocess,sys,time\n"
        f"cmds=[[{hf!r},'download',{H3_REPO!r},*{files!r},'--local-dir',{MODELS!r}],"
        f"[{hf!r},'download',{repo!r},{fname!r},'--local-dir',{adapter_dir!r}]]\n"
        "for c in cmds:\n"
        "  for t in range(1,6):\n"
        "    r=subprocess.call(c)\n"
        "    if r==0: break\n"
        "    print(f'Coupure : nouvelle tentative {t}/5 dans 10 s (reprise du fichier partiel)',flush=True)\n"
        "    time.sleep(10)\n"
        "  else: sys.exit(1)\n"
        "print('Telechargement termine.',flush=True)\n"
    )
    RUN.start([PY, "-c", script], "download", "Telechargement des poids",
              os.path.join(CACHE, "journal_telechargement.txt"))
    return "Telechargement lance (reprise automatique en cas de coupure)."


def models_state():
    rows = []
    for f in FILES_COMMON + [FILE_INT8, FILE_W4A8]:
        p = os.path.join(MODELS, f)
        base = os.path.basename(f)
        part = [x for x in glob.glob(os.path.join(MODELS, ".cache", "**", "*.incomplete"),
                                     recursive=True) if base in os.path.basename(x)]
        if os.path.isfile(p):
            rows.append(f"OK        {f}  ({os.path.getsize(p) / 1e9:.1f} Go)")
        elif part:
            rows.append(f"PARTIEL   {f}  (sera repris)")
        else:
            rows.append(f"absent    {f}")
    return "\n".join(rows)
