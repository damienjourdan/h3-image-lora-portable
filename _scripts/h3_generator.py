"""Pont vers le generateur MiniMaxH3-Portable (dossier separe, son propre environnement Python).

L'UI d'entrainement le demarre et l'arrete, l'affiche dans l'onglet Generer, lui envoie les checkpoints
(les LoRA d'ai-toolkit sont lus tels quels par son chargeur) et arbitre le GPU : il garde environ 20 Go de
VRAM charges, donc tout job GPU de l'entraineur l'arrete d'abord.
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time

import h3_core as core

# langues de l'entraineur -> langues du generateur (en, fr, de, es, nl, zh, ja)
LANG_MAP = {"en": "en", "fr": "fr", "de": "de", "es": "es", "pt": "en", "ja": "ja", "zh": "zh"}

LICENSE_NOTICE = (
    "**Licence MiniMax-H3-Image Community License** (rappel du generateur) : elle n'accorde AUCUN droit dans "
    "l'Union europeenne, au Royaume-Uni, en Coree du Sud ni aux Etats-Unis, y compris pour les images "
    "generees. Usage commercial au-dela d'1 M USD de chiffre d'affaires annuel ou service heberge : accord "
    "ecrit de MiniMax requis. Texte complet : https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE "
    "- Le respect de cette licence est de votre seule responsabilite.")


def is_generator(folder):
    return bool(folder) and os.path.isfile(os.path.join(folder, "app", "main.py")) \
        and os.path.isfile(os.path.join(folder, "app", "h3", "engine.py"))


def detect():
    """Dossier enregistre, sinon un dossier voisin de l'entraineur qui contient le generateur."""
    saved = (core.load_settings().get("gen_dir") or "").strip()
    if is_generator(saved):
        return saved
    parent = os.path.dirname(core.ROOT)
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return saved
    for n in names:
        p = os.path.join(parent, n)
        if os.path.isdir(p) and is_generator(p):
            return p
        sub = os.path.join(p, "MiniMaxH3-Portable")   # archive decompressee dans un dossier du meme nom
        if os.path.isdir(sub) and is_generator(sub):
            return sub
    return saved


def _config_path(folder):
    return os.path.join(folder, "config.json")


def read_config(folder):
    try:
        with open(_config_path(folder), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def write_config(folder, **changes):
    cfg = read_config(folder)
    cfg.update(changes)
    with open(_config_path(folder), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)


def _python(folder):
    return os.path.join(folder, "env", "Scripts", "python.exe") if core.IS_WIN \
        else os.path.join(folder, "env", "bin", "python")


def state(folder):
    """found, installed (environnement Python), prepared (modeles convertis), license, port."""
    cfg = read_config(folder) if is_generator(folder) else {}
    models = os.path.join(folder or "", "models")
    prepared = False
    if os.path.isdir(models):
        prepared = any(n.startswith("transformer_") and os.path.isfile(os.path.join(models, n, "manifest.json"))
                       for n in os.listdir(models))
    return {"found": is_generator(folder), "installed": os.path.isfile(_python(folder or "")),
            "prepared": prepared, "license": bool(cfg.get("license_ack")),
            "port": int(cfg.get("port") or 7865)}


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


class Generator:
    def __init__(self):
        self.proc = None
        self.folder = ""
        self.port = 7865
        self.lines = collections.deque(maxlen=300)
        self.ready = False
        self.started_at = 0.0
        self.exit_code = None
        self.note = ""
        self.lock = threading.Lock()
        atexit.register(self.stop)

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    # ---- demarrage
    def start(self, folder, lang="en"):
        folder = (folder or "").strip()
        if self.running():
            return "Le generateur tourne deja."
        if not is_generator(folder):
            return "Dossier du generateur introuvable : indiquez le dossier MiniMaxH3-Portable."
        st = state(folder)
        if not st["installed"]:
            return "Generateur non installe : lancez d'abord son install.bat."
        if not st["prepared"]:
            return "Modeles du generateur absents : lancez d'abord son prepare.bat."
        if not st["license"]:
            return "Acceptez d'abord la licence (case ci-dessous)."
        if core.RUN.running():
            return ("Un job utilise le GPU : attendez la fin, ou mettez-le en pause, avant de demarrer "
                    "le generateur.")
        if _port_open(st["port"]):
            self.folder, self.port, self.ready = folder, st["port"], True
            return f"Un generateur repond deja sur le port {st['port']} : il est affiche ci-dessous."
        core.save_settings(gen_dir=folder)
        write_config(folder, open_browser=False, language=LANG_MAP.get(lang, "en"))
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "HF_HOME", "HF_HUB_CACHE",
                            "TRANSFORMERS_CACHE", "MODELS_PATH")}
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"})
        if core.IS_WIN:
            # run.bat recale l'environnement si le dossier a ete deplace et installe les bibliotheques manquantes
            cmd = ["cmd", "/c", os.path.join(folder, "run.bat")]
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kw = {"creationflags": flags}
        else:
            run_sh = os.path.join(folder, "run.sh")
            cmd = ["bash", run_sh] if os.path.isfile(run_sh) else [_python(folder), os.path.join(folder, "app", "main.py"), "run"]
            kw = {"start_new_session": True}
        self.lines.clear()
        self.folder, self.port, self.ready, self.exit_code, self.note = folder, st["port"], False, None, ""
        self.started_at = time.time()
        self.proc = subprocess.Popen(cmd, cwd=folder, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, **kw)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()
        return "Demarrage du generateur : chargement des modeles (1 a 3 min)."

    def _read(self):
        proc = self.proc
        log = os.path.join(core.CACHE, "journal_generateur.txt")
        os.makedirs(core.CACHE, exist_ok=True)
        with open(log, "w", encoding="utf-8") as out:
            for raw in iter(proc.stdout.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip()
                if text:
                    self.lines.append(text)
                    out.write(text + "\n")
                    out.flush()
        code = proc.wait()
        if proc is self.proc:
            self.exit_code, self.ready = code, False

    def _watch(self):
        proc = self.proc
        while proc is self.proc and proc.poll() is None:
            if _port_open(self.port):
                self.ready = True
                core.notify("Generateur", "Le generateur est pret.")
                return
            time.sleep(2)

    # ---- arret
    def stop(self):
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return False
            try:
                if core.IS_WIN:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self.ready = False
            return True

    def free_gpu(self):
        """Appele avant chaque job GPU de l'entraineur."""
        if self.running() and self.stop():
            self.note = "Generateur arrete pour liberer le GPU (job de l'entraineur)."
            return " " + self.note
        return ""

    # ---- affichage
    def status(self):
        if self.running():
            if self.ready:
                return f"Pret : http://127.0.0.1:{self.port}"
            secs = int(time.time() - self.started_at)
            return f"Chargement des modeles... {secs} s (1 a 3 min)"
        if self.ready and _port_open(self.port):
            return f"Pret : http://127.0.0.1:{self.port}"
        if self.note:
            return self.note
        if self.exit_code not in (None, 0):
            return f"Arrete (code {self.exit_code}) : voir le journal."
        return "Arrete."

    def log_text(self, n=40):
        return "\n".join(list(self.lines)[-n:])

    # ---- LoRA
    def send_lora(self, path, folder=None):
        folder = (folder or self.folder or detect() or "").strip()
        if not is_generator(folder):
            return "Dossier du generateur introuvable : renseignez-le dans l'onglet Generer."
        if not path or not core.safetensors_ok(path):
            return "Checkpoint introuvable ou incomplet."
        dest_dir = os.path.join(folder, "loras")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path))
        shutil.copy2(path, dest)
        return (f"Envoye au generateur : {dest}. Dans le generateur, onglet LoRA : rafraichissez la liste "
                "puis choisissez-le (mot declencheur dans le prompt).")


GEN = Generator()
core.BEFORE_GPU_JOB.append(GEN.free_gpu)
