"""Ressemblance et fidelite au prompt des apercus, pour choisir le checkpoint sans tout regarder.

Deux mesures par apercu, calculees sur le processeur (la carte reste a l'entrainement) :
- ressemblance : similarite DINOv2 entre l'apercu et les images du dataset (moyenne) ;
- fidelite au prompt : similarite CLIP entre l'apercu et son prompt (sans le mot declencheur).
C'est le couple de mesures de l'article DreamBooth (scores DINO et CLIP-T). La ressemblance seule
recompenserait le sur-apprentissage (une copie du dataset ressemble parfaitement) : le checkpoint
conseille est le plus ressemblant parmi ceux qui respectent encore le prompt aussi bien qu'au debut.

Cache : output/<job>/ressemblance.json (rien n'est recalcule deux fois).
"""

import glob
import json
import os
import re
import threading
import time

import h3_core as core

DINO_ID = "facebook/dinov2-small"          # ~90 Mo
CLIP_ID = "openai/clip-vit-base-patch32"    # ~600 Mo
CACHE = "ressemblance.json"
SAMPLE_RX = re.compile(r"__(\d+)_(\d+)\.[a-z]+$", re.I)
CLIP_TOLERANCE = 0.025   # baisse de fidelite toleree par rapport au debut (similarite cosinus CLIP)


def _job_config(job):
    """Prompts d'apercu, mot declencheur et dossier du dataset, d'apres le YAML genere par l'UI."""
    path = os.path.join(core.CONFIGS, job + ".yaml")
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        proc = cfg["config"]["process"][0]
        return {"prompts": list(proc.get("sample", {}).get("prompts") or []),
                "trigger": proc.get("trigger_word") or "",
                "folder": proc["datasets"][0]["folder_path"]}
    except Exception:  # noqa: BLE001
        return None


def _clean_prompt(text, trigger):
    text = text.replace("[trigger]", " ")
    if trigger:
        text = re.sub(re.escape(trigger), " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" ,")


def read_cache(job):
    try:
        with open(os.path.join(core.OUTPUT, job, CACHE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def summary(job):
    """[(pas, ressemblance 0-100, fidelite 0-100, nb apercus)] par pas, trie."""
    data = read_cache(job).get("samples", {})
    by_step = {}
    for v in data.values():
        by_step.setdefault(v["step"], []).append(v)
    rows = []
    for st in sorted(by_step):
        vals = by_step[st]
        rows.append((st, 100 * sum(v["dino"] for v in vals) / len(vals),
                     100 * sum(v["clip"] for v in vals) / len(vals), len(vals)))
    return rows


def ck_steps(job, rows=None):
    """Pas des checkpoints presents ; le checkpoint final (sans numero) compte pour le dernier pas."""
    steps = list(core.checkpoint_steps(job))
    rows = rows if rows is not None else summary(job)
    if rows and os.path.isfile(os.path.join(core.OUTPUT, job, job + ".safetensors")):
        last = rows[-1][0]
        if last not in steps:
            steps.append(last)
    return sorted(steps)


def recommend(job, rows=None):
    """Pas conseille (avec checkpoint) : le plus ressemblant parmi ceux dont la fidelite au prompt
    n'a pas baisse de plus de CLIP_TOLERANCE par rapport au debut. None si pas assez de donnees."""
    rows = rows if rows is not None else summary(job)
    cks = set(ck_steps(job, rows))
    cand = [r for r in rows if r[0] in cks]
    if len(rows) < 2 or not cand:
        return None
    ref = max(r[2] for r in rows[:2])          # fidelite du debut (pas 0 = modele sans LoRA)
    ok = [r for r in cand if r[2] >= ref - 100 * CLIP_TOLERANCE]
    if ok:
        return max(ok, key=lambda r: r[1]), ref, False
    # tous les checkpoints ont decroche du prompt : meilleur compromis
    d_lo, d_hi = min(r[1] for r in cand), max(r[1] for r in cand)
    c_lo, c_hi = min(r[2] for r in cand), max(r[2] for r in cand)
    score = lambda r: ((r[1] - d_lo) / ((d_hi - d_lo) or 1)) + ((r[2] - c_lo) / ((c_hi - c_lo) or 1))  # noqa: E731
    return max(cand, key=score), ref, True


def _vec(out, name, torch):
    """Selon la version de transformers, get_*_features rend un tenseur ou une sortie structuree."""
    if torch.is_tensor(out):
        return out
    for attr in (name, "pooler_output"):
        val = getattr(out, attr, None)
        if val is not None:
            return val
    return out[0]


class Scorer:
    """Calcule en tache de fond les mesures des nouveaux apercus des jobs demandes et du job en cours."""

    def __init__(self):
        self.lock = threading.Lock()
        self.wanted = set()
        self.state = {}          # job -> texte d'etat
        self.error = None
        self.models = None
        self._wake = threading.Event()
        self._thread = None

    def enabled(self):
        return bool(core.load_settings().get("likeness_on", True))

    def request(self, job):
        if not job or not self.enabled():
            return
        with self.lock:
            self.wanted.add(job)
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        self._wake.set()

    def status(self, job):
        if not self.enabled():
            return "Mesure de ressemblance desactivee (onglet Outils)."
        if self.error:
            return self.error
        return self.state.get(job, "")

    # ------------------------------------------------------------- modeles
    def _load(self):
        if self.models:
            return self.models
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))  # laisse du processeur au reste
        dproc = AutoImageProcessor.from_pretrained(DINO_ID)
        dino = AutoModel.from_pretrained(DINO_ID).eval()
        cproc = CLIPProcessor.from_pretrained(CLIP_ID)
        clip = CLIPModel.from_pretrained(CLIP_ID).eval()
        self.models = (torch, dproc, dino, cproc, clip)
        return self.models

    def _dino(self, images):
        torch, dproc, dino, _cp, _c = self._load()
        with torch.no_grad():
            out = dino(**dproc(images=images, return_tensors="pt"))
            emb = out.last_hidden_state[:, 0]          # jeton CLS
            return torch.nn.functional.normalize(emb, dim=-1)

    def _clip(self, image, text):
        torch, _dp, _d, cproc, clip = self._load()
        with torch.no_grad():
            inp = cproc(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
            img = _vec(clip.get_image_features(pixel_values=inp["pixel_values"]), "image_embeds", torch)
            txt = _vec(clip.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]),
                       "text_embeds", torch)
            return float((torch.nn.functional.normalize(img, dim=-1) * torch.nn.functional.normalize(txt, dim=-1)).sum())

    # ------------------------------------------------------------- calcul
    def _process(self, job):
        cfg = _job_config(job)
        if not cfg or not os.path.isdir(cfg["folder"]):
            self.state[job] = "Ressemblance : configuration ou dataset du job introuvable."
            return
        samples = sorted(glob.glob(os.path.join(core.OUTPUT, job, "samples", "*.*")))
        samples = [p for p in samples if SAMPLE_RX.search(os.path.basename(p))]
        if not samples:
            return
        cache = read_cache(job)
        done = cache.get("samples", {})
        todo = [p for p in samples if os.path.basename(p) not in done]
        refs = sorted(f for f in os.listdir(cfg["folder"]) if f.lower().endswith(core.IMG_EXT))
        sig = "|".join(f"{f}:{os.path.getmtime(os.path.join(cfg['folder'], f)):.0f}" for f in refs)
        if not refs or (not todo and cache.get("refs_sig") == sig):
            return
        from PIL import Image

        if self.models is None:
            self.state[job] = ("Ressemblance : chargement des modeles de mesure (telecharges au premier usage, "
                               "environ 700 Mo)...")
        torch = self._load()[0]
        if cache.get("refs_sig") != sig:   # dataset change : tout est a refaire
            self.state[job] = f"Ressemblance : analyse du dataset ({len(refs)} images)..."
            embs = []
            for i in range(0, len(refs), 8):
                batch = [Image.open(os.path.join(cfg["folder"], f)).convert("RGB") for f in refs[i:i + 8]]
                embs.append(self._dino(batch))
            ref = torch.cat(embs)
            cache = {"refs_sig": sig, "ref_emb": [[round(x, 5) for x in r] for r in ref.tolist()], "samples": {}}
            done = cache["samples"]
            todo = samples
        ref = torch.tensor(cache["ref_emb"])
        for n, path in enumerate(todo, 1):
            self.state[job] = f"Ressemblance : mesure des apercus {n}/{len(todo)}..."
            m = SAMPLE_RX.search(os.path.basename(path))
            step, idx = int(m.group(1)), int(m.group(2))
            prompt = cfg["prompts"][idx] if idx < len(cfg["prompts"]) else ""
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                continue   # apercu en cours d'ecriture : repris au tour suivant
            emb = self._dino([img])[0]
            dino = float((ref @ emb).mean())
            clip = self._clip(img, _clean_prompt(prompt, cfg["trigger"])) if prompt else 0.0
            done[os.path.basename(path)] = {"step": step, "prompt": idx, "dino": round(dino, 4),
                                            "clip": round(clip, 4)}
            self._save(job, cache)
        self.state[job] = ""

    def _save(self, job, cache):
        path = os.path.join(core.OUTPUT, job, CACHE)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, path)

    def _loop(self):
        while True:
            self._wake.wait(30)
            self._wake.clear()
            if not self.enabled():
                continue
            jobs = set()
            with self.lock:
                jobs |= self.wanted
            if core.RUN.running() and core.RUN.kind == "train" and core.RUN.job_name:
                jobs.add(core.RUN.job_name)
            for job in sorted(jobs):
                try:
                    self._process(job)
                    self.error = None
                except Exception as exc:  # noqa: BLE001 - jamais bloquant pour l'UI
                    self.error = f"Ressemblance indisponible : {type(exc).__name__}: {str(exc)[:160]}"
                    self.state[job] = ""
                    time.sleep(60)


SCORER = Scorer()
