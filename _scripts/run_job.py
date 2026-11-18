"""Lance ai-toolkit/run.py (dossier courant = ai-toolkit) avec des ajustements
optionnels, actives par variables d'environnement. Rien n'est modifie sur le
disque : les modules d'ai-toolkit sont ajustes en memoire, a leur import, et
chaque ajustement verifie d'abord que le code attendu est bien la ; sinon il
s'abstient et le dit dans le journal.

  H3UI_W4A8_FIX=1        Correctif w4a8 pour les GPU anterieurs aux RTX 40 : le
                         noyau triton de decompression utilise un type FP8 que
                         triton ne compile qu'a partir de sm_89. On force le
                         chemin "torch pur" prevu par ai-toolkit.
  H3UI_LIKENESS=1        Optimisation de ressemblance (recette Fizgig) : tire le
                         niveau de bruit le plus net (sigma < 0,5) plus souvent
                         que 1 fois sur 6, selon un calendrier par passage.
                         H3UI_EPOCH_STEPS = pas par passage (nombre d'images),
                         H3UI_START_STEP = pas de reprise.
  H3UI_LCM=1             Apercus avec l'echantillonneur LCM (bruit frais a
                         chaque pas, 1,1 -> 0,9) au lieu d'Euler.
  H3UI_CAPTION_VARIANTS=1
                         Legendes multiples : un .txt de plusieurs lignes donne
                         une variante par ligne. Chaque image est dupliquee en
                         memoire (comme les retournements d'ai-toolkit), une
                         copie par variante : memes latents, une legende chacune.
  H3UI_FIZGIG_ADAPTER=<fichier de destination>
                         Telecharge l'adaptateur d'entrainement de Fizgig et le
                         convertit du format kohya vers celui d'ai-toolkit.

Tout ceci est experimental : non valide sur materiel reel par l'auteur.
"""

import importlib.abc
import importlib.util
import os
import random
import re
import runpy
import sys

W4A8_MODULE = os.environ.get("H3UI_FIX_TARGET", "toolkit.util.convrot_w4a8_quant")
SCHEDULER_MODULE = "toolkit.samplers.custom_flowmatch_sampler"
PIPELINE_MODULE = "extensions_built_in.diffusion_models.minimax_h3.src.pipeline"
DATALOADER_MODULE = "toolkit.data_loader"

FIZGIG_REPO = "ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter"
FIZGIG_FILE = "fizgig_minimax_h3_image_training_adapter.safetensors"


def say(text):
    print("[H3-UI] " + text, flush=True)


# ------------------------------------------------------------------ w4a8
def patch_w4a8(module):
    try:
        import torch

        if not torch.cuda.is_available():
            return
        cap = torch.cuda.get_device_capability(0)
        if cap >= (8, 9):
            return  # RTX 40 et plus : le noyau d'origine fonctionne
        if not hasattr(module, "_triton_available"):
            say("Correctif w4a8 : module inattendu, correctif non applique.")
            return
        module._triton_available = lambda: False
        say("Correctif w4a8 actif (GPU sm_%d%d) : decompression en torch pur." % cap)
    except Exception as exc:  # ne jamais empecher l'entrainement de demarrer
        say("Correctif w4a8 non applique : %s" % exc)


# ------------------------------------------------------------------ ressemblance
# Calendrier par defaut de Fizgig (families/minimax_image.py, LIKENESS_DEFAULTS) :
# passages 1, 3, 5 : 16,67 % -> 33,33 % -> 50 % ; passages 2, 4, 6 : 75 % ;
# a partir du passage 7 : 75 %.
LIKENESS = {"before": 16.67, "before_end": 50.0, "hinge": 6, "after": 75.0}


def clean_share(epoch, cfg=LIKENESS):
    """Part (en %) des pas tires au niveau de bruit le plus net, pour ce passage (>= 1)."""
    hinge = int(cfg["hinge"])
    if epoch > hinge:
        return cfg["after"]
    if (epoch - 1) % 2 == 1:          # passages pairs : la part d'apres la charniere
        return cfg["after"]
    k, n = (epoch - 1) // 2, (hinge + 1) // 2
    frac = k / (n - 1) if n > 1 else 0.0
    return round(cfg["before"] + (cfg["before_end"] - cfg["before"]) * frac, 2)


def pick_sigma(sigmas, share_pct, rng=random):
    """Un sigma de la grille : share_pct % pour ceux < 0,5, le reste reparti sur les autres."""
    lo = [s for s in sigmas if s < 0.5]
    hi = [s for s in sigmas if s >= 0.5]
    if not lo or not hi:
        return rng.choice(list(sigmas))
    p = min(max(share_pct / 100.0, 0.0), 1.0)
    return rng.choice(lo) if rng.random() < p else rng.choice(hi)


def patch_scheduler(module):
    try:
        cls = getattr(module, "CustomFlowMatchEulerDiscreteScheduler", None)
        original = getattr(cls, "set_train_timesteps", None)
        if original is None:
            say("Ressemblance : planificateur inattendu, option non appliquee.")
            return
        import torch

        epoch_steps = max(1, int(os.environ.get("H3UI_EPOCH_STEPS", "1") or 1))
        state = {"calls": int(os.environ.get("H3UI_START_STEP", "0") or 0), "said": None}

        def set_train_timesteps(self, num_timesteps, device, *args, **kwargs):
            fixed = getattr(self, "fixed_train_sigmas", None)
            if fixed is None or len(fixed) < 2:
                return original(self, num_timesteps, device, *args, **kwargs)
            epoch = state["calls"] // epoch_steps + 1
            state["calls"] += 1
            share = clean_share(epoch)
            if state["said"] != share:
                state["said"] = share
                say("Ressemblance : %s %% des pas au niveau le plus net (passage %d)." % (share, epoch))
            # une grille a un seul point : le tirage uniforme d'ai-toolkit tombe
            # forcement dessus, sans toucher a son code (lots de 1 image)
            sigmas = torch.tensor([pick_sigma([float(s) for s in fixed], share)],
                                  dtype=torch.float32, device=device)
            self.timestep_type = kwargs.get("timestep_type", getattr(self, "timestep_type", "linear"))
            self.timesteps = sigmas * self.config.num_train_timesteps
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
            return self.timesteps

        cls.set_train_timesteps = set_train_timesteps
        say("Optimisation de ressemblance active (%d pas par passage)." % epoch_steps)
    except Exception as exc:
        say("Ressemblance non appliquee : %s" % exc)


# ------------------------------------------------------------------ LCM
EULER_LINE = "video_rows = ratio_v * video_rows + (1.0 - ratio_v) * denoised_v"
LCM_LINE = "video_rows = _h3ui_lcm_step(video_rows, denoised_v, sv_next, i, num_steps, generator)"
LCM_NOISE = (1.1, 0.9)   # force du bruit frais, du premier au dernier pas (reglage de Fizgig)


def lcm_step(x, denoised, sigma_next, i, n, generator=None):
    """Pas LCM : repart de l'image nette estimee, avec un bruit frais au niveau suivant."""
    import torch

    s_next = float(sigma_next)
    if s_next <= 0.0:
        return denoised
    a, b = LCM_NOISE
    k = a + (b - a) * (i / (n - 1) if n > 1 else 0.0)
    gen_device = getattr(generator, "device", None)
    if generator is not None and gen_device is not None and gen_device.type != x.device.type:
        noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device, x.dtype)
    else:
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return s_next * (k * noise) + (1.0 - s_next) * denoised


def rewrite_pipeline_source(source):
    """Source du pipeline avec le pas d'Euler remplace par le pas LCM, ou None."""
    if source.count(EULER_LINE) != 1:
        return None
    for needed in ("sv_next", "num_steps", "generator", "denoised_v"):
        if needed not in source:
            return None
    return source.replace(EULER_LINE, LCM_LINE)


def exec_pipeline_with_lcm(module, loader, original_exec):
    try:
        source = loader.get_source(module.__name__)
        new_source = rewrite_pipeline_source(source or "")
        if new_source is None:
            say("LCM : le pipeline d'ai-toolkit a change, apercus laisses en Euler.")
            return original_exec(module)
        module.__dict__["_h3ui_lcm_step"] = lcm_step
        exec(compile(new_source, getattr(module, "__file__", "<pipeline>"), "exec"), module.__dict__)
        say("Apercus en LCM (bruit %.1f -> %.1f)." % LCM_NOISE)
    except Exception as exc:
        say("LCM non applique (%s), apercus en Euler." % exc)
        module.__dict__.pop("_h3ui_lcm_step", None)
        return original_exec(module)


# ------------------------------------------------------------------ crochet d'import
# ------------------------------------------------------------------ legendes multiples
def caption_variants(path, ext):
    try:
        with open(os.path.splitext(path)[0] + ext, encoding="utf-8", errors="replace") as fh:
            return [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return []


def patch_dataloader(module):
    cls = getattr(module, "AiToolkitDataset", None)
    original = getattr(cls, "_scan_source", None) if cls is not None else None
    if original is None:
        say("Legendes multiples : code d'ai-toolkit inattendu, seule la legende entiere sera lue.")
        return
    import copy

    def _scan_source(self, dataset_config):
        start = len(self.file_list)
        original(self, dataset_config)
        try:
            ext = getattr(dataset_config, "caption_ext", "txt") or "txt"
            ext = ext if ext.startswith(".") else "." + ext
            out, multi = [], 0
            for item in self.file_list[start:]:
                lines = [] if getattr(item, "is_reg", False) else caption_variants(item.path, ext)
                if len(lines) < 2:
                    out.append(item)
                    continue
                multi += 1
                for line in lines:
                    # objets partages (config du dataset, transformations) : gardes communs, pas copies
                    memo = {id(v): v for k, v in vars(item).items()
                            if k in ("dataset_config", "dataloader_transforms", "transform", "size_database")}
                    twin = copy.deepcopy(item, memo)
                    twin.raw_caption = line
                    twin.raw_caption_short = None
                    twin._text_embedding_path = None
                    twin.load_caption()          # recalcule legende, mot declencheur et DOP
                    out.append(twin)
            if multi:
                self.file_list[start:] = out
                say("Legendes multiples : %d image(s) a plusieurs variantes -> %d elements dans ce dossier."
                    % (multi, len(out)))
        except Exception as exc:  # ne jamais empecher l'entrainement de demarrer
            say("Legendes multiples non appliquees (%s)." % exc)

    cls._scan_source = _scan_source
    say("Legendes multiples actives : une copie de chaque image par ligne de sa legende.")


class _Hook(importlib.abc.MetaPathFinder):
    def __init__(self, wanted):
        self.wanted = dict(wanted)     # nom du module -> "w4a8" | "scheduler" | "pipeline"

    def find_spec(self, name, path, target=None):
        kind = self.wanted.pop(name, None)
        if kind is None:
            return None
        if not self.wanted and self in sys.meta_path:
            sys.meta_path.remove(self)
        active = self in sys.meta_path
        if active:
            sys.meta_path.remove(self)       # evite de se rappeler soi-meme
        try:
            spec = importlib.util.find_spec(name)
        finally:
            if active:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader, original = spec.loader, spec.loader.exec_module

        def exec_module(module):
            if kind == "pipeline":
                return exec_pipeline_with_lcm(module, loader, original)
            original(module)
            if kind == "w4a8":
                patch_w4a8(module)
            elif kind == "scheduler":
                patch_scheduler(module)
            elif kind == "dataloader":
                patch_dataloader(module)

        spec.loader.exec_module = exec_module
        return spec


# ------------------------------------------------------------------ adaptateur Fizgig
KOHYA_KEY = re.compile(r"^lora_unet_blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)"
                       r"\.(lora_down\.weight|lora_up\.weight|alpha)$")
MODULE_PATH = {"attn_qkv_proj": "attn.qkv_proj", "attn_out_proj": "attn.out_proj",
               "mlp_fc1": "mlp.fc1", "mlp_fc2": "mlp.fc2"}


def convert_kohya_adapter(state):
    """Cles kohya (lora_unet_blocks_N_attn_qkv_proj.lora_down.weight + alpha) ->
    cles d'ai-toolkit (diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight).
    ai-toolkit charge l'adaptateur avec alpha = rang : le facteur alpha / rang du
    fichier d'origine est donc integre dans lora_B. Retourne (dict, cles ignorees)."""
    if any(".lora_A." in k for k in state):          # deja au format d'ai-toolkit
        return dict(state), []
    groups, skipped = {}, []
    for key, value in state.items():
        m = KOHYA_KEY.match(key)
        if not m:
            skipped.append(key)
            continue
        groups.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = value
    out = {}
    for (block, mod), parts in sorted(groups.items()):
        down, up = parts.get("lora_down.weight"), parts.get("lora_up.weight")
        if down is None or up is None:
            skipped.append("blocks_%d_%s (incomplet)" % (block, mod))
            continue
        rank = down.shape[0]
        alpha = float(parts["alpha"]) if "alpha" in parts else float(rank)
        prefix = "diffusion_model.blocks.%d.%s" % (block, MODULE_PATH[mod])
        out[prefix + ".lora_A.weight"] = down
        out[prefix + ".lora_B.weight"] = up * (alpha / rank) if alpha != rank else up
    return out, skipped


def prepare_fizgig_adapter(dest):
    """Telecharge puis convertit l'adaptateur ; arrete le job si c'est impossible,
    plutot que d'entrainer en silence sans adaptateur."""
    if os.path.isfile(dest):
        return
    try:
        import huggingface_hub
        from safetensors.torch import load_file, save_file

        say("Telechargement de l'adaptateur d'entrainement Fizgig (~150 Mo)...")
        source = huggingface_hub.hf_hub_download(
            repo_id=FIZGIG_REPO, filename=FIZGIG_FILE,
            local_dir=os.path.join(os.path.dirname(dest), "fizgig_original"))
        converted, skipped = convert_kohya_adapter(load_file(source))
        n = len(converted) // 2
        if n < 100:
            sys.exit("[H3-UI] Adaptateur Fizgig : seulement %d modules convertis (200 attendus). "
                     "Son format a change : choisissez l'adaptateur ostris." % n)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        save_file(converted, dest + ".part")
        os.replace(dest + ".part", dest)
        say("Adaptateur Fizgig converti : %d modules%s." % (
            n, (", %d cles ignorees (ex. %s)" % (len(skipped), skipped[0])) if skipped else ""))
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit("[H3-UI] Adaptateur Fizgig indisponible (%s). Choisissez l'adaptateur ostris." % exc)


# ------------------------------------------------------------------ lancement
if __name__ == "__main__":
    wanted = {}
    if os.environ.get("H3UI_W4A8_FIX") == "1":
        wanted[W4A8_MODULE] = "w4a8"
    if os.environ.get("H3UI_LIKENESS") == "1":
        wanted[SCHEDULER_MODULE] = "scheduler"
    if os.environ.get("H3UI_LCM") == "1":
        wanted[PIPELINE_MODULE] = "pipeline"
    if os.environ.get("H3UI_CAPTION_VARIANTS") == "1":
        wanted[DATALOADER_MODULE] = "dataloader"
    if wanted:
        sys.meta_path.insert(0, _Hook(wanted))
    if os.environ.get("H3UI_FIZGIG_ADAPTER"):
        prepare_fizgig_adapter(os.environ["H3UI_FIZGIG_ADAPTER"])
    script = os.path.join(os.getcwd(), "run.py")
    if not os.path.isfile(script):
        sys.exit("run.py introuvable : ce script doit etre lance depuis le dossier ai-toolkit")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
