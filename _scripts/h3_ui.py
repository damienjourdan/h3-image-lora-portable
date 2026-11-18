"""UI Gradio (100 % Python, sans Node.js) pour entrainer des LoRA
MiniMax-H3-Image avec ostris/ai-toolkit sur une RTX 3090 24 Go / 64 Go RAM.

Lancee par 2_LANCER_UI.bat avec le Python du venv d'ai-toolkit.
La logique est dans h3_core.py ; ce fichier ne contient que l'interface.
"""

import os
import sys
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402

import h3_core as core  # noqa: E402
import h3_crop as crop  # noqa: E402
import h3_i18n as i18n  # noqa: E402
import h3_likeness as likeness  # noqa: E402
import h3_audit as audit  # noqa: E402
import h3_captionfix as capfix  # noqa: E402
import h3_generator as gen  # noqa: E402
from h3_core import MON, ORCH, RUN  # noqa: E402

# ------------------------------------------------------------------ textes d'aide
AIDE_DATASET = """
**Images**
- **Nombre** : 15 a 40 pour un sujet (personnage, objet), 30 a 100 pour un style.
- **Qualite** : nettes, sans filigrane ni texte parasite ; cote court d'au moins 1024 px.
- **Variete** : angles, cadrages, eclairages et decors differents. Ce qui ne change jamais
  d'une image a l'autre sera appris comme faisant partie du sujet.
- Les images sont redimensionnees et classees par format automatiquement : inutile de les recadrer en carre.

**Legendes** (un fichier .txt du meme nom que l'image)
- **Regle d'or : decrire ce qui varie, taire ce qui est permanent.** Ce qui est decrit s'attache
  aux mots de la legende ; ce qui est tu s'attache au mot declencheur.
- **Personnage / sujet** : ne decrivez NI cheveux, NI barbe, NI carrure, NI cicatrices. Decrivez le
  cadrage, l'angle, la pose, l'expression, la tenue (si elle change), le fond, l'eclairage.
  Exemple : `m41ne, 3D render, full body, back view, fighting stance, black vest, white background`.
- **Style** : c'est l'inverse, decrivez tout le contenu mais pas le style ni la technique.
- **Mot declencheur** : un mot invente avec des chiffres (m41ne), pas un vrai mot ni un nom connu.
- **Une image = un sujet, une vue.** Decoupez les planches a plusieurs vues, retirez les images
  ou un autre personnage apparait.
- Choisissez le type de LoRA dans le legendage automatique : la consigne s'adapte, et le bouton
  « Analyser le dossier » signale les legendes qui decrivent des traits permanents.
"""

AIDE_REGLAGES = """
| Reglage | Preconise | Source |
|---|---|---|
| Preset | int8 + offload 0.35 | defaut ai-toolkit pour ce modele, adapte a 24 Go |
| Resolutions | 1024 + 1536 | compromis memoire pour 24 Go |
| Pas | 1500 a 3000 | defaut ai-toolkit : 3000 |
| Learning rate | 0.0001 | defaut ai-toolkit |
| Rang LoRA | 16 | defaut ai-toolkit pour ce modele |
| Sauvegarde / apercus | 250 / 250 | defaut ai-toolkit |
| Apercus | 2048, 6 pas, guidance 1 | imposes par le modele (non reglables pour les 2 derniers) |

**Demarche conseillee** : 1. *Test rapide* (mesure vitesse et memoire, propose les reglages) ;
2. entrainement ; 3. choix du checkpoint dans *Tester un LoRA*.

**Selon l'objectif**
- **Sujet** (personnage, objet) : 15 a 40 images, 1500 a 2500 pas, rang 16, mot declencheur.
- **Style** : 30 a 100 images, 2000 a 3000 pas, rang 16 ou 32, pas de mot declencheur.

**Lire les apercus et la courbe**
- La perte de ce modele est tres bruitee (6 niveaux de bruit fixes) : seule la courbe lissee
  est lisible, et elle baisse peu. Les apercus restent le meilleur juge.
- *Rien ne change apres 1000 pas* : legendes trop pauvres, ou learning rate trop bas (essayez 0.0002).
- *Le sujet apparait puis l'image se degrade* (couleurs brulees, memes poses que le dataset) :
  sur-apprentissage. Prenez un checkpoint plus ancien, ou relancez avec moins de pas ou 0.00005.
- *Le sujet apparait meme sans le mot declencheur* : dataset trop peu varie ou trop de pas.

**Erreur memoire (CUDA out of memory)**, dans l'ordre : 1024 seule ; apercus en 1536 ou desactives ;
part en RAM 0.5 puis 0.7 ; rang 8.

*Ces valeurs sont les defauts d'ai-toolkit et des reperes generaux pour les LoRA. Elles n'ont pas
ete mesurees sur MiniMax-H3-Image avec une 3090 : le Test rapide les remplace par vos mesures.*
"""

DEFAULT_PROMPTS = (
    "photo detaillee d'un chat roux assis sur un rebord de fenetre, lumiere du matin\n"
    "affiche retro d'un TGV traversant un viaduc au coucher du soleil"
)


# ------------------------------------------------------------------ affichage
def _thumbs(items, size=384):
    """Vignettes PIL : marche pour n'importe quel dossier (Gradio refuse de
    servir des fichiers hors de son dossier de travail) et evite de charger
    des images 2048x2048 entieres dans le navigateur.
    items : chemins, ou tuples (chemin, legende)."""
    from PIL import Image

    out = []
    for item in items:
        path, label = item if isinstance(item, tuple) else (item, os.path.basename(item))
        try:
            with Image.open(path) as im:
                im.thumbnail((size, size))
                out.append((im.convert("RGB"), label))
        except Exception:
            pass
    return out


def _bar_html(title, d, color, energy=False):
    pct = int(d["pct"]) if d else 0
    detail = ""
    if d:
        detail = f"{d['n']} / {d['tot']}"
        if d.get("el"):
            detail += f" &nbsp;|&nbsp; ecoule {d['el']}"
        if d.get("eta"):
            detail += f" &nbsp;|&nbsp; reste {d['eta']}"
        if d.get("rate"):
            detail += f" &nbsp;|&nbsp; {d['rate']}"
    if energy:
        cost = core.energy_text((d or {}).get("eta"), i18n.lang_now())
        if cost:
            detail += f" &nbsp;|&nbsp; {cost}"
    name = (d or {}).get("desc") or ""
    return f"""
<div style="margin:6px 0">
  <div style="display:flex;justify-content:space-between;font-size:13px">
    <b>{title}</b><span>{pct} %</span></div>
  <div style="background:#2a2a2a;border-radius:6px;height:16px;overflow:hidden">
    <div style="width:{pct}%;height:100%;background:{color};transition:width .5s"></div></div>
  <div style="font-size:12px;opacity:.8;margin-top:2px">{name} {detail}</div>
</div>"""


def _meter(label, used, total, extra=""):
    pct = max(0, min(100, int(100 * used / total))) if total else 0
    color = "#e53935" if pct >= 96 else "#fb8c00" if pct >= 88 else "#7e57c2"
    return f"""
<div style="flex:1;min-width:180px">
  <div style="font-size:12px"><b>{label}</b> {used:.1f} / {total:.1f} Go {extra}</div>
  <div style="background:#2a2a2a;border-radius:5px;height:10px;overflow:hidden">
    <div style="width:{pct}%;height:100%;background:{color}"></div></div>
</div>"""


def _monitor_html():
    parts = []
    if MON.vram_used is not None and MON.vram_total:
        extra = f"(pic {MON.peak_all:.1f})" if MON.peak_all else ""
        if MON.gpu_util is not None:
            alert = int(core.load_settings().get("temp_alert") or 0)
            hot = alert and MON.gpu_temp is not None and MON.gpu_temp >= alert
            temp = (f"<b style='color:#e53935'>{MON.gpu_temp} °C</b>" if hot else f"{MON.gpu_temp} °C")
            extra += f" &nbsp;GPU {MON.gpu_util} % &nbsp;{temp}"
            if MON.fan is not None:
                extra += f" &nbsp;ventilateur {MON.fan} %"
            if MON.power_draw is not None:
                extra += f" &nbsp;{MON.power_draw:.0f}"
                extra += f" / {MON.power_limit:.0f} W" if MON.power_limit else " W"
        parts.append(_meter("VRAM", MON.vram_used, MON.vram_total, extra))
    if MON.ram_used and MON.ram_total:
        parts.append(_meter("RAM", MON.ram_used, MON.ram_total))
    if not parts:
        return ""
    return f"<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:8px'>{''.join(parts)}</div>"


TAB_OF_KIND = {"caption": "1. Dataset", "train": "2. Entrainement", "calib": "2. Entrainement",
               "test": "4. Tester un LoRA", "download": "Modeles"}


def render_status():
    s = RUN.snapshot()
    if s["running"]:
        html = (PANEL_CSS + f"<div style='font-size:14px'><span class='h3-spin'></span><b>EN COURS</b> "
                f"&nbsp; {s['label']} &nbsp;<span style='opacity:.8'>({_dur(s['elapsed'])})</span>"
                f" &nbsp;<span style='opacity:.7'>- details et journal dans l'onglet "
                f"{TAB_OF_KIND.get(s['kind'], '')}</span><br>Phase : <b>{s['phase']}</b></div>")
    else:
        html = "<div style='font-size:14px'><b>INACTIF</b> &nbsp;<span style='opacity:.7'>aucun job en cours</span></div>"
    if s["running"] and s["kind"] in RUN.CONTROLLED:
        st = core.db_get(RUN.job_name)
        if st and st["info"]:
            html += (f"<div style='font-size:13px;opacity:.85'>ai-toolkit : {st['info']}"
                     f"{' &nbsp;|&nbsp; ' + st['speed'] if st['speed'] else ''}</div>")
    if s["running"] and (s["train_bar"] or s["bar"]):
        html += _bar_html("Progression", s["train_bar"] or s["bar"], "#4caf50" if s["train_bar"] else "#2196f3",
                          energy=s["kind"] != "download")
    html += _monitor_html()
    extra = []
    if ORCH.thermal_message:
        extra.append(f"<b style='color:#e53935'>{ORCH.thermal_message}</b>")
    if ORCH.ram_message:
        extra.append(f"<b style='color:#e53935'>{ORCH.ram_message}</b>")
    if core.SWEEP.note and s["running"]:
        extra.append(core.SWEEP.note)
    if ORCH._thermo_note and s["running"]:
        extra.append(ORCH._thermo_note)
    elif ORCH.power_message:
        extra.append(ORCH.power_message)
    if ORCH.pause_at_step or ORCH.pause_at_time:
        extra.append(ORCH.schedule_text())
    if ORCH.queue:
        extra.append(f"File d'attente : {len(ORCH.queue)} job(s)"
                     + (" - SUSPENDUE" if ORCH.hold else ""))
    if ORCH.last_message and not s["running"]:
        extra.append(ORCH.last_message)
    if extra:
        html += "<div style='font-size:12px;opacity:.85;margin-top:6px'>" + "<br>".join(extra) + "</div>"
    return (html, ORCH.calib_report, ORCH.queue_text(),
            *render_caption_panel(), *render_train_panel(), *render_test_panel(), *render_download_panel())


# ------------------------------------------------------------------ panneaux par onglet
PANEL_CSS = """<style>
@keyframes h3spin{to{transform:rotate(360deg)}}
@keyframes h3stripe{from{background-position:0 0}to{background-position:40px 0}}
.h3-spin{display:inline-block;width:13px;height:13px;border:2px solid #9e9e9e;border-top-color:#ff7a00;
 border-radius:50%;animation:h3spin .8s linear infinite;vertical-align:-2px;margin-right:6px}
.h3-indet{height:14px;border-radius:6px;background:repeating-linear-gradient(45deg,#2196f3 0 10px,#64b5f6 10px 20px);
 background-size:40px 14px;animation:h3stripe 1s linear infinite}
.h3-steps{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0;font-size:12px}
.h3-step{padding:2px 8px;border-radius:10px;border:1px solid #9e9e9e55;opacity:.55}
.h3-step.done{opacity:.85;border-color:#4caf5088}
.h3-step.now{opacity:1;border-color:#ff7a00;font-weight:600}
</style>"""


def _dur(seconds):
    seconds = int(max(0, seconds))
    h, rest = divmod(seconds, 3600)
    m, s_ = divmod(rest, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s_:02d} s" if m else f"{s_} s")


def panel(kinds, idle_text):
    """(html, journal) pour l'onglet des jobs de ce type : etapes, animation, barre et ETA."""
    v = RUN.view(kinds)
    if v is None:
        return PANEL_CSS + f"<div style='opacity:.7;font-size:13px'>{idle_text}</div>", ""
    if not v["live"]:
        if v["stopped"]:
            icon, color = "&#10074;&#10074;", "#fb8c00"
        elif v["exit_code"] == 0:
            icon, color = "&#10004;", "#43a047"
        else:
            icon, color = "&#10008;", "#e53935"
        when = time.strftime("%H:%M", time.localtime(v["ended"]))
        html = (f"<div style='font-size:13px'><b style='color:{color}'>{icon}</b> <b>{v['label']}</b> "
                f"&nbsp;-&nbsp; {v['phase']} &nbsp;<span style='opacity:.75'>(fin a {when}, "
                f"duree {_dur(v['duration'])})</span></div>")
        spent = core.energy_summary(v.get("wh"), i18n.lang_now())
        if spent:
            html += f"<div style='font-size:12px;opacity:.8;margin-top:2px'>Consommation : {spent}</div>"
        return PANEL_CSS + html, v["log"]
    html = (f"<div style='font-size:14px'><span class='h3-spin'></span><b>{v['label']}</b> "
            f"&nbsp;<span style='opacity:.75'>en cours depuis {_dur(v['elapsed'])}</span></div>")
    stages = v["stages"]
    if stages:
        now = stages.index(v["stage"]) if v["stage"] in stages else 0
        chips = []
        for i, name in enumerate(stages):
            cls = "done" if i < now else "now" if i == now else ""
            mark = "&#10004; " if i < now else ""
            chips.append(f"<span class='h3-step {cls}'>{mark}{i + 1}. {name}</span>")
        html += "<div class='h3-steps'>" + "".join(chips) + "</div>"
    html += f"<div style='font-size:13px'>{v['phase']}</div>"
    if v["train_bar"]:
        html += _bar_html("Progression (pas)", v["train_bar"], "#4caf50", energy=True)
    if v["bar"]:
        title = v["bar"].get("desc") or v["stage"] or "Etape en cours"
        html += _bar_html(title, v["bar"], "#2196f3", energy=not v["train_bar"] and v.get("kind") != "download")
    if not v["train_bar"] and not v["bar"]:
        detail = f"etape en cours depuis {_dur(v['stage_elapsed'])}"
        if v["stage"] == "Legendage" and v["total"]:
            detail = f"0 / {v['total']} images - {detail}"
        if v.get("kind") != "download":
            cost = core.energy_text(None, i18n.lang_now())
            if cost:
                detail += f" &nbsp;|&nbsp; {cost}"
        html += (f"<div style='margin:6px 0'><div class='h3-indet'></div>"
                 f"<div style='font-size:12px;opacity:.8;margin-top:2px'>{detail} "
                 "(duree non mesurable a ce stade)</div></div>")
    return PANEL_CSS + html, v["log"]


def render_caption_panel():
    return panel(("caption",), "Aucun legendage lance depuis l'ouverture de l'interface.")


def render_train_panel():
    return panel(("train", "calib"), "Aucun entrainement ni Test rapide lance depuis l'ouverture de l'interface.")


def render_test_panel():
    return panel(("test",), "Aucun test lance depuis l'ouverture de l'interface.")


def render_download_panel():
    return panel(("download",), "Aucun telechargement lance depuis l'ouverture de l'interface.")


# ------------------------------------------------------------------ dataset
def dataset_info(folder, resolutions=None, lora_type=None, trigger="", extra=""):
    msg, paths = core.dataset_report(folder)
    msg += core.image_size_report(folder, resolutions)
    msg += core.caption_check(folder, lora_type or core.LORA_SUBJECT, trigger, extra)
    return msg, _thumbs(paths)


def dataset_upload(folder, files, resolutions=None, lora_type=None, trigger="", extra=""):
    import shutil

    if not folder:
        return "Indiquez d'abord un dossier.", []
    os.makedirs(folder, exist_ok=True)
    for f in files or []:
        src = f if isinstance(f, str) else f.name
        shutil.copy2(src, os.path.join(folder, os.path.basename(src)))
    return dataset_info(folder, resolutions, lora_type, trigger, extra)


def fill_captions(folder, text):
    if not folder or not os.path.isdir(folder) or not (text or "").strip():
        return "Dossier ou texte manquant.", []
    n = core.fill_captions(folder, text)
    msg, gal = dataset_info(folder)
    return f"{n} legende(s) creee(s).\n" + msg, gal


# ------------------------------------------------------------------ audit du dataset
def audit_run(folder, progress=gr.Progress()):
    def cb(i, n, name):
        progress((i, max(1, n)), desc=i18n.tr(f"Analyse : {name}"), unit="images")
    try:
        report, gallery, fixes = audit.audit(folder, cb)
    except Exception as exc:  # noqa: BLE001 - modele absent, image illisible...
        return f"Audit impossible : {type(exc).__name__}: {str(exc)[:200]}", [], [], gr.update(visible=False)
    return report, gallery, fixes, gr.update(visible=bool(fixes))


def audit_fix(folder, fixes):
    if not fixes:
        return "Aucune legende a completer.", gr.update(visible=False)
    return audit.add_background(folder, fixes), gr.update(visible=False)


# ------------------------------------------------------------------ correction des legendes
CAPFIX_HEADERS = ["Appliquer", "Fichier", "Actuelle", "Proposee"]


def capfix_propose(folder, lora_type, trigger, extra):
    rows, msg = capfix.proposals(folder, lora_type or core.LORA_SUBJECT, trigger, extra)
    headers = [i18n.tr(h) for h in CAPFIX_HEADERS]   # une mise a jour du tableau reprend ses en-tetes
    return (gr.update(value={"headers": headers, "data": rows}, visible=bool(rows)), msg,
            gr.update(visible=bool(rows)))


def capfix_apply(folder, table):
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    if isinstance(table, dict):
        rows = table.get("data") or []
    return capfix.apply(folder, rows), gr.update(visible=False), gr.update(visible=False)


# ------------------------------------------------------------------ rognage des textes
def crop_scan(folder, keep, margin, conf, progress=gr.Progress()):
    def step(i, n, name):
        progress((i, max(1, n)), desc=i18n.tr(f"Analyse : {name}"), unit="images")

    report, gallery, names = crop.analyze_folder(folder, core.MODELS, min_keep=float(keep) / 100,
                                                 margin_pct=float(margin), min_conf=float(conf), progress=step)
    return report, gallery, gr.update(choices=names, value=names)


def crop_apply(folder, picked):
    msg = crop.apply_crops(folder, picked or [])
    left = [n for n, it in crop._last["items"].items() if it["crop"]]
    return msg, gr.update(choices=left, value=left)


def crop_restore(folder):
    return crop.restore_originals(folder), gr.update(choices=[], value=[]), []


# ------------------------------------------------------------------ entrainement
def _params(args):
    return dict(zip(core.PARAM_KEYS, args))


def preview(*args):
    return core.dump_yaml(core.build_config(_params(args))[1])


def start_training(*args):
    msg, name = core.start_training(_params(args))
    if name:
        return msg, gr.update(choices=core.list_jobs(), value=name)
    return msg, gr.update()


def reg_generate(count, *args):
    return core.start_reg_generation(_params(args), count)


def enqueue_training(*args):
    return core.start_training(_params(args), queue=True)[0]


def start_calibration(*args):
    return core.start_calibration(_params(args))


def save_settings_now(*args):
    P = _params(args)
    P["name"] = core.safe_name(P["name"])
    core.save_job_settings(P)
    return f"Reglages enregistres : {P['name']}", gr.update(choices=core.list_job_settings(), value=P["name"])


PROCESS_FIELDS = ["resolutions", "rank", "lr", "steps", "save_every", "sample_every",
                  "optimizer", "ema", "adapter", "likeness", "sampler"]


def apply_process(process, folder):
    """Remplit les champs concernes d'apres le procede choisi (Standard : ne touche a rien)."""
    values = core.process_values(process, folder)
    if values is None:
        return [gr.update()] * len(PROCESS_FIELDS) + [
            "Procede Standard : vos reglages actuels sont conserves."]
    n = core.count_images(folder)
    note = (f"{values['steps']} pas pour {n} images (30 passages, 300 pas au minimum)." if n else
            "Dossier du dataset introuvable : 600 pas par defaut, a ajuster (30 x nombre d'images).")
    return [values[k] for k in PROCESS_FIELDS] + [
        f"Procede applique. {note} Lancez le Test rapide pour mesurer la vitesse reelle."]


def load_settings_now(name):
    P = core.load_job_settings(name) if name else None
    if not P:
        return [gr.update()] * len(core.PARAM_KEYS) + ["Aucun reglage enregistre sous ce nom."]
    if P.get("preset") not in (core.PRESET_INT8, core.PRESET_W4A8):
        P["preset"] = core.PRESET_INT8
    P["resolutions"] = [str(r) for r in P.get("resolutions") or ["1024"]]
    P.setdefault("process", core.PROC_STANDARD)        # reglages enregistres avant ces options
    P.setdefault("optimizer", core.OPT_ADAMW)
    P.setdefault("ema", core.EMA_OFF)
    P.setdefault("adapter", core.ADAPTER_OSTRIS)
    P.setdefault("likeness", False)
    P.setdefault("sampler", core.SAMPLER_EULER)
    return [P.get(k, gr.update()) for k in core.PARAM_KEYS] + [f"Reglages charges : {name}"]


def prompt_check(trigger, prompts, no_sample=False, control=True):
    """Sous les prompts d'apercu : ecarts avec le mot declencheur, et prompt temoin (debordement)."""
    if no_sample:
        return ""
    issues = core.prompt_trigger_issues(trigger, prompts)
    out = [f"<span style='color:#e65100'>&#9888; {x[0].upper() + x[1:]}</span>  " for x in issues]
    ctl, auto = core.control_prompt(prompts, trigger)
    if ctl and not auto:
        out.append(f"<span style='color:#2e7d32'>&#10004; Prompt temoin sans mot declencheur : '{ctl}'</span>  ")
    elif ctl and control:
        out.append(f"<span style='color:#1565c0'>&#9432; Prompt temoin ajoute automatiquement aux apercus : "
                   f"'{ctl}'</span>  ")
    elif ctl:
        out.append("<span style='color:#e65100'>&#9888; Aucun prompt sans mot declencheur : le debordement ne "
                   "pourra pas etre mesure. Cochez la case ci-dessus ou ecrivez un prompt sans le mot.</span>  ")
    return "\n".join(out)


def _newest(paths):
    paths = [p for p in paths if os.path.exists(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def restore_page():
    """A l'ouverture d'une page (changement de langue, actualisation) : formulaire
    d'entrainement rempli avec le job en cours (sinon le dernier lance ou enregistre),
    et ce job selectionne dans Resultats et Tester un LoRA."""
    running = (RUN.job_name if RUN.running() and RUN.kind == "train" and RUN.job_name
               and not RUN.job_name.startswith("_") else "")
    saved = core.list_job_settings()
    form_job = running
    if not form_job and saved:
        newest = _newest([core.job_settings_path(n) for n in saved])
        form_job = next((n for n in saved if core.job_settings_path(n) == newest), "")
    fields = load_settings_now(form_job)[:-1] if form_job else [gr.update()] * len(core.PARAM_KEYS)
    jobs = core.list_jobs()
    pick = running if running in jobs else None
    if not pick and jobs:
        newest = _newest([os.path.join(core.OUTPUT, j) for j in jobs])
        pick = os.path.basename(newest) if newest else None
    return fields + [gr.update(choices=saved, value=form_job if form_job in saved else None),
                     gr.update(choices=jobs, value=pick), gr.update(choices=jobs, value=pick)]


# ------------------------------------------------------------------ resultats
def loss_svg(job):
    """Courbe de perte lissee en SVG, echelle Y recalee sur son minimum et son maximum,
    reperes verticaux aux pas des checkpoints."""
    rows = core.LossLog.read(os.path.join(core.OUTPUT, job, "pertes.csv")) if job else []
    if len(rows) < 2:
        return "<div style='opacity:.7;font-size:13px'>Courbe de perte : pas encore de donnees.</div>"
    W, H, L, R, T, B = 1000, 300, 52, 12, 30, 34
    steps = [r[0] for r in rows]
    # moyenne exponentielle demarree sur la moyenne des 20 premiers pas (ni collee au premier
    # pas, ni nerveuse au debut : le debut n'ecrase pas l'echelle)
    head = [r[1] for r in rows[:20]]
    alpha, ema, smooth = 0.04, sum(head) / len(head), []
    for _st, v in rows:
        ema += alpha * (v - ema)
        smooth.append(ema)
    lo, hi = min(smooth), max(smooth)
    pad = max((hi - lo) * 0.08, abs(hi) * 0.002, 1e-6)
    ymin, ymax = lo - pad, hi + pad
    x0, x1 = steps[0], max(steps[-1], steps[0] + 1)

    def X(st):
        return L + (st - x0) / (x1 - x0) * (W - L - R)

    def Y(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - B)

    span_y = ymax - ymin
    dec = 2 if span_y >= 0.2 else 3 if span_y >= 0.02 else 4
    k = len(rows) // 1500 + 1  # allege les longs entrainements
    pts = " ".join(f"{X(st):.1f},{Y(v):.1f}" for st, v in list(zip(steps, smooth))[::k])
    out = [f"<svg viewBox='0 0 {W} {H}' style='width:100%;height:auto;font-family:sans-serif' "
           "xmlns='http://www.w3.org/2000/svg'>"]
    for i in range(5):  # grille + graduations Y
        v = ymin + span_y * i / 4
        out.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='currentColor' "
                   f"stroke-opacity='.15'/><text x='{L - 6}' y='{Y(v) + 4:.1f}' font-size='11' "
                   f"text-anchor='end' fill='currentColor'>{v:.{dec}f}</text>")
    span = x1 - x0
    tick = next(t for t in (50, 100, 250, 500, 1000, 2500, 5000, 10000, 10 ** 9) if span / t <= 12)
    st = (x0 // tick + 1) * tick
    while st <= x1:  # graduations X
        out.append(f"<text x='{X(st):.1f}' y='{H - B + 15}' font-size='11' text-anchor='middle' "
                   f"fill='currentColor'>{st}</text>")
        st += tick
    for i, cs in enumerate(c for c in core.checkpoint_steps(job) if x0 <= c <= x1):
        out.append(f"<line x1='{X(cs):.1f}' x2='{X(cs):.1f}' y1='{T}' y2='{H - B}' stroke='#43a047' "
                   f"stroke-dasharray='4 3'/><text x='{X(cs):.1f}' y='{T - 4 - (i % 2) * 11}' font-size='10' "
                   f"text-anchor='middle' fill='#43a047'>{cs}</text>")
    out.append(f"<polyline points='{pts}' fill='none' stroke='#ff8f00' stroke-width='2.5'/>")
    out.append(f"<text x='{(L + W - R) / 2}' y='{H - 4}' font-size='11' text-anchor='middle' "
               "fill='currentColor'>pas</text></svg>")
    legend = ("<div style='font-size:12px;margin-top:2px'>"
              "<span style='color:#ff8f00'>&#9472;&#9472;</span> perte lissee &nbsp; "
              "<span style='color:#43a047'>&#9482;</span> checkpoints disponibles &nbsp;|&nbsp; "
              f"lissee au pas {steps[-1]} : <b>{smooth[-1]:.3f}</b> &nbsp;|&nbsp; "
              f"min {lo:.3f} - max {hi:.3f}</div>")
    return "<div><b style='font-size:13px'>Perte</b>" + "".join(out) + legend + "</div>"


def _axis(lo, hi, rel=0.08):
    pad = max((hi - lo) * rel, abs(hi) * 0.01, 1e-6)
    return lo - pad, hi + pad


def perf_svg(job):
    """Puissance (et limite), temperature et vitesse au fil des pas, puis un tableau par palier
    de puissance pour reperer la limite la plus basse qui ne ralentit pas l'entrainement."""
    rows = core.PerfLog.read(job) if job else []
    if len(rows) < 3:
        return ("<div style='opacity:.7;font-size:13px'>Puissance et vitesse : pas encore de mesures "
                "(relevees pendant l'entrainement, par tranches de 5 pas sans apercu ni sauvegarde).</div>")
    steps = [r[1] for r in rows]
    watts, temps, limits, spi = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows]
    W, L, R = 1000, 52, 52
    T1, H1 = 26, 150          # panneau haut : puissance + temperature
    T2, H2 = T1 + H1 + 34, 110  # panneau bas : vitesse
    H = T2 + H2 + 34
    x0, x1 = steps[0], max(steps[-1], steps[0] + 1)

    def X(st):
        return L + (st - x0) / (x1 - x0) * (W - L - R)

    lim_ok = [v for v in limits if v > 0]
    w_lo, w_hi = _axis(min(watts + lim_ok), max(watts + lim_ok))
    t_lo, t_hi = _axis(min(temps), max(temps))
    s_lo, s_hi = _axis(min(spi), max(spi))

    def Y(v, lo, hi, top, h):
        return top + (hi - v) / (hi - lo) * h

    def line(vals, lo, hi, top, h, color, width=2.0, dash=""):
        pts = " ".join(f"{X(st):.1f},{Y(v, lo, hi, top, h):.1f}" for st, v in zip(steps, vals))
        return (f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{width}'"
                + (f" stroke-dasharray='{dash}'" if dash else "") + "/>")

    out = [f"<svg viewBox='0 0 {W} {H}' style='width:100%;height:auto;font-family:sans-serif' "
           "xmlns='http://www.w3.org/2000/svg'>"]
    for top, h, lo, hi, color, unit, dec, side in (
            (T1, H1, w_lo, w_hi, "#e65100", "W", 0, "l"), (T1, H1, t_lo, t_hi, "#c62828", "°C", 0, "r"),
            (T2, H2, s_lo, s_hi, "#1e88e5", "s", 2, "l")):
        for i in range(4):
            v = lo + (hi - lo) * i / 3
            y = Y(v, lo, hi, top, h)
            if side == "l":
                out.append(f"<line x1='{L}' x2='{W - R}' y1='{y:.1f}' y2='{y:.1f}' stroke='currentColor' "
                           "stroke-opacity='.12'/>")
            xt, anchor = (L - 6, "end") if side == "l" else (W - R + 6, "start")
            out.append(f"<text x='{xt}' y='{y + 4:.1f}' font-size='11' text-anchor='{anchor}' "
                       f"fill='{color}'>{v:.{dec}f} {unit}</text>")
    if lim_ok:
        out.append(line([v if v > 0 else w_hi for v in limits], w_lo, w_hi, T1, H1, "#9e9e9e", 1.5, "6 4"))
    out.append(line(watts, w_lo, w_hi, T1, H1, "#e65100"))
    out.append(line(temps, t_lo, t_hi, T1, H1, "#c62828", 1.5))
    out.append(line(spi, s_lo, s_hi, T2, H2, "#1e88e5"))
    span = x1 - x0
    tick = next(t for t in (25, 50, 100, 250, 500, 1000, 2500, 5000, 10 ** 9) if span / t <= 12)
    st = (x0 // tick + 1) * tick
    while st <= x1:
        out.append(f"<text x='{X(st):.1f}' y='{H - 18}' font-size='11' text-anchor='middle' "
                   f"fill='currentColor'>{st}</text>")
        st += tick
    out.append(f"<text x='{(L + W - R) / 2}' y='{H - 3}' font-size='11' text-anchor='middle' "
               "fill='currentColor'>pas</text></svg>")
    legend = ("<div style='font-size:12px;margin:2px 0 8px'>"
              "<span style='color:#e65100'>&#9472;&#9472;</span> puissance du GPU &nbsp; "
              "<span style='color:#9e9e9e'>&#9476;&#9476;</span> limite de puissance &nbsp; "
              "<span style='color:#c62828'>&#9472;&#9472;</span> temperature &nbsp; "
              "<span style='color:#1e88e5'>&#9472;&#9472;</span> secondes par pas (plus bas = plus rapide)</div>")

    # paliers de 10 W de puissance mesuree
    bins = {}
    for w, t, sp in zip(watts, temps, spi):
        bins.setdefault(int(round(w / 10.0)) * 10, []).append((w, t, sp))
    stats = []
    for b, vals in sorted(bins.items()):
        n = len(vals)
        mw, mt = sum(v[0] for v in vals) / n, sum(v[1] for v in vals) / n
        ms = sorted(v[2] for v in vals)[n // 2]   # mediane : insensible aux pas lents isoles
        stats.append((b, mw, mt, ms, mw * ms / 3600, n))
    solid = [x for x in stats if x[5] >= 3] or stats
    best = min(solid, key=lambda x: x[3])
    pick = min((x for x in solid if x[3] <= best[3] * 1.03), key=lambda x: x[1])
    price = float(core.load_settings().get("kwh_price") or 0)
    cur = core.currency_for(i18n.lang_now())
    head = ("<tr><th>Puissance mesuree</th><th>s/pas</th><th>Temperature</th><th>Wh par pas</th>"
            + ("<th>Cout pour 1000 pas</th>" if price > 0 else "") + "<th>Mesures</th><th></th></tr>")
    body = []
    # un palier mesure une seule fois chevauche en general un changement de limite : on l'ecarte
    shown = [x for x in stats if x[5] >= 2 or x in (best, pick)] if len(stats) > 2 else stats
    for b, mw, mt, ms, wh, n in shown:
        mark = []
        if (b, n) == (best[0], best[5]):
            mark.append("le plus rapide")
        if (b, n) == (pick[0], pick[5]):
            mark.append("seuil conseille")
        style = " style='background:rgba(67,160,71,.18);font-weight:600'" if (b, n) == (pick[0], pick[5]) else ""
        body.append(f"<tr{style}><td>{mw:.0f} W</td><td>{ms:.2f}</td><td>{mt:.0f} °C</td><td>{wh:.3f}</td>"
                    + (f"<td>{core.fmt_money(wh * price, cur)}</td>" if price > 0 else "")
                    + f"<td>{n}</td><td>{', '.join(mark)}</td></tr>")
    table = ("<table style='font-size:12px;border-collapse:collapse;width:100%' cellpadding='4'>"
             + head + "".join(body) + "</table>")
    if len(stats) < 2:
        verdict = ("Une seule puissance mesuree pour l'instant : faites varier la limite (onglet Outils, "
                   "ou le thermostat) pendant l'entrainement pour comparer.")
    elif pick is best:
        verdict = (f"Seuil conseille : environ {pick[1]:.0f} W, la puissance la plus rapide mesuree ({pick[2]:.0f} °C). "
                   "Essayez une limite plus basse pour voir si la vitesse tient.")
    else:
        slower = (pick[3] / best[3] - 1) * 100
        verdict = (f"Seuil conseille : environ {pick[1]:.0f} W, soit {slower:.0f} % plus lent que le plus rapide "
                   f"pour {best[1] - pick[1]:.0f} W de moins, a {pick[2]:.0f} °C.")
    return ("<div><b style='font-size:13px'>Puissance, vitesse et temperature</b>" + "".join(out) + legend
            + table + f"<div style='font-size:13px;margin-top:6px'><b>{verdict}</b></div></div>")


def likeness_svg(job):
    """Ressemblance au dataset et fidelite au prompt des apercus, par pas, et checkpoint conseille."""
    status = likeness.SCORER.status(job) if job else ""
    rows = likeness.summary(job) if job else []
    note = f"<div style='font-size:12px;opacity:.8;margin-top:4px'>{status}</div>" if status else ""
    if len(rows) < 2:
        return ("<div style='opacity:.7;font-size:13px'>Ressemblance : pas encore assez d'apercus mesures "
                "(au moins deux series d'apercus).</div>" + note)
    steps = [r[0] for r in rows]
    dino, clip = [r[1] for r in rows], [r[2] for r in rows]
    W, H, L, R, T, B = 1000, 260, 52, 52, 30, 34
    x0, x1 = steps[0], max(steps[-1], steps[0] + 1)

    def X(st):
        return L + (st - x0) / (x1 - x0) * (W - L - R)

    d_lo, d_hi = _axis(min(dino), max(dino))
    c_lo, c_hi = _axis(min(clip), max(clip))

    def Y(v, lo, hi):
        return T + (hi - v) / (hi - lo) * (H - T - B)

    rec = likeness.recommend(job, rows)
    cks = set(likeness.ck_steps(job, rows))
    lk = likeness.leaks(rows)
    has_ctl = any(r[4] is not None for r in rows)
    out = [f"<svg viewBox='0 0 {W} {H}' style='width:100%;height:auto;font-family:sans-serif' "
           "xmlns='http://www.w3.org/2000/svg'>"]
    for lo, hi, color, side in ((d_lo, d_hi, "#2e7d32", "l"), (c_lo, c_hi, "#6a1b9a", "r")):
        for i in range(4):
            v = lo + (hi - lo) * i / 3
            y = Y(v, lo, hi)
            if side == "l":
                out.append(f"<line x1='{L}' x2='{W - R}' y1='{y:.1f}' y2='{y:.1f}' stroke='currentColor' "
                           "stroke-opacity='.12'/>")
            xt, anchor = (L - 6, "end") if side == "l" else (W - R + 6, "start")
            out.append(f"<text x='{xt}' y='{y + 4:.1f}' font-size='11' text-anchor='{anchor}' "
                       f"fill='{color}'>{v:.1f}</text>")
    if rec:
        xr = X(rec[0][0])
        out.append(f"<line x1='{xr:.1f}' x2='{xr:.1f}' y1='{T}' y2='{H - B}' stroke='#f9a825' stroke-width='3' "
                   "stroke-opacity='.6'/>")
    for vals, lo, hi, color in ((dino, d_lo, d_hi, "#2e7d32"), (clip, c_lo, c_hi, "#6a1b9a")):
        pts = " ".join(f"{X(st):.1f},{Y(v, lo, hi):.1f}" for st, v in zip(steps, vals))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        for st, v in zip(steps, vals):
            out.append(f"<circle cx='{X(st):.1f}' cy='{Y(v, lo, hi):.1f}' r='3.5' fill='{color}'/>")
    for st in steps:
        out.append(f"<text x='{X(st):.1f}' y='{H - B + 15}' font-size='11' text-anchor='middle' "
                   f"fill='currentColor'>{st}</text>")
        if lk.get(st, 0) >= likeness.LEAK_SOME:   # repere rouge : debordement
            color = "#c62828" if lk[st] >= likeness.LEAK_HIGH else "#ef6c00"
            out.append(f"<rect x='{X(st) - 7:.1f}' y='{T - 14}' width='14' height='8' rx='2' fill='{color}'/>")
    out.append(f"<text x='{(L + W - R) / 2}' y='{H - 3}' font-size='11' text-anchor='middle' "
               "fill='currentColor'>pas</text></svg>")
    legend = ("<div style='font-size:12px;margin:2px 0 8px'>"
              "<span style='color:#2e7d32'>&#9472;&#9472;</span> ressemblance au dataset &nbsp; "
              "<span style='color:#6a1b9a'>&#9472;&#9472;</span> fidelite au prompt &nbsp; "
              "<span style='color:#f9a825'>&#9474;</span> checkpoint conseille &nbsp; "
              "<span style='color:#c62828'>&#9644;</span> debordement</div>")
    head = ("<tr><th>Pas</th><th>Ressemblance</th><th>Fidelite au prompt</th>"
            + ("<th>Debordement</th>" if has_ctl else "") + "<th>Checkpoint</th><th></th></tr>")
    body = []
    for st, d, c, _n, _t in rows:
        is_rec = bool(rec) and st == rec[0][0]
        style = " style='background:rgba(249,168,37,.22);font-weight:600'" if is_rec else ""
        cell = ""
        if has_ctl:
            v = lk.get(st)
            col = ("#c62828" if v is not None and v >= likeness.LEAK_HIGH else
                   "#ef6c00" if v is not None and v >= likeness.LEAK_SOME else "inherit")
            cell = f"<td style='color:{col}'>{'-' if v is None else f'{v * 100:.0f} %'}</td>"
        body.append(f"<tr{style}><td>{st}</td><td>{d:.1f}</td><td>{c:.1f}</td>{cell}"
                    f"<td>{'oui' if st in cks else '-'}</td><td>{'conseille' if is_rec else ''}</td></tr>")
    table = ("<table style='font-size:12px;border-collapse:collapse;width:100%' cellpadding='4'>"
             + head + "".join(body) + "</table>")
    if not rec:
        verdict = "Pas encore de checkpoint correspondant aux apercus mesures."
    elif rec[2]:
        verdict = (f"Checkpoint conseille : pas {rec[0][0]} (meilleur compromis). Tous les checkpoints "
                   "suivent moins bien le prompt qu'au debut : le LoRA est peut-etre trop entraine.")
    else:
        verdict = (f"Checkpoint conseille : pas {rec[0][0]}, le plus ressemblant parmi ceux qui suivent "
                   "encore le prompt comme au debut.")
    leak_txt = ""
    if not has_ctl:
        leak_txt = ("Pas de prompt temoin sans mot declencheur dans ces apercus : le debordement n'est pas mesure "
                    "(case 'Ajouter un prompt temoin' de l'onglet Entrainement, pour le prochain job).")
    elif lk:
        first = next((st for st in sorted(lk) if lk[st] >= likeness.LEAK_HIGH), None)
        if first is not None:
            leak_txt = (f"Debordement fort a partir du pas {first} : le personnage apparait aussi sans mot "
                        "declencheur. Ces checkpoints sont ecartes du conseil.")
    leak_html = f"<div style='font-size:12px;margin-top:4px;color:#c62828'>{leak_txt}</div>" if leak_txt else ""
    return ("<div><b style='font-size:13px'>Ressemblance et fidelite au prompt</b>" + "".join(out) + legend
            + table + f"<div style='font-size:13px;margin-top:6px'><b>{verdict}</b></div>" + leak_html + note
            + "</div>")


def _results_sig(job):
    files = core.job_samples(job) + core.job_checkpoints(job)
    likeness.SCORER.request(job)
    files_sig = likeness.SCORER.status(job)
    for extra in ("pertes.csv", "gpu_perf.csv", likeness.CACHE):
        path = os.path.join(core.OUTPUT, job, extra)
        if os.path.isfile(path):
            files.append(path)
    return files_sig + "|".join(f"{os.path.basename(p)}:{os.path.getmtime(p):.0f}" for p in files)


def job_results(job):
    if not job:
        return [], "", gr.update(choices=[], value=None), loss_svg(""), perf_svg(""), likeness_svg("")
    cks = core.job_checkpoints(job)
    listing = "\n".join(f"{p}   ({os.path.getsize(p) / 1e6:.0f} Mo)" for p in cks)
    names = [os.path.basename(p) for p in cks]
    return (_thumbs(core.job_samples(job), 1024), listing or "Aucun checkpoint pour l'instant.",
            gr.update(choices=names, value=names[0] if names else None), loss_svg(job), perf_svg(job),
            likeness_svg(job))


def auto_results(job, last_sig):
    """Toutes les 5 s : ne recharge que si un fichier du job a change."""
    if not job:
        return (gr.update(),) * 6 + (last_sig,)
    sig = _results_sig(job)
    if sig == last_sig:
        return (gr.update(),) * 6 + (last_sig,)
    return (*job_results(job), sig)


def open_job_folder(job):
    d = os.path.join(core.OUTPUT, job) if job else core.OUTPUT
    if core.IS_WIN and os.path.isdir(d):
        os.startfile(d)  # noqa: S606 - dossier local de l'utilisateur
    return d


def send_to_generator(job, ck_name):
    if not job or not ck_name:
        return "Selectionnez un job et un checkpoint."
    return gen.GEN.send_lora(os.path.join(core.OUTPUT, job, ck_name))


# ------------------------------------------------------------ generateur
def gen_state_text(folder):
    folder = (folder or "").strip()
    if not gen.is_generator(folder):
        return ("Generateur introuvable : indiquez le dossier MiniMaxH3-Portable (celui qui contient run.bat "
                "et le dossier app).")
    st = gen.state(folder)
    lines = [f"Dossier : {folder}",
             "Environnement Python : installe" if st["installed"] else
             "Environnement Python : absent (lancez install.bat du generateur)",
             "Modeles convertis : presents" if st["prepared"] else
             "Modeles convertis : absents (lancez prepare.bat du generateur, environ 39 Go)",
             "Licence : acceptee" if st["license"] else "Licence : a accepter (case ci-dessous)",
             f"Port : {st['port']}"]
    return "\n".join(lines)


def gen_frame_html(ready):
    if not ready:
        return ("<div style='padding:24px;opacity:.7'>Le generateur s'affichera ici une fois demarre et "
                "pret.</div>")
    url = f"http://127.0.0.1:{gen.GEN.port}/"
    return (f"<div style='text-align:right;margin-bottom:4px'><a href='{url}' target='_blank'>Ouvrir dans un "
            f"onglet</a></div><iframe src='{url}' style='width:100%;height:1250px;border:1px solid "
            f"var(--border-color-primary);border-radius:8px' allow='clipboard-write'></iframe>")


def gen_is_ready():
    return bool(gen.GEN.ready and gen.GEN.status().startswith("Pret"))


def gen_tick(last):
    ready = gen_is_ready()
    frame = gr.update(value=gen_frame_html(ready)) if ready != last else gr.update()
    return gen.GEN.status(), gen.GEN.log_text(), frame, ready


def gen_start(folder):
    return gen.GEN.start(folder, i18n.lang_now())


def gen_license(folder, accepted):
    folder = (folder or "").strip()
    if not gen.is_generator(folder):
        return "Generateur introuvable : indiquez le dossier MiniMaxH3-Portable (celui qui contient run.bat et le dossier app)."
    gen.write_config(folder, license_ack=bool(accepted))
    core.save_settings(gen_dir=folder)
    return gen_state_text(folder)


def gen_job_choices(job):
    cks = core.job_checkpoints(job) if job else []
    names = [os.path.basename(p) for p in cks]
    value = names[0] if names else None
    try:
        best = likeness.recommend(job) if job else None
    except Exception:  # noqa: BLE001 - conseil facultatif
        best = None
    if best:
        tag = f"_{int(best[0][0]):09d}.safetensors"
        value = next((n for n in names if n.endswith(tag)), value)
    return gr.update(choices=names, value=value)


def copy_to_comfy(job, ck_name):
    if not job or not ck_name:
        return "Selectionnez un job et un checkpoint."
    return core.copy_to_comfy(os.path.join(core.OUTPUT, job, ck_name))


# ------------------------------------------------------------------ test de LoRA
def test_choices(job):
    names = [os.path.basename(p) for p in core.job_checkpoints(job)] if job else []
    return gr.update(choices=names, value=[])


def start_lora_test(job, cks, prompts, size, seed):
    return core.start_lora_test(job, cks, prompts, size, seed)


def auto_test_results(job, last_sig):
    if not job:
        return gr.update(), last_sig
    items = core.lora_test_results(job)
    sig = "|".join(f"{p}:{os.path.getmtime(p):.0f}" for p, _ in items)
    if sig == last_sig:
        return gr.update(), last_sig
    return _thumbs(items, 1024), sig


# ------------------------------------------------------------------ outils
def save_tools(sound, topic, comfy, awake, t_alert, t_pause, watts, th_on=False, th_target=72, th_min=150,
               kwh_price=0, currency="", likeness_on=True, leak_pause=False, sweep_on=False, sweep_min=200,
               sweep_step=30):
    t_alert, t_pause, watts = int(t_alert or 0), int(t_pause or 0), int(watts or 0)
    th_target, th_min = int(th_target or 72), int(th_min or 150)
    if th_on:
        if not 50 <= th_target <= 85:
            return "Temperature cible du thermostat : entre 50 et 85 °C."
        if t_alert and th_target >= t_alert:
            return "La cible du thermostat doit etre inferieure au seuil d'alerte."
        if watts and th_min >= watts:
            return "Le plancher du thermostat doit etre inferieur a la limite de puissance (son plafond)."
    if t_alert and t_pause and t_pause <= t_alert:
        return "Le seuil de pause doit etre superieur au seuil d'alerte."
    note = ""
    if watts:
        info = core.power_info()
        if info and not info["min"] <= watts <= info["max"]:
            return f"Limite de puissance hors plage : {info['min']:.0f} a {info['max']:.0f} W."
    core.save_settings(sound=bool(sound), ntfy_topic=(topic or "").strip(),
                       comfy_loras_dir=(comfy or "").strip(), keep_awake=bool(awake),
                       temp_alert=t_alert, temp_pause=t_pause, power_limit_w=watts,
                       thermo_on=bool(th_on), thermo_target=th_target, thermo_min_w=th_min,
                       kwh_price=max(0.0, float(kwh_price or 0)), currency=currency or "",
                       likeness_on=bool(likeness_on), leak_pause=bool(leak_pause), sweep_on=bool(sweep_on),
                       sweep_min_w=max(100, int(sweep_min or 200)), sweep_step_w=max(10, int(sweep_step or 30)))
    if th_on:
        return ("Reglages enregistres. Thermostat actif : il regule pendant les jobs"
                + ("." if core.is_admin() else
                   ", mais il lui faut les droits administrateur (2_LANCER_UI_ADMIN.bat)."))
    if watts and RUN.running():
        # un job tourne : la nouvelle limite s'applique tout de suite, sans l'interrompre
        note = " " + core.apply_power_now()
    elif watts:
        note = " La limite sera appliquee au lancement du prochain job."
    return "Reglages enregistres." + note


def apply_power_field(watts):
    """'Appliquer maintenant' : enregistre la valeur saisie dans le champ, puis l'applique."""
    watts = int(watts or 0)
    if watts <= 0:
        return "Saisissez une limite en watts dans le champ ci-dessus."
    info = core.power_info()
    if info and not info["min"] <= watts <= info["max"]:
        return f"Limite de puissance hors plage : {info['min']:.0f} a {info['max']:.0f} W."
    core.save_settings(power_limit_w=watts)
    return core.apply_power_now()


def test_notification():
    core.notify("Test H3-Image LoRA", "Ceci est une notification de test.")
    return "Notification envoyee (son et/ou ntfy selon vos reglages)."


# ------------------------------------------------------------------ interface
_cfg = core.load_settings()

def build(code):
    """Construit l'interface dans une langue (une page par langue : /en/, /fr/, ...)."""
    with gr.Blocks(title="H3-Image LoRA - RTX 3090") as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                gr.Markdown("## LoRA MiniMax-H3-Image &nbsp;·&nbsp; ai-toolkit &nbsp;·&nbsp; RTX 3090 24 Go")
            with gr.Column(scale=1, min_width=150):
                lang_sel = gr.Dropdown([(name, c) for c, name in i18n.LANGS], value=code,
                                       show_label=False, container=False, interactive=True)
        lang_sel.input(None, lang_sel, None, js=i18n.SELECTOR_JS)

        with gr.Row():
            with gr.Column(scale=3):
                status = gr.HTML()
            with gr.Column(scale=1, min_width=180):
                pause_btn = gr.Button("Pause (sauvegarder l'etat)", variant="primary")
                sample_btn = gr.Button("Apercu maintenant")
                stop_btn = gr.Button("Arreter sans sauvegarder", variant="stop")
        msg = gr.Textbox(label="Message", lines=1, interactive=False)

        # ============================================================ 1. Dataset
        with gr.Tab("1. Dataset"):
            with gr.Accordion("Aide : preparer un bon dataset", open=False):
                gr.Markdown(AIDE_DATASET)
            ds_folder = gr.Textbox(label="Dossier du dataset", value=os.path.join(core.DATASETS, "mon_dataset"),
                                   info="Un dossier par LoRA. Preconise : 15 a 40 images (style : 30 a 100), "
                                        "nettes, variees, cote court >= 1024 px. Sans espaces dans le chemin.")
            with gr.Row():
                ds_files = gr.File(label="Ajouter des images / .txt", file_count="multiple")
                with gr.Column():
                    ds_check = gr.Button("Analyser le dossier")
                    cap_text = gr.Textbox(label="Legende par defaut pour les images sans .txt",
                                          info="Depannage seulement : preferez le legendage automatique ci-dessous.")
                    cap_btn = gr.Button("Creer les legendes manquantes")
            with gr.Accordion("Legendage automatique (Qwen3-VL)", open=False):
                gr.Markdown("Un modele de vision decrit chaque image et ecrit le .txt correspondant. "
                            "A lancer **avant** l'entrainement (un seul job GPU a la fois). "
                            "Le modele est telecharge au premier usage.")
                with gr.Row():
                    lora_type = gr.Radio(core.LORA_TYPES, value=core.LORA_SUBJECT, label="Type de LoRA",
                                         info="Regle la consigne ci-dessous et le controle des legendes. "
                                              "Personnage : les traits permanents ne sont pas decrits. "
                                              "Style : seul le contenu est decrit.")
                    cap_extra = gr.Textbox(label="Autres elements permanents a ne pas decrire (optionnel)",
                                           placeholder="sunglasses, red crescent patch",
                                           info="En anglais, separes par des virgules. Ex. un accessoire que le "
                                                "personnage porte sur toutes les images. Il fera alors partie "
                                                "du mot declencheur et ne pourra plus etre retire.")
                with gr.Row():
                    cap_model = gr.Dropdown(core.CAPTION_MODELS, value=core.CAPTION_MODELS[0], label="Modele",
                                            info="Preconise : 8B (meilleures descriptions, ~10 Go de VRAM). "
                                                 "4B / 2B : plus rapides, moins precis.")
                    cap_tokens = gr.Slider(64, 512, value=256, step=32, label="Longueur maximale",
                                           info="Preconise : 256 (1 a 3 phrases detaillees).")
                    cap_redo = gr.Checkbox(label="Refaire les legendes existantes", value=False,
                                           info="Decoche : seules les images sans .txt sont traitees.")
                cap_prompt = gr.Textbox(label="Consigne donnee au modele (modifiable)",
                                        value=core.caption_prompt(core.LORA_SUBJECT), lines=5,
                                        info="Generee d'apres le type de LoRA, le mot declencheur de l'onglet "
                                             "Entrainement et les elements permanents. Un modele de vision ne "
                                             "respecte pas toujours une interdiction : relancez 'Analyser le "
                                             "dossier' apres le legendage pour reperer les ecarts.")
                cap_auto_btn = gr.Button("Lancer le legendage automatique", variant="primary")
            cap_panel = gr.HTML()
            cap_log = gr.Textbox(label="Journal du legendage", lines=8, max_lines=8, autoscroll=True,
                                 interactive=False)
            with gr.Accordion("Corriger les legendes : retirer les traits permanents", open=False):
                gr.Markdown("Pour un LoRA de personnage, retire des legendes les traits permanents du sujet "
                            "(cheveux, barbe, carrure, cicatrices, tatouages...) et les elements permanents "
                            "saisis plus haut. Les expressions et vetements restent ('eyes closed', 'skin-tight "
                            "suit'). Chaque proposition est modifiable : decochez celles a laisser. Les "
                            "originales sont sauvegardees a cote du dataset (`<dataset>_legendes_originales`).")
                with gr.Row():
                    capfix_btn = gr.Button("Proposer les corrections")
                    capfix_restore_btn = gr.Button("Restaurer les legendes d'origine")
                capfix_msg = gr.Textbox(label="Message", lines=2, interactive=False)
                capfix_table = gr.Dataframe(headers=CAPFIX_HEADERS,
                                            datatype=["bool", "str", "str", "str"], interactive=True,
                                            type="array", wrap=True, visible=False, column_widths=["8%", "14%", "39%", "39%"])
                capfix_apply_btn = gr.Button("Appliquer les corrections cochees", variant="primary", visible=False)
            with gr.Accordion("Audit du dataset : fonds unis, doublons, cadrages trop semblables", open=False):
                gr.Markdown("Repere ce que l'oeil laisse passer : un meme fond uni sur beaucoup d'images et absent "
                            "des legendes (il s'attache au mot declencheur : le LoRA le met partout), les "
                            "quasi-doublons (comptent double), les groupes d'images au meme cadrage (surappris). "
                            "Fonds : analyse des bords de l'image. Doublons : modele DINOv2 sur le processeur "
                            "(~90 Mo, telecharge au premier usage).")
                audit_btn = gr.Button("Auditer le dataset")
                audit_report = gr.Textbox(label="Rapport d'audit", lines=8, interactive=False)
                audit_fix_btn = gr.Button("Ajouter le fond aux legendes", visible=False, variant="primary")
                audit_gallery = gr.Gallery(label="Quasi-doublons (par paires)", columns=6, height=220)
                audit_state = gr.State([])
            with gr.Accordion("Rogner les filigranes et textes (detection automatique)", open=False):
                gr.Markdown(
                    "Repere les zones de texte (filigrane, signature, adresse de site, legende) et propose, pour "
                    "chaque image, le plus grand cadrage qui n'en contient aucune. L'image est seulement rognee : "
                    "rien n'est efface ni repeint. Les originaux sont copies dans un dossier voisin du dataset "
                    "(`<dataset>_originaux`). Non detectes : les logos sans texte et les filigranes repetes sur "
                    "toute l'image.")
                with gr.Row():
                    crop_keep = gr.Slider(40, 95, value=65, step=5, label="Part minimale de l'image a conserver (%)",
                                          info="Preconise : 65. En dessous, l'image est declaree non rognable "
                                               "plutot que d'etre trop amputee.")
                    crop_margin = gr.Slider(0, 5, value=1.5, step=0.5, label="Marge autour du texte (% du grand cote)",
                                            info="Preconise : 1,5. Augmentez si un bout de filigrane subsiste.")
                    crop_conf = gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="Seuil de detection",
                                          info="Preconise : 0,5. Plus bas : detecte les filigranes pales, mais "
                                               "aussi plus de faux textes.")
                crop_scan_btn = gr.Button("Analyser les images", variant="primary")
                crop_report = gr.Textbox(label="Resultat", lines=7, interactive=False)
                crop_gallery = gr.Gallery(label="Texte detecte en rouge, cadrage propose en vert, zone retiree assombrie",
                                          columns=5, height=380)
                crop_pick = gr.CheckboxGroup(choices=[], label="Images a rogner",
                                             info="Decochez celles ou le texte fait partie de la scene.")
                with gr.Row():
                    crop_apply_btn = gr.Button("Rogner les images cochees", variant="primary")
                    crop_restore_btn = gr.Button("Restaurer tous les originaux")
            ds_msg = gr.Textbox(label="Etat", lines=14, interactive=False,
                                info="Le controle des tailles utilise les resolutions cochees dans l'onglet Entrainement.")
            ds_gallery = gr.Gallery(label="Apercu", columns=8, height=260)

        # ============================================================ 2. Entrainement
        with gr.Tab("2. Entrainement"):
            with gr.Row():
                preset_sel = gr.Dropdown(core.list_job_settings(), label="Reglages enregistres",
                                         info="Chaque job lance enregistre ses reglages sous son nom.")
                preset_load = gr.Button("Charger")
                preset_save = gr.Button("Enregistrer les reglages actuels")
            process = gr.Radio(core.PROCESSES, value=core.PROC_STANDARD, label="Procede d'entrainement",
                               info="Standard : vos reglages, inchanges. Rapide 512 : resolution 512, rang 32, "
                                    "optimiseur automagic4, EMA 0,98, 30 passages - bien plus rapide par pas. "
                                    "Recette Fizgig : la meme, plus son adaptateur, l'optimisation de ressemblance "
                                    "et les apercus LCM. Choisir un procede remplit les champs ; tout reste modifiable.")
            with gr.Accordion("Options du procede (optimiseur, EMA, adaptateur, ressemblance, echantillonneur)",
                              open=False):
                with gr.Row():
                    optimizer = gr.Dropdown(list(core.OPTIMIZERS), value=core.OPT_ADAMW, label="Optimiseur",
                                            info="adamw8bit : le taux saisi est applique tel quel. automagic : le "
                                                 "taux saisi est un point de depart, l'optimiseur le regle seul "
                                                 "sans depasser 2,5e-4.")
                    ema = gr.Dropdown(list(core.EMA_CHOICES), value=core.EMA_OFF, label="Moyenne des poids (EMA)",
                                      info="Lisse le LoRA sur les derniers pas : apercus et checkpoints plus "
                                           "stables. Preconise avec 512 px : 0,98.")
                with gr.Row():
                    adapter = gr.Dropdown(core.ADAPTERS, value=core.ADAPTER_OSTRIS, label="Adaptateur d'entrainement",
                                          info="Garde le modele distille stable pendant l'apprentissage ; jamais "
                                               "dans le LoRA final. Fizgig : telecharge (~150 Mo) et converti au "
                                               "premier job. Changez de nom de job quand vous changez d'adaptateur.")
                    sampler = gr.Dropdown(core.SAMPLERS, value=core.SAMPLER_EULER, label="Echantillonneur des apercus",
                                          info="N'affecte que les apercus et l'onglet Tester un LoRA, pas le LoRA. "
                                               "LCM : la meme graine donne une autre image qu'Euler.")
                likeness = gr.Checkbox(label="Optimisation de ressemblance (calendrier Fizgig)", value=False,
                                       info="Tire plus souvent le niveau de bruit le plus net, ou se forment peau, "
                                            "yeux et details : 17 -> 33 -> 50 % aux passages 1, 3, 5 ; 75 % aux "
                                            "passages 2, 4, 6 et ensuite. Pour un sujet ; inutile pour un style.")
                gr.Markdown("**Contre le debordement** (le personnage qui apparait partout) : deux protections "
                            "d'ai-toolkit, combinables. Mesurez leur effet avec la colonne Debordement de "
                            "Resultats > Ressemblance.")
                with gr.Row():
                    dop = gr.Checkbox(label="Preservation (DOP)", value=False,
                                      info="A chaque pas, le LoRA est penalise s'il change ce que le modele de base "
                                           "dessine pour la legende ou le mot declencheur est remplace par la classe "
                                           "ci-contre. Aucune image a preparer, mais chaque pas est environ 1,6 a 2 "
                                           "fois plus lent et demande plus de memoire : verifiez avec le Test rapide.")
                    dop_class = gr.Textbox(label="Classe preservee", value=core.CONTROL_SUBJECT,
                                           info="En anglais : ce que le mot declencheur designe sans le LoRA "
                                                "(a person, a man, a woman, a dog...).")
                with gr.Row():
                    reg = gr.Checkbox(label="Images de regularisation", value=False,
                                      info="Melange a l'entrainement des images 'a person...' produites par le modele "
                                           "de base (dossier <dataset>_regularisation). Pas de perte de vitesse par "
                                           "pas, mais elles prennent une part des pas : avec autant d'images que le "
                                           "dataset, doublez les pas pour garder le meme apprentissage du sujet.")
                    reg_weight = gr.Number(label="Poids des images de regularisation", value=1.0, minimum=0.1,
                                           maximum=2.0, step=0.1,
                                           info="Preconise : 1. Plus bas : protection plus douce.")
                with gr.Row():
                    reg_count = gr.Number(label="Nombre d'images a generer", value=0, precision=0,
                                          info="0 = autant que d'images dans le dataset (60 au plus).")
                    reg_btn = gr.Button("Generer les images de regularisation")
                reg_msg = gr.Textbox(label="Images de regularisation", interactive=False, lines=2)
            with gr.Row():
                job_name = gr.Textbox(label="Nom du job (meme nom = reprise)", value="h3img_mon_lora_v1",
                                      info="Lettres, chiffres, _ et -. Changez le nom (v2, v3...) pour repartir "
                                           "de zero ; gardez-le pour reprendre.")
                trigger = gr.Textbox(label="Mot declencheur (optionnel)",
                                     info="Preconise : un mot invente avec des chiffres (ex. m41ne) pour un sujet, "
                                          "jamais un vrai mot ni un nom connu ; vide pour un style. "
                                          "A reutiliser ensuite dans vos prompts.")
            preset = gr.Radio([core.PRESET_INT8, core.PRESET_W4A8], value=core.PRESET_INT8, label="Preset 3090",
                              info="Preconise : int8. w4a8 : a essayer avec le Test rapide ; s'il fonctionne "
                                   "il evite l'aller-retour avec la RAM.")
            w4a8_fix = gr.Checkbox(label="Correctif w4a8 pour GPU anterieurs aux RTX 40 (experimental)", value=True,
                                   info="Sans effet en int8. En w4a8 sur 3090 : contourne le noyau triton "
                                        "incompatible en decompressant en torch pur. Laissez coche.")
            offload = gr.Slider(0, 0.8, value=0.35, step=0.05, label="Part du transformer en RAM",
                                info="Preconise : 0.35 en int8, 0 en w4a8. Erreur memoire -> 0.5 puis 0.7 "
                                     "(plus lent). Le Test rapide indique s'il y a de la marge.")
            resol = gr.CheckboxGroup(["512", "768", "1024", "1536", "2048"], value=["1024", "1536"],
                                     label="Resolutions d'entrainement",
                                     info="512 : le reglage de Fizgig, de loin le plus rapide (la memoire change "
                                          "peu avec la resolution, la duree d'un pas beaucoup). 1024 + 1536 : plus "
                                          "lent, a comparer sur vos apercus. Defaut ai-toolkit : 512 + 768 + 1024.")
            with gr.Row():
                steps = gr.Number(label="Pas", value=2000, precision=0,
                                  info="Preconise : 1500 a 3000 (defaut ai-toolkit : 3000). Repere : ~100 pas "
                                       "par image. Jugez sur les apercus, pas sur le nombre.")
                lr = gr.Number(label="Learning rate", value=1e-4,
                               info="adamw8bit : 0.0001 (defaut ai-toolkit) ; brule ou deforme -> 0.00005 ; "
                                    "n'apprend rien -> 0.0002. automagic : 0.00025 (point de depart et plafond).")
                rank = gr.Dropdown([4, 8, 16, 32], value=16, label="Rang LoRA",
                                   info="Preconise : 16 (defaut ai-toolkit pour ce modele). 8 = plus leger, "
                                        "moins de memoire. 32 = sujets/styles complexes, plus de memoire.")
            with gr.Row():
                save_every = gr.Number(label="Sauvegarde tous les N pas", value=250, precision=0,
                                       info="Preconise : 250. Les 6 derniers checkpoints sont gardes : "
                                            "le meilleur LoRA est souvent avant le dernier.")
                sample_every = gr.Number(label="Apercus tous les N pas", value=250, precision=0,
                                         info="Preconise : 250, aligne sur les sauvegardes pour pouvoir "
                                              "choisir le checkpoint d'apres ses apercus.")
                sample_size = gr.Dropdown([1024, 1536, 2048], value=2048, label="Taille des apercus",
                                          info="Preconise : 2048 (le modele vise 4 MP+). Erreur memoire "
                                               "pendant les apercus -> 1536. 1024 = artefacts normaux.")
                no_sample = gr.Checkbox(label="Desactiver les apercus", value=False,
                                        info="A cocher si les apercus saturent la memoire.")
            ctrl = gr.Textbox(label="Dossier d'images de reference (optionnel)",
                              info="Vide = LoRA texte-vers-image (cas normal). Rempli = LoRA d'edition : "
                                   "memes noms de fichiers que le dataset, references nommees <Picture 1> "
                                   "dans les legendes.")
            prompts = gr.Textbox(label="Prompts d'apercu (un par ligne)", value=DEFAULT_PROMPTS, lines=3,
                                 info="Preconise : 2 a 4 prompts detailles, avec votre mot declencheur, dont "
                                      "un proche du dataset et un tres different (teste la generalisation). "
                                      "Chaque prompt rallonge la pause d'apercu.")
            control = gr.Checkbox(value=True, label="Ajouter un prompt temoin sans mot declencheur (preconise)",
                                  info="Sert a detecter le debordement : le personnage qui apparait meme sans son "
                                       "mot, signe de sur-apprentissage. Si aucun de vos prompts ne s'en passe, "
                                       "l'UI ajoute le premier prompt avec 'a person' a la place du mot. Il est "
                                       "affiche ci-dessous.")
            prompt_note = gr.Markdown("")
            with gr.Accordion("Aide : valeurs preconisees et diagnostic", open=False):
                gr.Markdown(AIDE_REGLAGES)

            with gr.Row():
                calib_btn = gr.Button("Test rapide (20 pas : vitesse + memoire)")
                prev_btn = gr.Button("Voir le YAML")
                queue_btn = gr.Button("Ajouter a la file d'attente")
                go_btn = gr.Button("Lancer / Reprendre", variant="primary")
            train_panel = gr.HTML()
            train_log = gr.Textbox(label="Journal de l'entrainement et du Test rapide", lines=10, max_lines=10,
                                   autoscroll=True, interactive=False)
            calib_box = gr.Textbox(label="Calibrage (resultat du Test rapide)", lines=6, interactive=False,
                                   info="Mesure la vitesse reelle et le pic de VRAM avec les reglages ci-dessus, "
                                        "puis propose des ajustements et une duree estimee.")

            with gr.Accordion("Pause automatique et file d'attente", open=False):
                with gr.Row():
                    pause_step = gr.Number(label="Mettre en pause au pas", value=0, precision=0,
                                           info="0 = desactive. La pause sauvegarde l'etat, comme le bouton.")
                    pause_time = gr.Textbox(label="Mettre en pause a (HH:MM)", placeholder="07:30",
                                            info="Vide = desactive. Si l'heure est passee : demain. "
                                                 "Utile pour liberer le PC le matin.")
                    sched_btn = gr.Button("Programmer")
                sched_box = gr.Textbox(label="Pause programmee", interactive=False,
                                       value=ORCH.schedule_text)
                queue_box = gr.Textbox(label="File d'attente (jobs lances l'un apres l'autre)", lines=4,
                                       interactive=False,
                                       info="Un job en erreur n'arrete pas la file. Une pause ou un arret "
                                            "manuel la suspend.")
                with gr.Row():
                    queue_resume = gr.Button("Reprendre la file")
                    queue_clear = gr.Button("Vider la file")
            yaml_box = gr.Code(label="Config generee", language="yaml")

        # ============================================================ 3. Resultats
        with gr.Tab("3. Resultats"):
            with gr.Row():
                job_sel = gr.Dropdown(choices=core.list_jobs(), label="Job")
                refresh_btn = gr.Button("Rafraichir la liste")
                open_btn = gr.Button("Ouvrir le dossier du job")
            gr.Markdown("Apercus, courbe et checkpoints du job selectionne se mettent a jour tout seuls (5 s).")
            res_sig = gr.State("")
            res_gallery = gr.Gallery(label="Apercus (plus recents d'abord)", columns=4, height=460)
            with gr.Tabs():
                with gr.Tab("Perte"):
                    loss_plot = gr.HTML()
                    gr.Markdown("*Lecture : la courbe lissee qui devient plate indique que les pas suivants "
                                "apportent peu ; comparez les checkpoints reperes en vert dans l'onglet Tester un "
                                "LoRA. La courbe ne detecte pas le sur-apprentissage : seuls des apercus varies "
                                "le montrent.*")
                with gr.Tab("Ressemblance"):
                    likeness_plot = gr.HTML()
                    gr.Markdown("*Chaque apercu est compare aux images du dataset (ressemblance, modele DINOv2) et "
                                "a son prompt sans le mot declencheur (fidelite, modele CLIP), sur le processeur : "
                                "la carte reste a l'entrainement. La ressemblance seule recompenserait le "
                                "sur-apprentissage : le checkpoint conseille est le plus ressemblant parmi ceux "
                                "qui suivent encore le prompt comme au debut. Mesures indicatives : verifiez "
                                "toujours avec Tester un LoRA. Modeles telecharges au premier usage (environ "
                                "700 Mo).*")
                with gr.Tab("Puissance, vitesse et temperature"):
                    perf_plot = gr.HTML()
                    gr.Markdown("*Releve pendant l'entrainement, par tranches de 5 pas (les apercus et sauvegardes "
                                "sont exclus). Pour trouver la bonne limite : faites varier la limite de puissance "
                                "(onglet Outils) ou laissez le thermostat la faire bouger, puis lisez le tableau. Le "
                                "seuil conseille est la puissance la plus basse qui reste a moins de 3 % de la "
                                "vitesse maximale ; Wh par pas indique l'energie pour un meme travail.*")
            res_files = gr.Textbox(label="Checkpoints LoRA (.safetensors) - chemins sur le disque",
                                   lines=5, interactive=False)
            with gr.Row():
                ck_sel = gr.Dropdown(choices=[], label="Checkpoint a exporter",
                                     info="Le meilleur est souvent un checkpoint intermediaire : "
                                          "comparez-les dans l'onglet Tester un LoRA.")
                comfy_btn = gr.Button("Copier dans ComfyUI")
                gensend_btn = gr.Button("Envoyer au generateur")

        # ============================================================ 4. Tester un LoRA
        with gr.Tab("4. Tester un LoRA"):
            gr.Markdown(
                "Genere les memes prompts avec plusieurs checkpoints (meme graine) pour les comparer. "
                "Chaque checkpoint est un job a part qui recharge le modele : comptez plusieurs minutes "
                "par checkpoint. A lancer quand aucun entrainement ne tourne, ou a mettre en file. "
                "*Experimental : repose sur un job ai-toolkit a 0 pas d'entrainement.*")
            with gr.Row():
                test_job = gr.Dropdown(choices=core.list_jobs(), label="Job")
                test_refresh = gr.Button("Rafraichir")
            test_cks = gr.CheckboxGroup(choices=[], label="Checkpoints a comparer",
                                        info="Preconise : 2 a 4 checkpoints espaces (ex. 1000, 1500, 2000).")
            test_prompts = gr.Textbox(label="Prompts de test (un par ligne)", lines=3,
                                      info="Preconise : un prompt proche du dataset, un tres different, "
                                           "et un SANS le mot declencheur (il ne doit pas faire apparaitre le sujet).")
            with gr.Row():
                test_size = gr.Dropdown([1024, 1536, 2048], value=1536, label="Taille",
                                        info="Preconise : 1536 pour comparer vite, 2048 pour juger la qualite finale.")
                test_seed = gr.Number(label="Graine", value=42, precision=0,
                                      info="Identique pour tous les checkpoints : seules les differences "
                                           "dues au LoRA restent visibles.")
                test_btn = gr.Button("Lancer les tests", variant="primary")
            test_panel = gr.HTML()
            test_log = gr.Textbox(label="Journal des tests", lines=8, max_lines=8, autoscroll=True, interactive=False)
            test_sig = gr.State("")
            test_gallery = gr.Gallery(label="Resultats (legende : checkpoint | prompt)", columns=4, height=520)

        # ============================================================ Modeles
        # ============================================================ 5. Generer
        with gr.Tab("5. Generer") as gen_tab:
            gr.Markdown(
                "Genere des images avec vos LoRA grace au generateur **MiniMaxH3-Portable** (installe a part, "
                "avec son propre environnement et ses modeles convertis). Il est demarre d'ici et affiche "
                "ci-dessous. Il garde environ 20 Go de VRAM : tout job de l'entraineur (entrainement, test, "
                "legendage...) l'arrete d'abord automatiquement, et il ne demarre pas pendant un job.")
            with gr.Row():
                gen_dir = gr.Textbox(label="Dossier du generateur", value=gen.detect, scale=3,
                                     info="Detecte automatiquement s'il est a cote de ce dossier.")
                gen_check = gr.Button("Verifier", scale=1)
            gen_state = gr.Textbox(label="Etat", lines=5, interactive=False,
                                   value=lambda: gen_state_text(gen.detect()))
            gr.Markdown(gen.LICENSE_NOTICE)
            gen_lic = gr.Checkbox(label="J'ai lu la licence et je l'accepte",
                                  value=lambda: gen.state(gen.detect())["license"])
            with gr.Row():
                gen_start_btn = gr.Button("Demarrer le generateur", variant="primary")
                gen_stop_btn = gr.Button("Arreter le generateur", variant="stop")
            gen_status = gr.Textbox(label="Generateur", interactive=False)
            with gr.Row():
                gen_job = gr.Dropdown(choices=core.list_jobs(), value=None, label="Job", scale=2)
                gen_ck = gr.Dropdown(choices=[], label="Checkpoint a envoyer", scale=3,
                                     info="Preselection : le checkpoint conseille par Resultats > Ressemblance "
                                          "s'il existe, sinon le plus recent.")
                gen_send = gr.Button("Envoyer au generateur", scale=1)
            with gr.Accordion("Journal du generateur", open=False):
                gen_log = gr.Textbox(lines=12, max_lines=12, autoscroll=True, interactive=False, show_label=False)
            gen_ready = gr.State(False)
            gen_frame = gr.HTML(gen_frame_html(False))

        with gr.Tab("Modeles"):
            gr.Markdown("Optionnel : sinon les poids se telechargent au premier entrainement. "
                        "Une coupure reprend automatiquement ; relancer reprend aussi le fichier partiel.")
            dl_choice = gr.Radio(["int8", "w4a8", "les deux"], value="int8", label="Transformer",
                                 info="Preconise : int8 (correspond au preset par defaut). "
                                      "Prevoir 35 a 50 Go libres selon le choix.")
            dl_btn = gr.Button("Telecharger", variant="primary")
            dl_panel = gr.HTML()
            dl_log = gr.Textbox(label="Journal du telechargement", lines=8, max_lines=8, autoscroll=True,
                                interactive=False)
            dl_state = gr.Textbox(label="Etat des fichiers", lines=5, interactive=False, value=core.models_state)
            dl_refresh = gr.Button("Rafraichir l'etat")

        # ============================================================ Outils
        with gr.Tab("Outils"):
            gr.Markdown("### Notifications")
            with gr.Row():
                opt_sound = gr.Checkbox(label="Son a la fin d'un job ou en cas d'erreur", value=_cfg["sound"])
                opt_awake = gr.Checkbox(label="Empecher la mise en veille pendant un job", value=_cfg["keep_awake"],
                                        info="Preconise : coche. L'ecran peut s'eteindre, le PC reste actif. "
                                             "Sans effet si vous fermez le capot d'un portable.")
                opt_like = gr.Checkbox(label="Mesurer la ressemblance des apercus",
                                       value=lambda: bool(core.load_settings().get("likeness_on", True)),
                                       info="Onglet Resultats > Ressemblance. Calcul sur le processeur ; modeles "
                                            "telecharges au premier usage (environ 700 Mo).")
                opt_leakpause = gr.Checkbox(label="Pause automatique si le debordement devient fort",
                                            value=lambda: bool(core.load_settings().get("leak_pause", False)),
                                            info="Le personnage apparait sur le prompt temoin, sans son mot : "
                                                 "pause avec sauvegarde. Sinon, simple notification.")
            opt_ntfy = gr.Textbox(label="Notification sur telephone : sujet ntfy.sh (optionnel)",
                                  value=_cfg["ntfy_topic"],
                                  info="Installez l'application ntfy, abonnez-vous a un sujet au nom long et "
                                       "difficile a deviner, et saisissez-le ici. Le nom du job transite par le "
                                       "service public ntfy.sh ; laissez vide pour ne rien envoyer.")
            gr.Markdown("### Temperature et puissance du GPU")
            with gr.Row():
                opt_talert = gr.Number(label="Alerte de temperature (°C)", value=_cfg["temp_alert"], precision=0,
                                       info="Preconise : 5 C sous la limite thermique de la carte (78 pour une limite "
                                            "a 83). Notification + temperature en rouge. 0 = desactive.")
                opt_tpause = gr.Number(label="Pause automatique (°C)", value=_cfg["temp_pause"], precision=0,
                                       info="Preconise : 88. Au-dela pendant 60 s : pause avec sauvegarde "
                                            "(filet de securite, pas une regulation). 0 = desactive.")
                opt_watts = gr.Number(label="Limite de puissance pendant les jobs (W)",
                                      value=_cfg["power_limit_w"], precision=0,
                                      info="0 = ne pas toucher. Preconise sur 3090 si elle chauffe : 250 a 280 "
                                           "(la plage de votre carte s'affiche dessous). Cliquez Enregistrer : "
                                           "appliquee tout de suite si un job tourne, sinon au prochain "
                                           "lancement. Windows peut demander une confirmation.")
            with gr.Row():
                opt_thermo = gr.Checkbox(label="Thermostat : ajuster la puissance pour tenir une temperature",
                                         value=_cfg["thermo_on"],
                                         info="Toutes les 45 s pendant un job : baisse la limite de 10 a 30 W si la "
                                              "carte depasse la cible, la remonte de 10 W si elle est 3 °C en "
                                              "dessous. Necessite 2_LANCER_UI_ADMIN.bat.")
                opt_thtarget = gr.Number(label="Temperature cible (°C)", value=_cfg["thermo_target"], precision=0,
                                         info="Preconise : 10 °C sous la limite thermique de la carte "
                                              "(72 pour une limite a 83).")
                opt_thmin = gr.Number(label="Puissance plancher (W)", value=_cfg["thermo_min_w"], precision=0,
                                      info="Preconise : 150. Le thermostat ne descend jamais en dessous. Son "
                                           "plafond est la limite de puissance ci-dessus (ou celle d'origine si 0).")
            with gr.Row():
                opt_sweep = gr.Checkbox(label="Balayage de puissance au debut de l'entrainement",
                                        value=lambda: bool(core.load_settings().get("sweep_on", False)),
                                        info="Une fois par job : la limite descend par paliers de 30 pas, puis le "
                                             "seuil conseille (le plus bas a moins de 3 % de la vitesse maximale) "
                                             "est applique et enregistre comme limite. Necessite "
                                             "2_LANCER_UI_ADMIN.bat. Resultat dans Resultats > Puissance.")
                opt_sweepmin = gr.Number(label="Plancher du balayage (W)", precision=0,
                                         value=lambda: core.load_settings().get("sweep_min_w", 200),
                                         info="Preconise : 200 sur une 3090. Le balayage part de la limite "
                                              "ci-dessus (ou celle d'origine si 0).")
                opt_sweepstep = gr.Number(label="Ecart entre paliers (W)", precision=0,
                                          value=lambda: core.load_settings().get("sweep_step_w", 30),
                                          info="Preconise : 30 (environ 5 paliers, 6 a 8 minutes sur une 3090).")
            temp_save = gr.Button("Enregistrer temperature, puissance et thermostat", variant="primary")
            thermo_box = gr.Textbox(label="Thermostat (etat et derniers ajustements)", interactive=False,
                                    value=ORCH.thermo_text, lines=4,
                                    info="Reflete les reglages ENREGISTRES : une case cochee ou une valeur modifiee "
                                         "ne compte qu'apres un clic sur Enregistrer (cocher la case l'enregistre "
                                         "automatiquement).")
            power_box = gr.Textbox(label="Puissance", interactive=False, value=core.power_text, lines=2)
            with gr.Row():
                power_apply = gr.Button("Appliquer maintenant la valeur saisie")
                power_reset = gr.Button("Retablir la limite d'origine")
            gr.Markdown("La vitesse des ventilateurs n'est pas pilotee par cette UI (risque en cas de plantage) : "
                        "reglez une courbe dans MSI Afterburner ou FanControl. Sans droits administrateur, la limite "
                        "de puissance reste en place apres le job : utilisez 'Retablir la limite d'origine' "
                        "ou redemarrez le PC.")
            gr.Markdown("### Cout de l'electricite (GPU)")
            with gr.Row():
                opt_price = gr.Number(label="Prix du kWh", value=lambda: core.load_settings().get("kwh_price") or 0, minimum=0,
                                      info="Le montant de votre facture d'electricite, abonnement exclu. "
                                           "0 = afficher seulement les kWh.")
                opt_cur = gr.Dropdown([(f"{c} ({v[0]})", c) for c, v in core.CURRENCIES.items()],
                                      value=lambda: core.currency_for(code), label="Monnaie")
            gr.Markdown("Pendant un job, la consommation du GPU (lue par nvidia-smi toutes les 2 s), son cout "
                        "et le cout projete a la fin s'affichent a cote du temps ecoule, du temps restant et de "
                        "la vitesse. Seule la carte graphique est comptee : processeur, ecran et reste du PC "
                        "s'y ajoutent. Enregistrez avec le bouton Enregistrer ci-dessous.")
            gr.Markdown("### ComfyUI")
            opt_comfy = gr.Textbox(label="Dossier 'loras' de ComfyUI", value=_cfg["comfy_loras_dir"],
                                   placeholder="D:\\ComfyUI\\models\\loras",
                                   info="Utilise par le bouton 'Copier dans ComfyUI' de l'onglet Resultats.")
            with gr.Row():
                tools_save = gr.Button("Enregistrer", variant="primary")
                tools_test = gr.Button("Tester la notification")
            gr.Markdown("### Version d'ai-toolkit")
            ver_box = gr.Textbox(label="Version", lines=2, interactive=False, value=core.version_text)
            with gr.Row():
                pin_btn = gr.Button("Figer la version actuelle")
                unpin_btn = gr.Button("Defiger")
            gr.Markdown("Preconise : figer apres un premier entrainement reussi. Une version figee n'est plus "
                        "modifiee par 3_METTRE_A_JOUR.bat tant que vous ne la defigez pas.")

        cfg_inputs = [job_name, ds_folder, ctrl, preset, offload, resol, steps, lr, rank,
                      trigger, save_every, sample_every, sample_size, no_sample, prompts, w4a8_fix,
                      process, optimizer, ema, adapter, likeness, sampler, control,
                      dop, dop_class, reg, reg_weight]
        process.input(apply_process, [process, ds_folder],
                      [resol, rank, lr, steps, save_every, sample_every,
                       optimizer, ema, adapter, likeness, sampler, msg])
        assert len(cfg_inputs) == len(core.PARAM_KEYS)

        # dataset
        ds_extra = [resol, lora_type, trigger, cap_extra]
        ds_check.click(dataset_info, [ds_folder] + ds_extra, [ds_msg, ds_gallery])
        ds_files.upload(dataset_upload, [ds_folder, ds_files] + ds_extra, [ds_msg, ds_gallery])
        crop_scan_btn.click(crop_scan, [ds_folder, crop_keep, crop_margin, crop_conf],
                            [crop_report, crop_gallery, crop_pick])
        crop_apply_btn.click(crop_apply, [ds_folder, crop_pick], [crop_report, crop_pick]).then(
            dataset_info, [ds_folder] + ds_extra, [ds_msg, ds_gallery])
        crop_restore_btn.click(crop_restore, ds_folder, [crop_report, crop_pick, crop_gallery]).then(
            dataset_info, [ds_folder] + ds_extra, [ds_msg, ds_gallery])
        for comp in (lora_type, trigger, cap_extra):
            comp.change(core.caption_prompt, [lora_type, trigger, cap_extra], cap_prompt)
        audit_btn.click(audit_run, ds_folder, [audit_report, audit_gallery, audit_state, audit_fix_btn])
        capfix_btn.click(capfix_propose, [ds_folder, lora_type, trigger, cap_extra],
                         [capfix_table, capfix_msg, capfix_apply_btn])
        capfix_apply_btn.click(capfix_apply, [ds_folder, capfix_table], [capfix_msg, capfix_table, capfix_apply_btn])
        capfix_restore_btn.click(capfix.restore, ds_folder, capfix_msg)
        audit_fix_btn.click(audit_fix, [ds_folder, audit_state], [audit_report, audit_fix_btn])
        reg_btn.click(reg_generate, [reg_count] + cfg_inputs, reg_msg)
        ds_folder.change(core.reg_status, ds_folder, reg_msg)
        for comp in (trigger, prompts, no_sample, control):
            comp.change(prompt_check, [trigger, prompts, no_sample, control], prompt_note)
        cap_btn.click(fill_captions, [ds_folder, cap_text], [ds_msg, ds_gallery])
        # retour immediat : le panneau passe a "Lancement" des le clic, sans attendre le minuteur
        cap_auto_btn.click(lambda f, m, r, p, t: core.start_captioning(f, m, r, p, t),
                           [ds_folder, cap_model, cap_redo, cap_prompt, cap_tokens], msg,
                           show_progress="minimal").then(render_caption_panel, None, [cap_panel, cap_log])
        # entrainement
        # .input : seulement quand l'utilisateur change le preset, pas quand des reglages sont recharges
        preset.input(lambda p: 0.35 if p == core.PRESET_INT8 else 0.0, preset, offload)
        prev_btn.click(preview, cfg_inputs, yaml_box)
        go_btn.click(start_training, cfg_inputs, [msg, job_sel]).then(
            render_train_panel, None, [train_panel, train_log])
        queue_btn.click(enqueue_training, cfg_inputs, msg)
        calib_btn.click(start_calibration, cfg_inputs, msg).then(
            render_train_panel, None, [train_panel, train_log])
        preset_save.click(save_settings_now, cfg_inputs, [msg, preset_sel])
        preset_load.click(load_settings_now, preset_sel, cfg_inputs + [msg])
        sched_btn.click(lambda s, t: ORCH.schedule_pause(s, t), [pause_step, pause_time], sched_box)
        queue_resume.click(core.resume_queue, None, queue_box)
        queue_clear.click(core.clear_queue, None, queue_box)
        # commandes
        stop_btn.click(core.stop_job, None, msg)
        pause_btn.click(lambda: core.pause_job(), None, msg)
        sample_btn.click(core.sample_now, None, msg)
        # resultats
        refresh_btn.click(lambda: gr.update(choices=core.list_jobs()), None, job_sel)
        job_sel.change(job_results, job_sel, [res_gallery, res_files, ck_sel, loss_plot, perf_plot, likeness_plot])
        open_btn.click(open_job_folder, job_sel, msg)
        comfy_btn.click(copy_to_comfy, [job_sel, ck_sel], msg)
        gensend_btn.click(send_to_generator, [job_sel, ck_sel], msg)
        # generateur
        gen_check.click(gen_state_text, gen_dir, gen_state)
        gen_lic.input(gen_license, [gen_dir, gen_lic], gen_state)
        gen_start_btn.click(gen_start, gen_dir, msg).then(gen_state_text, gen_dir, gen_state)
        gen_stop_btn.click(lambda: "Generateur arrete." if gen.GEN.stop() else "Le generateur ne tournait pas.",
                           None, msg)
        gen_tab.select(lambda: gr.update(choices=core.list_jobs()), None, gen_job)
        gen_job.change(gen_job_choices, gen_job, gen_ck)
        gen_send.click(send_to_generator, [gen_job, gen_ck], msg)
        gr.Timer(2.0).tick(gen_tick, gen_ready, [gen_status, gen_log, gen_frame, gen_ready])
        # tests
        test_refresh.click(lambda: gr.update(choices=core.list_jobs()), None, test_job)
        test_job.change(test_choices, test_job, test_cks)
        test_btn.click(start_lora_test, [test_job, test_cks, test_prompts, test_size, test_seed], msg).then(
            render_test_panel, None, [test_panel, test_log])
        # modeles / outils
        dl_btn.click(core.start_download, dl_choice, msg).then(render_download_panel, None, [dl_panel, dl_log])
        dl_refresh.click(core.models_state, None, dl_state)
        tools_inputs = [opt_sound, opt_ntfy, opt_comfy, opt_awake, opt_talert, opt_tpause, opt_watts,
                        opt_thermo, opt_thtarget, opt_thmin, opt_price, opt_cur, opt_like,
                    opt_leakpause, opt_sweep, opt_sweepmin, opt_sweepstep]
        # trois facons d'enregistrer, meme effet : bouton general, bouton de la section, case thermostat
        for trigger_event in (tools_save.click, temp_save.click, opt_thermo.input):
            trigger_event(save_tools, tools_inputs, msg).then(
                core.power_text, None, power_box).then(ORCH.thermo_text, None, thermo_box)
        gr.Timer(5.0).tick(ORCH.thermo_text, None, thermo_box)
        tools_test.click(test_notification, None, msg)
        power_apply.click(apply_power_field, opt_watts, msg).then(core.power_text, None, power_box)
        power_reset.click(core.reset_power_now, None, msg).then(core.power_text, None, power_box)
        pin_btn.click(core.pin_version, None, ver_box)
        unpin_btn.click(core.unpin_version, None, ver_box)

        status_outputs = [status, calib_box, queue_box, cap_panel, cap_log, train_panel, train_log,
                          test_panel, test_log, dl_panel, dl_log]
        gr.Timer(1.0).tick(render_status, None, status_outputs)
        gr.Timer(5.0).tick(auto_results, [job_sel, res_sig],
                           [res_gallery, res_files, ck_sel, loss_plot, perf_plot, likeness_plot, res_sig])
        gr.Timer(5.0).tick(auto_test_results, [test_job, test_sig], [test_gallery, test_sig])
        demo.load(render_status, None, status_outputs)
        demo.load(lambda: i18n.save_lang(code), None, None)
        demo.load(restore_page, None, cfg_inputs + [preset_sel, job_sel, test_job])

    # champs dont le contenu appartient a l'utilisateur (chemins, prompts, legendes, journaux)
    keep = [ds_folder, cap_text, cap_extra, cap_prompt, cap_log, job_name, trigger, ctrl, prompts, dop_class,
            train_log, pause_time, test_prompts, test_log, dl_log, opt_ntfy, opt_comfy, gen_dir, gen_log]
    i18n.localize_blocks(demo, code, keep_values=keep)
    i18n.wrap_events(demo, code, keep_values=keep)
    return demo


def main():
    import threading
    import webbrowser

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    ORCH.start()
    app = FastAPI()

    @app.get("/")
    def _root():
        return RedirectResponse(f"/{i18n.saved_lang()}/")

    for code, name in i18n.LANGS:
        print(f"  interface {name} ...", flush=True)
        app = gr.mount_gradio_app(app, build(code).queue(), path=f"/{code}", show_error=True,
                                  allowed_paths=[core.ROOT], js=i18n.RESTORE_TAB_JS)
    url = "http://127.0.0.1:7860/"
    print(f"UI : {url}   ({core.MODELS})", flush=True)
    if os.environ.get("H3_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")


if __name__ == "__main__":
    main()
