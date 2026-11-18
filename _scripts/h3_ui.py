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
        if d.get("lr"):
            detail += f" &nbsp;|&nbsp; taux {d['lr']}"
    if energy:
        cost = core.energy_text((d or {}).get("eta"), i18n.lang_now())
        if cost:
            detail += f" &nbsp;|&nbsp; {cost}"
    name = (d or {}).get("desc") or ""
    alert = ("<div class='h3-alert'>Taux d'apprentissage effondre : "
             "l'entrainement n'apprend presque plus. Mettez en pause et relancez avec adamw8bit.</div>"
             if (d or {}).get("lr_low") else "")
    return f"""
<div class="h3-bar">
  <div class="h3-bar-head"><b>{title}</b><span>{pct} %</span></div>
  <div class="h3-track"><div class="h3-fill" style="width:{pct}%;background:{color}"></div></div>
  <div class="h3-bar-detail">{name} {detail}</div>{alert}
</div>"""


def _meter(label, used, total, extra=""):
    pct = max(0, min(100, int(100 * used / total))) if total else 0
    color = "var(--h3-bad)" if pct >= 96 else "var(--h3-warn)" if pct >= 88 else "var(--h3-accent)"
    return (f"<div class='h3-meter'><div class='h3-meter-head'><b>{label}</b> {used:.1f} / {total:.1f} Go "
            f"{extra}</div><div class='h3-track thin'><div class='h3-fill' style='width:{pct}%;"
            f"background:{color}'></div></div></div>")


def _monitor_html():
    parts, chips = [], []
    if MON.vram_used is not None and MON.vram_total:
        extra = f"<span class='h3-muted'>(pic {MON.peak_all:.1f})</span>" if MON.peak_all else ""
        parts.append(_meter("VRAM", MON.vram_used, MON.vram_total, extra))
        if MON.gpu_util is not None:
            alert = int(core.load_settings().get("temp_alert") or 0)
            hot = alert and MON.gpu_temp is not None and MON.gpu_temp >= alert
            chips.append(f"<span class='h3-chip'>GPU {MON.gpu_util} %</span>")
            chips.append(f"<span class='h3-chip{' hot' if hot else ''}'>{MON.gpu_temp} °C</span>")
            if MON.fan is not None:
                chips.append(f"<span class='h3-chip'>ventilateur {MON.fan} %</span>")
            if MON.power_draw is not None:
                watts = f"{MON.power_draw:.0f}" + (f" / {MON.power_limit:.0f} W" if MON.power_limit else " W")
                chips.append(f"<span class='h3-chip'>{watts}</span>")
    if MON.ram_used and MON.ram_total:
        parts.append(_meter("RAM", MON.ram_used, MON.ram_total))
    if not parts:
        return ""
    return f"<div class='h3-meters'>{''.join(parts)}<div class='h3-chips'>{''.join(chips)}</div></div>"


TAB_OF_KIND = {"caption": "1. Dataset", "train": "2. Entrainement", "calib": "2. Entrainement",
               "test": "4. Tester un LoRA", "download": "Outils"}


def render_status():
    s = RUN.snapshot()
    if s["running"]:
        html = (f"<div class='h3-st-line'><span class='h3-dot run'></span><b>EN COURS</b> "
                f"&nbsp; {s['label']} &nbsp;<span class='h3-muted'>({_dur(s['elapsed'])})</span>"
                f" &nbsp;<span class='h3-muted'>- details et journal dans l'onglet "
                f"{TAB_OF_KIND.get(s['kind'], '')}</span></div>"
                f"<div class='h3-st-sub'>Phase : <b>{s['phase']}</b></div>")
    else:
        html = ("<div class='h3-st-line'><span class='h3-dot'></span><b>INACTIF</b> &nbsp;"
                "<span class='h3-muted'>aucun job en cours</span></div>")
    if s["running"] and s["kind"] in RUN.CONTROLLED:
        st = core.db_get(RUN.job_name)
        if st and st["info"]:
            html += (f"<div class='h3-st-sub'>ai-toolkit : {st['info']}"
                     f"{' &nbsp;|&nbsp; ' + st['speed'] if st['speed'] else ''}</div>")
    if s["running"] and (s["train_bar"] or s["bar"]):
        html += _bar_html("Progression", s["train_bar"] or s["bar"],
                          "var(--h3-ok)" if s["train_bar"] else "var(--h3-info)",
                          energy=s["kind"] != "download")
    html += _monitor_html()
    extra = []
    if ORCH.thermal_message:
        extra.append(f"<b class='h3-bad'>{ORCH.thermal_message}</b>")
    if ORCH.ram_message:
        extra.append(f"<b class='h3-bad'>{ORCH.ram_message}</b>")
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
        html += "<div class='h3-st-extra'>" + "<br>".join(extra) + "</div>"
    return ("<div class='h3-st'>" + html + "</div>", ORCH.calib_report, ORCH.queue_text(),
            *render_caption_panel(), *render_train_panel(), *render_test_panel(), *render_download_panel())


# ------------------------------------------------------------------ panneaux par onglet
PANEL_CSS = ""   # styles dans UI_CSS (feuille globale)


def _dur(seconds):
    seconds = int(max(0, seconds))
    h, rest = divmod(seconds, 3600)
    m, s_ = divmod(rest, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s_:02d} s" if m else f"{s_} s")


def panel(kinds, idle_text):
    """(html, journal) pour l'onglet des jobs de ce type : etapes, animation, barre et ETA."""
    v = RUN.view(kinds)
    if v is None:
        return PANEL_CSS + f"<div class='h3-idle'>{idle_text}</div>", ""
    if not v["live"]:
        if v["stopped"]:
            icon, color = "&#10074;&#10074;", "var(--h3-warn)"
        elif v["exit_code"] == 0:
            icon, color = "&#10004;", "var(--h3-ok)"
        else:
            icon, color = "&#10008;", "var(--h3-bad)"
        when = time.strftime("%H:%M", time.localtime(v["ended"]))
        html = (f"<div class='h3-st-line'><b style='color:{color}'>{icon}</b> <b>{v['label']}</b> "
                f"&nbsp;-&nbsp; {v['phase']} &nbsp;<span class='h3-muted'>(fin a {when}, "
                f"duree {_dur(v['duration'])})</span></div>")
        spent = core.energy_summary(v.get("wh"), i18n.lang_now())
        if spent:
            html += f"<div class='h3-bar-detail'>Consommation : {spent}</div>"
        return PANEL_CSS + html, v["log"]
    html = (f"<div class='h3-st-line'><span class='h3-spin'></span><b>{v['label']}</b> "
            f"&nbsp;<span class='h3-muted'>en cours depuis {_dur(v['elapsed'])}</span></div>")
    stages = v["stages"]
    if stages:
        now = stages.index(v["stage"]) if v["stage"] in stages else 0
        chips = []
        for i, name in enumerate(stages):
            cls = "done" if i < now else "now" if i == now else ""
            mark = "&#10004; " if i < now else ""
            chips.append(f"<span class='h3-step {cls}'>{mark}{i + 1}. {name}</span>")
        html += "<div class='h3-steps'>" + "".join(chips) + "</div>"
    html += f"<div class='h3-st-sub'>{v['phase']}</div>"
    if v["train_bar"]:
        html += _bar_html("Progression (pas)", v["train_bar"], "var(--h3-ok)", energy=True)
    if v["bar"]:
        title = v["bar"].get("desc") or v["stage"] or "Etape en cours"
        html += _bar_html(title, v["bar"], "var(--h3-info)",
                          energy=not v["train_bar"] and v.get("kind") != "download")
    if not v["train_bar"] and not v["bar"]:
        detail = f"etape en cours depuis {_dur(v['stage_elapsed'])}"
        if v["stage"] == "Legendage" and v["total"]:
            detail = f"0 / {v['total']} images - {detail}"
        if v.get("kind") != "download":
            cost = core.energy_text(None, i18n.lang_now())
            if cost:
                detail += f" &nbsp;|&nbsp; {cost}"
        html += (f"<div class='h3-bar'><div class='h3-indet'></div>"
                 f"<div class='h3-bar-detail'>{detail} "
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
    n = core.effective_count(folder)      # repetitions des sous-dossiers N_nom comprises
    note = (f"{values['steps']} pas pour {n} images (30 passages, 300 pas au minimum)." if n else
            "Dossier du dataset introuvable : 600 pas par defaut, a ajuster (30 x nombre d'images).")
    return [values[k] for k in PROCESS_FIELDS] + [
        f"Procede applique. {note} Lancez le Test rapide pour mesurer la vitesse reelle."]


def load_settings_now(name):
    P = core.load_job_settings(name) if name else None
    if not P:
        return [gr.update()] * len(core.PARAM_KEYS) + ["Aucun reglage enregistre sous ce nom."]
    if P.get("preset") not in core.PRESETS:
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

# ------------------------------------------------------------------ apparence (pur CSS, sans Node.js)
UI_THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="violet", neutral_hue="slate",
    spacing_size="sm", text_size="sm", radius_size="md",
    font=["Inter", "Segoe UI", "system-ui", "-apple-system", "Roboto", "sans-serif"],
    font_mono=["Cascadia Mono", "Consolas", "ui-monospace", "monospace"],
).set(
    block_padding="10px 12px", layout_gap="10px", block_label_text_weight="600",
    button_large_padding="12px 18px", button_primary_text_color="white",
    block_title_background_fill="transparent", block_title_padding="0",
    block_title_text_color="*neutral_700", block_title_text_color_dark="*neutral_200",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_text_color="*neutral_600", block_label_text_color_dark="*neutral_300",
    block_shadow="none", block_border_width="1px",
)

UI_CSS = """
:root{--h3-ok:#22a06b;--h3-info:#4f6bed;--h3-warn:#e08a00;--h3-bad:#e5484d;--h3-accent:#7c5cff;
 --h3-track:#e6e8ef;--h3-card:#ffffff;--h3-border:#e3e6ee}
.dark{--h3-track:#2a2f3c;--h3-card:#171a22;--h3-border:#2b3040}
.gradio-container{max-width:1360px!important;margin:0 auto!important;padding-top:6px!important}
footer{display:none!important}
/* en-tete */
#h3-header{align-items:center;gap:8px;margin-bottom:2px}
.h3-brand{display:flex;align-items:center;gap:12px}
.h3-logo{display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:10px;
 background:linear-gradient(135deg,#6366f1,#a855f7);color:#fff;font-weight:800;font-size:15px;letter-spacing:.5px;
 box-shadow:0 4px 14px #6366f155}
.h3-title{font-size:19px;font-weight:700;line-height:1.1}
.h3-sub{font-size:12px;opacity:.65}
#h3-banner p{margin:0;padding:6px 10px;border-radius:8px;background:#6366f11a;font-size:13px}
#h3-banner:empty,#h3-banner:has(.prose:empty){display:none}
/* barre d'etat */
#h3-statusbar{background:var(--h3-card);border:1px solid var(--h3-border);border-radius:12px;padding:8px 12px;
 align-items:center}
#h3-statusbar > div{padding:0!important}
.h3-cmds{gap:6px!important}
.h3-cmds button{white-space:nowrap;min-width:0!important;flex:0 0 auto!important;width:auto!important;
 padding:4px 12px!important}
#h3-msg textarea{background:transparent!important;border:none!important;box-shadow:none!important;font-size:13px;
 padding:2px 4px!important;min-height:0!important;color:var(--body-text-color-subdued)}
.h3-st{font-size:13px}
.h3-st-line{font-size:13.5px}
.h3-st-sub,.h3-bar-detail,.h3-st-extra{font-size:12px;opacity:.8;margin-top:2px}
.h3-muted{opacity:.65}
.h3-idle{opacity:.6;font-size:13px}
.h3-bad{color:var(--h3-bad)}
.h3-alert{font-size:12px;color:var(--h3-bad);margin-top:2px;font-weight:600}
.h3-dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#9aa3b5;margin-right:6px;
 vertical-align:1px}
.h3-dot.run{background:var(--h3-ok);box-shadow:0 0 0 0 #22a06b88;animation:h3pulse 1.6s infinite}
@keyframes h3pulse{0%{box-shadow:0 0 0 0 #22a06b88}70%{box-shadow:0 0 0 8px #22a06b00}100%{box-shadow:0 0 0 0 #22a06b00}}
.h3-meters{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin-top:6px}
.h3-meter{flex:1;min-width:160px;max-width:300px}
.h3-meter-head{font-size:11.5px;opacity:.85;margin-bottom:2px}
.h3-chips{display:flex;gap:6px;flex-wrap:wrap}
.h3-chip{font-size:11.5px;padding:1px 8px;border-radius:999px;border:1px solid var(--h3-border);opacity:.9}
.h3-chip.hot{border-color:var(--h3-bad);color:var(--h3-bad);font-weight:600}
/* barres */
.h3-bar{margin:6px 0}
.h3-bar-head{display:flex;justify-content:space-between;font-size:12.5px}
.h3-track{background:var(--h3-track);border-radius:999px;height:10px;overflow:hidden}
.h3-track.thin{height:6px}
.h3-fill{height:100%;border-radius:999px;transition:width .5s}
@keyframes h3spin{to{transform:rotate(360deg)}}
@keyframes h3stripe{from{background-position:0 0}to{background-position:40px 0}}
.h3-spin{display:inline-block;width:12px;height:12px;border:2px solid #9aa3b5;border-top-color:var(--h3-accent);
 border-radius:50%;animation:h3spin .8s linear infinite;vertical-align:-2px;margin-right:6px}
.h3-indet{height:10px;border-radius:999px;background:repeating-linear-gradient(45deg,#6366f1 0 10px,#a5b4fc 10px 20px);
 background-size:40px 10px;animation:h3stripe 1s linear infinite}
.h3-steps{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0;font-size:11.5px}
.h3-step{padding:1px 8px;border-radius:999px;border:1px solid var(--h3-border);opacity:.55}
.h3-step.done{opacity:.85;border-color:#22a06b88}
.h3-step.now{opacity:1;border-color:var(--h3-accent);font-weight:600}
.h3-card{background:var(--h3-card);border:1px solid var(--h3-border);border-radius:10px;padding:8px 10px!important}
/* onglets */
#main-tabs > .tab-wrapper button[role=tab]{font-weight:600;font-size:14px;padding:8px 14px}
.tabs .tabs button[role=tab]{font-size:12.5px;padding:5px 10px}
/* colonne d'actions qui reste visible */
.h3-sticky{position:sticky;top:8px;align-self:flex-start}
/* aides : une ligne, texte complet au survol (infobulle) ou pendant la saisie */
.info-text,[data-testid=block-info] + .info{display:-webkit-box!important;-webkit-line-clamp:1;-webkit-box-orient:vertical;
 overflow:hidden;opacity:.7;font-size:11.5px!important;line-height:1.35}
.block:focus-within .info-text{-webkit-line-clamp:unset;opacity:.9}
.h3-note{font-size:12.5px;opacity:.8}
.h3-note p{margin:2px 0 6px}
.h3-group{border-radius:10px!important}
.h3-toolbar{align-items:center}
.h3-inline{align-items:flex-end!important}
.h3-inline > button{margin-bottom:10px}
#h3-statusbar{flex-wrap:nowrap!important}
#h3-statusbar > .h3-cmds{flex:0 0 auto!important;width:auto!important;gap:6px!important;flex-wrap:nowrap!important;
 align-self:flex-start;margin-top:2px}
@media (max-width:900px){#h3-statusbar{flex-wrap:wrap!important}}
#h3-msg:has(textarea:placeholder-shown){display:none}
#h3-msg{margin:-2px 0 2px}
"""

# infobulle : le texte d'aide complet s'affiche en survolant la ligne tronquee ; onglet restaure apres
# un changement de langue
UI_HEAD = """<script>
(() => {
  const tip = () => document.querySelectorAll('.info-text').forEach(e => {
    const t = e.textContent.trim(); if (t && e.title !== t) e.title = t; });
  new MutationObserver(tip).observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener('DOMContentLoaded', tip);
})();
</script>"""


def build(code):
    """Construit l'interface dans une langue (une page par langue : /en/, /fr/, ...)."""
    with gr.Blocks(title="H3-Image LoRA", fill_width=False) as demo:
        # ============================================================ en-tete
        with gr.Row(elem_id="h3-header", equal_height=True):
            gr.HTML("<div class='h3-brand'><span class='h3-logo'>H3</span><div><div class='h3-title'>"
                    "H3-Image LoRA</div><div class='h3-sub'>MiniMax-H3-Image &middot; ai-toolkit &middot; "
                    f"v{core.tool_version()}</div></div></div>")
            lang_sel = gr.Dropdown([(name, c) for c, name in i18n.LANGS], value=code, elem_id="h3-lang",
                                   show_label=False, container=False, interactive=True, scale=0, min_width=150)
        lang_sel.input(None, lang_sel, None, js=i18n.SELECTOR_JS)
        upd_banner = gr.Markdown(elem_id="h3-banner")

        with gr.Row(elem_id="h3-statusbar", equal_height=False):
            with gr.Column(scale=5, min_width=320):
                status = gr.HTML()
            with gr.Row(elem_classes="h3-cmds"):
                pause_btn = gr.Button("Pause (sauvegarder l'etat)", variant="primary", size="sm", scale=0,
                                      min_width=150)
                sample_btn = gr.Button("Apercu maintenant", size="sm", scale=0, min_width=120)
                stop_btn = gr.Button("Arreter sans sauvegarder", variant="stop", size="sm", scale=0,
                                     min_width=150)
        msg = gr.Textbox(label="Message", show_label=False, lines=1, max_lines=4, interactive=False,
                         container=False, elem_id="h3-msg", placeholder=" ")

        with gr.Tabs(elem_id="main-tabs"):
            # ======================================================== 1. Dataset
            with gr.Tab("1. Dataset"):
                with gr.Row():
                    with gr.Column(scale=5, min_width=340):
                        ds_folder = gr.Textbox(label="Dossier du dataset",
                                               value=os.path.join(core.DATASETS, "mon_dataset"),
                                               info="Un dossier par LoRA. Preconise : 15 a 40 images (style : 30 a "
                                                    "100), nettes, variees, cote court >= 1024 px. Sans espaces dans "
                                                    "le chemin.")
                        ds_check = gr.Button("Analyser le dossier", variant="primary")
                        ds_msg = gr.Textbox(label="Etat", lines=9, max_lines=16, interactive=False,
                                            info="Le controle des tailles utilise les resolutions cochees dans "
                                                 "l'onglet Entrainement.")
                        ds_gallery = gr.Gallery(label="Apercu", columns=6, height=220, show_label=False)
                    with gr.Column(scale=6, min_width=380):
                        with gr.Tabs():
                            with gr.Tab("Legendage"):
                                gr.Markdown("Un modele de vision decrit chaque image et ecrit le .txt correspondant. "
                                            "A lancer **avant** l'entrainement (un seul job GPU a la fois). "
                                            "Le modele est telecharge au premier usage.", elem_classes="h3-note")
                                with gr.Row():
                                    lora_type = gr.Radio(core.LORA_TYPES, value=core.LORA_SUBJECT,
                                                         label="Type de LoRA",
                                                         info="Regle la consigne ci-dessous et le controle des "
                                                              "legendes. Personnage : les traits permanents ne sont "
                                                              "pas decrits. Style : seul le contenu est decrit.")
                                cap_extra = gr.Textbox(label="Autres elements permanents a ne pas decrire (optionnel)",
                                                       placeholder="sunglasses, red crescent patch",
                                                       info="En anglais, separes par des virgules. Ex. un accessoire "
                                                            "que le personnage porte sur toutes les images. Il fera "
                                                            "alors partie du mot declencheur et ne pourra plus etre "
                                                            "retire.")
                                with gr.Row():
                                    cap_model = gr.Dropdown(core.CAPTION_MODELS, value=core.CAPTION_MODELS[0],
                                                            label="Modele",
                                                            info="Preconise : 8B (meilleures descriptions, ~10 Go "
                                                                 "de VRAM). 4B / 2B : plus rapides, moins precis.")
                                    cap_variants = gr.Slider(1, 3, value=1, step=1, label="Variantes par image",
                                                             info="2 ou 3 : plusieurs legendes par image, une par "
                                                                  "ligne du .txt ; une variante differente est vue a "
                                                                  "chaque passage, le LoRA s'attache moins a une "
                                                                  "formulation. Utile sous 30 images. Legendage plus "
                                                                  "long d'autant.")
                                with gr.Row():
                                    cap_tokens = gr.Slider(64, 512, value=256, step=32, label="Longueur maximale",
                                                           info="Preconise : 256 (1 a 3 phrases detaillees).")
                                    cap_redo = gr.Checkbox(label="Refaire les legendes existantes", value=False,
                                                           info="Decoche : seules les images sans .txt sont "
                                                                "traitees.")
                                with gr.Accordion("Consigne donnee au modele (modifiable)", open=False):
                                    cap_prompt = gr.Textbox(label="Consigne donnee au modele (modifiable)",
                                                            show_label=False,
                                                            value=core.caption_prompt(core.LORA_SUBJECT), lines=5,
                                                            info="Generee d'apres le type de LoRA, le mot declencheur "
                                                                 "de l'onglet Entrainement et les elements "
                                                                 "permanents. Un modele de vision ne respecte pas "
                                                                 "toujours une interdiction : relancez 'Analyser le "
                                                                 "dossier' apres le legendage pour reperer les "
                                                                 "ecarts.")
                                cap_auto_btn = gr.Button("Lancer le legendage automatique", variant="primary")
                                cap_panel = gr.HTML()
                                with gr.Accordion("Journal", open=False):
                                    cap_log = gr.Textbox(label="Journal du legendage", show_label=False, lines=8,
                                                         max_lines=8, autoscroll=True, interactive=False)
                            with gr.Tab("Correction"):
                                gr.Markdown("Pour un LoRA de personnage, retire des legendes les traits permanents du "
                                            "sujet (cheveux, barbe, carrure, cicatrices, tatouages...) et les elements "
                                            "permanents saisis plus haut. Les expressions et vetements restent ('eyes "
                                            "closed', 'skin-tight suit'). Chaque proposition est modifiable : "
                                            "decochez celles a laisser. Les originales sont sauvegardees a cote du "
                                            "dataset (`<dataset>_legendes_originales`).", elem_classes="h3-note")
                                with gr.Row():
                                    capfix_btn = gr.Button("Proposer les corrections", variant="primary")
                                    capfix_restore_btn = gr.Button("Restaurer les legendes d'origine")
                                capfix_msg = gr.Textbox(label="Message", lines=2, interactive=False)
                                capfix_table = gr.Dataframe(headers=CAPFIX_HEADERS,
                                                            datatype=["bool", "str", "str", "str"], interactive=True,
                                                            type="array", wrap=True, visible=False,
                                                            column_widths=["8%", "14%", "39%", "39%"])
                                capfix_apply_btn = gr.Button("Appliquer les corrections cochees", variant="primary",
                                                             visible=False)
                            with gr.Tab("Audit"):
                                gr.Markdown("Repere ce que l'oeil laisse passer : un meme fond uni sur beaucoup "
                                            "d'images et absent des legendes (il s'attache au mot declencheur : le "
                                            "LoRA le met partout), les quasi-doublons (comptent double), les groupes "
                                            "d'images au meme cadrage (surappris). Fonds : analyse des bords de "
                                            "l'image. Doublons : modele DINOv2 sur le processeur (~90 Mo, telecharge "
                                            "au premier usage).", elem_classes="h3-note")
                                audit_btn = gr.Button("Auditer le dataset", variant="primary")
                                audit_report = gr.Textbox(label="Rapport d'audit", lines=7, interactive=False)
                                audit_fix_btn = gr.Button("Ajouter le fond aux legendes", visible=False,
                                                          variant="primary")
                                audit_gallery = gr.Gallery(label="Quasi-doublons (par paires)", columns=6, height=200)
                                audit_state = gr.State([])
                            with gr.Tab("Rognage"):
                                gr.Markdown(
                                    "Repere les zones de texte (filigrane, signature, adresse de site, legende) et "
                                    "propose, pour chaque image, le plus grand cadrage qui n'en contient aucune. "
                                    "L'image est seulement rognee : rien n'est efface ni repeint. Les originaux sont "
                                    "copies dans un dossier voisin du dataset (`<dataset>_originaux`). Non detectes : "
                                    "les logos sans texte et les filigranes repetes sur toute l'image.",
                                    elem_classes="h3-note")
                                with gr.Row():
                                    crop_keep = gr.Slider(40, 95, value=65, step=5,
                                                          label="Part minimale de l'image a conserver (%)",
                                                          info="Preconise : 65. En dessous, l'image est declaree non "
                                                               "rognable plutot que d'etre trop amputee.")
                                    crop_margin = gr.Slider(0, 5, value=1.5, step=0.5,
                                                            label="Marge autour du texte (% du grand cote)",
                                                            info="Preconise : 1,5. Augmentez si un bout de "
                                                                 "filigrane subsiste.")
                                    crop_conf = gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="Seuil de detection",
                                                          info="Preconise : 0,5. Plus bas : detecte les filigranes "
                                                               "pales, mais aussi plus de faux textes.")
                                crop_scan_btn = gr.Button("Analyser les images", variant="primary")
                                crop_report = gr.Textbox(label="Resultat", lines=5, interactive=False)
                                crop_gallery = gr.Gallery(label="Texte detecte en rouge, cadrage propose en vert, "
                                                                "zone retiree assombrie", columns=4, height=300)
                                crop_pick = gr.CheckboxGroup(choices=[], label="Images a rogner",
                                                             info="Decochez celles ou le texte fait partie de la "
                                                                  "scene.")
                                with gr.Row():
                                    crop_apply_btn = gr.Button("Rogner les images cochees", variant="primary")
                                    crop_restore_btn = gr.Button("Restaurer tous les originaux")
                            with gr.Tab("Fichiers"):
                                ds_files = gr.File(label="Ajouter des images / .txt", file_count="multiple",
                                                   height=140)
                                cap_text = gr.Textbox(label="Legende par defaut pour les images sans .txt",
                                                      info="Depannage seulement : preferez le legendage "
                                                           "automatique ci-dessous.")
                                cap_btn = gr.Button("Creer les legendes manquantes")
                            with gr.Tab("Aide"):
                                gr.Markdown(AIDE_DATASET)

            # ======================================================== 2. Entrainement
            with gr.Tab("2. Entrainement"):
                with gr.Row():
                    with gr.Column(scale=7, min_width=420):
                        with gr.Row(elem_classes="h3-inline"):
                            preset_sel = gr.Dropdown(core.list_job_settings(), label="Reglages enregistres",
                                                     info="Chaque job lance enregistre ses reglages sous son nom.",
                                                     scale=3)
                            preset_load = gr.Button("Charger", size="sm", scale=1, min_width=90)
                            preset_save = gr.Button("Enregistrer les reglages actuels", size="sm", scale=1,
                                                    min_width=120)
                        with gr.Row():
                            job_name = gr.Textbox(label="Nom du job (meme nom = reprise)", value="h3img_mon_lora_v1",
                                                  info="Lettres, chiffres, _ et -. Changez le nom (v2, v3...) pour "
                                                       "repartir de zero ; gardez-le pour reprendre.")
                            trigger = gr.Textbox(label="Mot declencheur (optionnel)",
                                                 info="Preconise : un mot invente avec des chiffres (ex. m41ne) pour "
                                                      "un sujet, jamais un vrai mot ni un nom connu ; vide pour un "
                                                      "style. A reutiliser ensuite dans vos prompts.")
                        process = gr.Radio(core.PROCESSES, value=core.PROC_STANDARD, label="Procede d'entrainement",
                                           info="Standard : vos reglages, inchanges. Rapide 512 : resolution 512, "
                                                "rang 32, optimiseur automagic4, EMA 0,98, 30 passages - bien plus "
                                                "rapide par pas. Recette Fizgig : la meme, plus son adaptateur, "
                                                "l'optimisation de ressemblance et les apercus LCM. Choisir un "
                                                "procede remplit les champs ; tout reste modifiable.")
                        with gr.Group(elem_classes="h3-group"):
                            preset = gr.Radio(core.PRESETS, value=core.PRESET_INT8, label="Preset selon la carte",
                                              info="Preconise sur 24 Go : int8. w4a8 : a essayer avec le Test "
                                                   "rapide ; s'il fonctionne il evite l'aller-retour avec la RAM. "
                                                   "8 Go (RTX 2000 Ada, A2000...) : w4a8, 90 % du transformer en "
                                                   "RAM, 512 px ; environ 40 Go de RAM au pic, 64 Go conseilles, et "
                                                   "nettement plus lent (mesurez avec le Test rapide).")
                            with gr.Row():
                                offload = gr.Slider(0, core.OFFLOAD_MAX, value=0.35, step=0.05,
                                                    label="Part du transformer en RAM", scale=3,
                                                    info="Preconise : 0.35 en int8, 0 en w4a8, 0.9 en 8 Go. Erreur "
                                                         "memoire -> 0.5 puis 0.7 (plus lent). Le Test rapide "
                                                         "indique s'il y a de la marge.")
                                w4a8_fix = gr.Checkbox(label="Correctif w4a8 pour GPU anterieurs aux RTX 40 "
                                                             "(experimental)", value=True, scale=2,
                                                       info="Sans effet en int8. En w4a8 sur 3090 : contourne le "
                                                            "noyau triton incompatible en decompressant en torch "
                                                            "pur. Laissez coche.")
                        resol = gr.CheckboxGroup(["512", "768", "1024", "1536", "2048"], value=["1024", "1536"],
                                                 label="Resolutions d'entrainement",
                                                 info="512 : le reglage de Fizgig, de loin le plus rapide (la memoire "
                                                      "change peu avec la resolution, la duree d'un pas beaucoup). "
                                                      "1024 + 1536 : plus lent, a comparer sur vos apercus. Defaut "
                                                      "ai-toolkit : 512 + 768 + 1024.")
                        with gr.Row():
                            steps = gr.Number(label="Pas", value=2000, precision=0,
                                              info="Preconise : 1500 a 3000 (defaut ai-toolkit : 3000). Repere : "
                                                   "~100 pas par image. Jugez sur les apercus, pas sur le nombre.")
                            lr = gr.Number(label="Learning rate", value=1e-4,
                                           info="adamw8bit : 0.0001 (defaut ai-toolkit) ; brule ou deforme -> "
                                                "0.00005 ; n'apprend rien -> 0.0002. automagic : 0.00025 (point de "
                                                "depart et plafond).")
                            rank = gr.Dropdown([4, 8, 16, 32], value=16, label="Rang LoRA",
                                               info="Preconise : 16 (defaut ai-toolkit pour ce modele). 8 = plus "
                                                    "leger, moins de memoire. 32 = sujets/styles complexes, plus de "
                                                    "memoire.")
                        prompts = gr.Textbox(label="Prompts d'apercu (un par ligne)", value=DEFAULT_PROMPTS, lines=3,
                                             info="Preconise : 2 a 4 prompts detailles, avec votre mot declencheur, "
                                                  "dont un proche du dataset et un tres different (teste la "
                                                  "generalisation). Chaque prompt rallonge la pause d'apercu.")
                        control = gr.Checkbox(value=True,
                                              label="Ajouter un prompt temoin sans mot declencheur (preconise)",
                                              info="Sert a detecter le debordement : le personnage qui apparait meme "
                                                   "sans son mot, signe de sur-apprentissage. Si aucun de vos prompts "
                                                   "ne s'en passe, l'UI ajoute le premier prompt avec 'a person' a la "
                                                   "place du mot. Il est affiche ci-dessous.")
                        prompt_note = gr.Markdown("")
                        with gr.Tabs():
                            with gr.Tab("Apercus et sauvegardes"):
                                with gr.Row():
                                    save_every = gr.Number(label="Sauvegarde tous les N pas", value=250, precision=0,
                                                           info="Preconise : 250. Les 6 derniers checkpoints sont "
                                                                "gardes : le meilleur LoRA est souvent avant le "
                                                                "dernier.")
                                    sample_every = gr.Number(label="Apercus tous les N pas", value=250, precision=0,
                                                             info="Preconise : 250, aligne sur les sauvegardes pour "
                                                                  "pouvoir choisir le checkpoint d'apres ses "
                                                                  "apercus.")
                                    sample_size = gr.Dropdown([1024, 1536, 2048], value=2048,
                                                              label="Taille des apercus",
                                                              info="Preconise : 2048 (le modele vise 4 MP+). Erreur "
                                                                   "memoire pendant les apercus -> 1536. 1024 = "
                                                                   "artefacts normaux.")
                                no_sample = gr.Checkbox(label="Desactiver les apercus", value=False,
                                                        info="A cocher si les apercus saturent la memoire.")
                                ctrl = gr.Textbox(label="Dossier d'images de reference (optionnel)",
                                                  info="Vide = LoRA texte-vers-image (cas normal). Rempli = LoRA "
                                                       "d'edition : memes noms de fichiers que le dataset, references "
                                                       "nommees <Picture 1> dans les legendes.")
                            with gr.Tab("Options du procede"):
                                with gr.Row():
                                    optimizer = gr.Dropdown(list(core.OPTIMIZERS), value=core.OPT_ADAMW,
                                                            label="Optimiseur",
                                                            info="adamw8bit : le taux saisi est applique tel quel. "
                                                                 "automagic : le taux saisi est un point de depart, "
                                                                 "l'optimiseur le regle seul sans depasser 2,5e-4.")
                                    ema = gr.Dropdown(list(core.EMA_CHOICES), value=core.EMA_OFF,
                                                      label="Moyenne des poids (EMA)",
                                                      info="Lisse le LoRA sur les derniers pas : apercus et "
                                                           "checkpoints plus stables. Preconise avec 512 px : 0,98.")
                                with gr.Row():
                                    adapter = gr.Dropdown(core.ADAPTERS, value=core.ADAPTER_OSTRIS,
                                                          label="Adaptateur d'entrainement",
                                                          info="Garde le modele distille stable pendant "
                                                               "l'apprentissage ; jamais dans le LoRA final. Fizgig : "
                                                               "telecharge (~150 Mo) et converti au premier job. "
                                                               "Changez de nom de job quand vous changez "
                                                               "d'adaptateur.")
                                    sampler = gr.Dropdown(core.SAMPLERS, value=core.SAMPLER_EULER,
                                                          label="Echantillonneur des apercus",
                                                          info="N'affecte que les apercus et l'onglet Tester un "
                                                               "LoRA, pas le LoRA. LCM : la meme graine donne une "
                                                               "autre image qu'Euler.")
                                likeness = gr.Checkbox(label="Optimisation de ressemblance (calendrier Fizgig)",
                                                       value=False,
                                                       info="Tire plus souvent le niveau de bruit le plus net, ou se "
                                                            "forment peau, yeux et details : 17 -> 33 -> 50 % aux "
                                                            "passages 1, 3, 5 ; 75 % aux passages 2, 4, 6 et ensuite. "
                                                            "Pour un sujet ; inutile pour un style.")
                            with gr.Tab("Contre le debordement"):
                                gr.Markdown("**Contre le debordement** (le personnage qui apparait partout) : deux "
                                            "protections d'ai-toolkit, combinables. Mesurez leur effet avec la "
                                            "colonne Debordement de Resultats > Ressemblance.", elem_classes="h3-note")
                                with gr.Row():
                                    dop = gr.Checkbox(label="Preservation (DOP)", value=False,
                                                      info="A chaque pas, le LoRA est penalise s'il change ce que le "
                                                           "modele de base dessine pour la legende ou le mot "
                                                           "declencheur est remplace par la classe ci-contre. Aucune "
                                                           "image a preparer, mais chaque pas est environ 1,6 a 2 fois "
                                                           "plus lent et demande plus de memoire : verifiez avec le "
                                                           "Test rapide.")
                                    dop_class = gr.Textbox(label="Classe preservee", value=core.CONTROL_SUBJECT,
                                                           info="En anglais : ce que le mot declencheur designe sans "
                                                                "le LoRA (a person, a man, a woman, a dog...).")
                                with gr.Row():
                                    reg = gr.Checkbox(label="Images de regularisation", value=False,
                                                      info="Melange a l'entrainement des images 'a person...' "
                                                           "produites par le modele de base (dossier "
                                                           "<dataset>_regularisation). Pas de perte de vitesse par "
                                                           "pas, mais elles prennent une part des pas : avec autant "
                                                           "d'images que le dataset, doublez les pas pour garder le "
                                                           "meme apprentissage du sujet.")
                                    reg_weight = gr.Number(label="Poids des images de regularisation", value=1.0,
                                                           minimum=0.1, maximum=2.0, step=0.1,
                                                           info="Preconise : 1. Plus bas : protection plus douce.")
                                with gr.Row(elem_classes="h3-inline"):
                                    reg_count = gr.Number(label="Nombre d'images a generer", value=0, precision=0,
                                                          info="0 = autant que d'images dans le dataset (60 au "
                                                               "plus).")
                                    reg_btn = gr.Button("Generer les images de regularisation", size="sm")
                                reg_msg = gr.Textbox(label="Images de regularisation", interactive=False, lines=2)
                            with gr.Tab("Pause et file"):
                                with gr.Row(elem_classes="h3-inline"):
                                    pause_step = gr.Number(label="Mettre en pause au pas", value=0, precision=0,
                                                           info="0 = desactive. La pause sauvegarde l'etat, comme le "
                                                                "bouton.")
                                    pause_time = gr.Textbox(label="Mettre en pause a (HH:MM)", placeholder="07:30",
                                                            info="Vide = desactive. Si l'heure est passee : demain. "
                                                                 "Utile pour liberer le PC le matin.")
                                    sched_btn = gr.Button("Programmer", size="sm")
                                sched_box = gr.Textbox(label="Pause programmee", interactive=False,
                                                       value=ORCH.schedule_text)
                                queue_box = gr.Textbox(label="File d'attente (jobs lances l'un apres l'autre)",
                                                       lines=3, interactive=False,
                                                       info="Un job en erreur n'arrete pas la file. Une pause ou un "
                                                            "arret manuel la suspend.")
                                with gr.Row():
                                    queue_resume = gr.Button("Reprendre la file")
                                    queue_clear = gr.Button("Vider la file")
                            with gr.Tab("YAML"):
                                prev_btn = gr.Button("Voir le YAML")
                                yaml_box = gr.Code(label="Config generee", language="yaml")
                            with gr.Tab("Aide"):
                                gr.Markdown(AIDE_REGLAGES)
                    with gr.Column(scale=4, min_width=320, elem_classes="h3-sticky"):
                        go_btn = gr.Button("Lancer / Reprendre", variant="primary", size="lg")
                        with gr.Row():
                            calib_btn = gr.Button("Test rapide (20 pas : vitesse + memoire)", size="sm")
                            queue_btn = gr.Button("Ajouter a la file d'attente", size="sm")
                        train_panel = gr.HTML(elem_classes="h3-card")
                        calib_box = gr.Textbox(label="Calibrage (resultat du Test rapide)", lines=5, max_lines=12,
                                               interactive=False,
                                               info="Mesure la vitesse reelle et le pic de VRAM avec les reglages "
                                                    "ci-dessus, puis propose des ajustements et une duree estimee.")
                        with gr.Accordion("Journal", open=False):
                            train_log = gr.Textbox(label="Journal de l'entrainement et du Test rapide",
                                                   show_label=False, lines=12, max_lines=12, autoscroll=True,
                                                   interactive=False)

            # ======================================================== 3. Resultats
            with gr.Tab("3. Resultats"):
                with gr.Row(elem_classes="h3-inline"):
                    job_sel = gr.Dropdown(choices=core.list_jobs(), label="Job", scale=3)
                    refresh_btn = gr.Button("Rafraichir la liste", size="sm", scale=1, min_width=110)
                    open_btn = gr.Button("Ouvrir le dossier du job", size="sm", scale=1, min_width=110)
                res_sig = gr.State("")
                with gr.Row():
                    with gr.Column(scale=5, min_width=360):
                        res_gallery = gr.Gallery(label="Apercus (plus recents d'abord)", columns=3, height=520)
                        with gr.Row(elem_classes="h3-inline"):
                            ck_sel = gr.Dropdown(choices=[], label="Checkpoint a exporter", scale=3,
                                                 info="Le meilleur est souvent un checkpoint intermediaire : "
                                                      "comparez-les dans l'onglet Tester un LoRA.")
                            comfy_btn = gr.Button("Copier dans ComfyUI", size="sm", scale=1, min_width=120)
                        with gr.Accordion("Checkpoints LoRA (.safetensors) - chemins sur le disque", open=False):
                            res_files = gr.Textbox(label="Checkpoints LoRA (.safetensors) - chemins sur le disque",
                                                   show_label=False, lines=5, interactive=False)
                    with gr.Column(scale=6, min_width=380):
                        with gr.Tabs():
                            with gr.Tab("Perte"):
                                loss_plot = gr.HTML()
                                gr.Markdown("*Lecture : la courbe lissee qui devient plate indique que les pas "
                                            "suivants apportent peu ; comparez les checkpoints reperes en vert dans "
                                            "l'onglet Tester un LoRA. La courbe ne detecte pas le sur-apprentissage : "
                                            "seuls des apercus varies le montrent.*", elem_classes="h3-note")
                            with gr.Tab("Ressemblance"):
                                likeness_plot = gr.HTML()
                                gr.Markdown("*Chaque apercu est compare aux images du dataset (ressemblance, modele "
                                            "DINOv2) et a son prompt sans le mot declencheur (fidelite, modele CLIP), "
                                            "sur le processeur : la carte reste a l'entrainement. La ressemblance "
                                            "seule recompenserait le sur-apprentissage : le checkpoint conseille est "
                                            "le plus ressemblant parmi ceux qui suivent encore le prompt comme au "
                                            "debut. Mesures indicatives : verifiez toujours avec Tester un LoRA. "
                                            "Modeles telecharges au premier usage (environ 700 Mo).*",
                                            elem_classes="h3-note")
                            with gr.Tab("Puissance, vitesse et temperature"):
                                perf_plot = gr.HTML()
                                gr.Markdown("*Releve pendant l'entrainement, par tranches de 5 pas (les apercus et "
                                            "sauvegardes sont exclus). Pour trouver la bonne limite : faites varier la "
                                            "limite de puissance (onglet Outils) ou laissez le thermostat la faire "
                                            "bouger, puis lisez le tableau. Le seuil conseille est la puissance la "
                                            "plus basse qui reste a moins de 3 % de la vitesse maximale ; Wh par pas "
                                            "indique l'energie pour un meme travail.*", elem_classes="h3-note")

            # ======================================================== 4. Tester un LoRA
            with gr.Tab("4. Tester un LoRA"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=340):
                        gr.Markdown(
                            "Genere les memes prompts avec plusieurs checkpoints (meme graine) pour les comparer. "
                            "Chaque checkpoint est un job a part qui recharge le modele : comptez plusieurs minutes "
                            "par checkpoint. A lancer quand aucun entrainement ne tourne, ou a mettre en file. "
                            "*Experimental : repose sur un job ai-toolkit a 0 pas d'entrainement.*",
                            elem_classes="h3-note")
                        with gr.Row(elem_classes="h3-inline"):
                            test_job = gr.Dropdown(choices=core.list_jobs(), label="Job", scale=3)
                            test_refresh = gr.Button("Rafraichir", size="sm", scale=1, min_width=90)
                        test_cks = gr.CheckboxGroup(choices=[], label="Checkpoints a comparer",
                                                    info="Preconise : 2 a 4 checkpoints espaces (ex. 1000, 1500, "
                                                         "2000).")
                        test_prompts = gr.Textbox(label="Prompts de test (un par ligne)", lines=3,
                                                  info="Preconise : un prompt proche du dataset, un tres different, "
                                                       "et un SANS le mot declencheur (il ne doit pas faire "
                                                       "apparaitre le sujet).")
                        with gr.Row():
                            test_size = gr.Dropdown([1024, 1536, 2048], value=1536, label="Taille",
                                                    info="Preconise : 1536 pour comparer vite, 2048 pour juger la "
                                                         "qualite finale.")
                            test_seed = gr.Number(label="Graine", value=42, precision=0,
                                                  info="Identique pour tous les checkpoints : seules les differences "
                                                       "dues au LoRA restent visibles.")
                        test_btn = gr.Button("Lancer les tests", variant="primary")
                        test_panel = gr.HTML(elem_classes="h3-card")
                        with gr.Accordion("Journal", open=False):
                            test_log = gr.Textbox(label="Journal des tests", show_label=False, lines=8, max_lines=8,
                                                  autoscroll=True, interactive=False)
                    with gr.Column(scale=7, min_width=380):
                        test_sig = gr.State("")
                        test_gallery = gr.Gallery(label="Resultats (legende : checkpoint | prompt)", columns=3,
                                                  height=640)

            # ======================================================== Outils
            with gr.Tab("Outils"):
                with gr.Row(elem_classes="h3-inline h3-toolbar"):
                    gr.Markdown("Les reglages de cet onglet s'appliquent apres **Enregistrer**.",
                                elem_classes="h3-note")
                    tools_save = gr.Button("Enregistrer", variant="primary", size="sm", scale=0, min_width=130)
                    tools_test = gr.Button("Tester la notification", size="sm", scale=0, min_width=150)
                with gr.Tabs():
                    with gr.Tab("General"):
                        with gr.Row():
                            opt_sound = gr.Checkbox(label="Son a la fin d'un job ou en cas d'erreur",
                                                    value=_cfg["sound"])
                            opt_awake = gr.Checkbox(label="Empecher la mise en veille pendant un job",
                                                    value=_cfg["keep_awake"],
                                                    info="Preconise : coche. L'ecran peut s'eteindre, le PC reste "
                                                         "actif. Sans effet si vous fermez le capot d'un portable.")
                        with gr.Row():
                            opt_like = gr.Checkbox(label="Mesurer la ressemblance des apercus",
                                                   value=lambda: bool(core.load_settings().get("likeness_on", True)),
                                                   info="Onglet Resultats > Ressemblance. Calcul sur le processeur ; "
                                                        "modeles telecharges au premier usage (environ 700 Mo).")
                            opt_leakpause = gr.Checkbox(label="Pause automatique si le debordement devient fort",
                                                        value=lambda: bool(core.load_settings().get("leak_pause",
                                                                                                    False)),
                                                        info="Le personnage apparait sur le prompt temoin, sans son "
                                                             "mot : pause avec sauvegarde. Sinon, simple "
                                                             "notification.")
                        opt_ntfy = gr.Textbox(label="Notification sur telephone : sujet ntfy.sh (optionnel)",
                                              value=_cfg["ntfy_topic"],
                                              info="Installez l'application ntfy, abonnez-vous a un sujet au nom long "
                                                   "et difficile a deviner, et saisissez-le ici. Le nom du job "
                                                   "transite par le service public ntfy.sh ; laissez vide pour ne "
                                                   "rien envoyer.")
                        opt_comfy = gr.Textbox(label="Dossier 'loras' de ComfyUI", value=_cfg["comfy_loras_dir"],
                                               placeholder="D:\\ComfyUI\\models\\loras",
                                               info="Utilise par le bouton 'Copier dans ComfyUI' de l'onglet "
                                                    "Resultats.")
                    with gr.Tab("GPU"):
                        with gr.Row():
                            opt_talert = gr.Number(label="Alerte de temperature (°C)", value=_cfg["temp_alert"],
                                                   precision=0,
                                                   info="Preconise : 5 C sous la limite thermique de la carte (78 "
                                                        "pour une limite a 83). Notification + temperature en "
                                                        "rouge. 0 = desactive.")
                            opt_tpause = gr.Number(label="Pause automatique (°C)", value=_cfg["temp_pause"],
                                                   precision=0,
                                                   info="Preconise : 88. Au-dela pendant 60 s : pause avec "
                                                        "sauvegarde (filet de securite, pas une regulation). 0 = "
                                                        "desactive.")
                            opt_watts = gr.Number(label="Limite de puissance pendant les jobs (W)",
                                                  value=_cfg["power_limit_w"], precision=0,
                                                  info="0 = ne pas toucher. Preconise sur 3090 si elle chauffe : 250 "
                                                       "a 280 (la plage de votre carte s'affiche dessous). Cliquez "
                                                       "Enregistrer : appliquee tout de suite si un job tourne, sinon "
                                                       "au prochain lancement. Windows peut demander une "
                                                       "confirmation.")
                        with gr.Row():
                            opt_thermo = gr.Checkbox(label="Thermostat : ajuster la puissance pour tenir une "
                                                           "temperature", value=_cfg["thermo_on"],
                                                     info="Toutes les 45 s pendant un job : baisse la limite de 10 a "
                                                          "30 W si la carte depasse la cible, la remonte de 10 W si "
                                                          "elle est 3 °C en dessous. Necessite "
                                                          "2_LANCER_UI_ADMIN.bat.")
                            opt_thtarget = gr.Number(label="Temperature cible (°C)", value=_cfg["thermo_target"],
                                                     precision=0,
                                                     info="Preconise : 10 °C sous la limite thermique de la carte "
                                                          "(72 pour une limite a 83).")
                            opt_thmin = gr.Number(label="Puissance plancher (W)", value=_cfg["thermo_min_w"],
                                                  precision=0,
                                                  info="Preconise : 150. Le thermostat ne descend jamais en dessous. "
                                                       "Son plafond est la limite de puissance ci-dessus (ou celle "
                                                       "d'origine si 0).")
                        with gr.Row():
                            opt_sweep = gr.Checkbox(label="Balayage de puissance au debut de l'entrainement",
                                                    value=lambda: bool(core.load_settings().get("sweep_on", False)),
                                                    info="Une fois par job : la limite descend par paliers de 30 pas, "
                                                         "puis le seuil conseille (le plus bas a moins de 3 % de la "
                                                         "vitesse maximale) est applique et enregistre comme limite. "
                                                         "Necessite 2_LANCER_UI_ADMIN.bat. Resultat dans Resultats > "
                                                         "Puissance.")
                            opt_sweepmin = gr.Number(label="Plancher du balayage (W)", precision=0,
                                                     value=lambda: core.load_settings().get("sweep_min_w", 200),
                                                     info="Preconise : 200 sur une 3090. Le balayage part de la "
                                                          "limite ci-dessus (ou celle d'origine si 0).")
                            opt_sweepstep = gr.Number(label="Ecart entre paliers (W)", precision=0,
                                                      value=lambda: core.load_settings().get("sweep_step_w", 30),
                                                      info="Preconise : 30 (environ 5 paliers, 6 a 8 minutes sur une "
                                                           "3090).")
                        # meme action que le bouton Enregistrer de l'onglet : garde pour le cablage, masque
                        temp_save = gr.Button("Enregistrer temperature, puissance et thermostat", visible=False)
                        with gr.Row():
                            thermo_box = gr.Textbox(label="Thermostat (etat et derniers ajustements)",
                                                    interactive=False, value=ORCH.thermo_text, lines=4,
                                                    info="Reflete les reglages ENREGISTRES : une case cochee ou une "
                                                         "valeur modifiee ne compte qu'apres un clic sur Enregistrer "
                                                         "(cocher la case l'enregistre automatiquement).")
                            power_box = gr.Textbox(label="Puissance", interactive=False, value=core.power_text,
                                                   lines=4)
                        with gr.Row():
                            power_apply = gr.Button("Appliquer maintenant la valeur saisie", size="sm")
                            power_reset = gr.Button("Retablir la limite d'origine", size="sm")
                        gr.Markdown("La vitesse des ventilateurs n'est pas pilotee par cette UI (risque en cas de "
                                    "plantage) : reglez une courbe dans MSI Afterburner ou FanControl. Sans droits "
                                    "administrateur, la limite de puissance reste en place apres le job : utilisez "
                                    "'Retablir la limite d'origine' ou redemarrez le PC.", elem_classes="h3-note")
                    with gr.Tab("Electricite"):
                        with gr.Row():
                            opt_price = gr.Number(label="Prix du kWh",
                                                  value=lambda: core.load_settings().get("kwh_price") or 0, minimum=0,
                                                  info="Le montant de votre facture d'electricite, abonnement exclu. "
                                                       "0 = afficher seulement les kWh.")
                            opt_cur = gr.Dropdown([(f"{c} ({v[0]})", c) for c, v in core.CURRENCIES.items()],
                                                  value=lambda: core.currency_for(code), label="Monnaie")
                        gr.Markdown("Pendant un job, la consommation du GPU (lue par nvidia-smi toutes les 2 s), son "
                                    "cout et le cout projete a la fin s'affichent a cote du temps ecoule, du temps "
                                    "restant et de la vitesse. Seule la carte graphique est comptee : processeur, "
                                    "ecran et reste du PC s'y ajoutent. Enregistrez avec le bouton Enregistrer "
                                    "en haut de l'onglet.", elem_classes="h3-note")
                    with gr.Tab("Modeles"):
                        gr.Markdown("Optionnel : sinon les poids se telechargent au premier entrainement. "
                                    "Une coupure reprend automatiquement ; relancer reprend aussi le fichier "
                                    "partiel.", elem_classes="h3-note")
                        with gr.Row(elem_classes="h3-inline"):
                            dl_choice = gr.Radio(["int8", "w4a8", "les deux"], value="int8", label="Transformer",
                                                 scale=3,
                                                 info="Preconise : int8 (correspond au preset par defaut). "
                                                      "Prevoir 35 a 50 Go libres selon le choix.")
                            dl_btn = gr.Button("Telecharger", variant="primary", size="sm", scale=1)
                        dl_panel = gr.HTML(elem_classes="h3-card")
                        with gr.Accordion("Journal", open=False):
                            dl_log = gr.Textbox(label="Journal du telechargement", show_label=False, lines=8,
                                                max_lines=8, autoscroll=True, interactive=False)
                        dl_state = gr.Textbox(label="Etat des fichiers", lines=5, interactive=False,
                                              value=core.models_state)
                        dl_refresh = gr.Button("Rafraichir l'etat", size="sm")
                    with gr.Tab("Versions"):
                        with gr.Row():
                            with gr.Column():
                                upd_box = gr.Textbox(label="Versions", lines=4, interactive=False)
                                upd_btn = gr.Button("Verifier les mises a jour", size="sm")
                            with gr.Column():
                                ver_box = gr.Textbox(label="Version d'ai-toolkit", lines=4, interactive=False,
                                                     value=core.version_text)
                                with gr.Row():
                                    pin_btn = gr.Button("Figer la version actuelle", size="sm")
                                    unpin_btn = gr.Button("Defiger", size="sm")
                        gr.Markdown("Preconise : figer apres un premier entrainement reussi. Une version figee n'est "
                                    "plus modifiee par 3_METTRE_A_JOUR.bat tant que vous ne la defigez pas.",
                                    elem_classes="h3-note")

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
        cap_auto_btn.click(lambda f, m, r, p, t, v: core.start_captioning(f, m, r, p, t, v),
                           [ds_folder, cap_model, cap_redo, cap_prompt, cap_tokens, cap_variants], msg,
                           show_progress="minimal").then(render_caption_panel, None, [cap_panel, cap_log])
        # entrainement
        # .input : seulement quand l'utilisateur change le preset, pas quand des reglages sont recharges
        def preset_defaults(p):
            d = core.PRESET_DEFAULTS.get(p, {})
            return (d.get("offload", 0.0), gr.update(value=d["resolutions"]) if "resolutions" in d else gr.update(),
                    gr.update(value=d["sample_size"]) if "sample_size" in d else gr.update())
        preset.input(preset_defaults, preset, [offload, resol, sample_size])
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
        # verification en arriere-plan (au plus toutes les 12 h) : la page n'attend pas le reseau
        demo.load(core.update_banner, None, upd_banner).then(core.update_text, None, upd_box)
        upd_btn.click(lambda: core.update_text(True), None, upd_box).then(core.update_banner, None, upd_banner)
        demo.load(restore_page, None, cfg_inputs + [preset_sel, job_sel, test_job])

    # champs dont le contenu appartient a l'utilisateur (chemins, prompts, legendes, journaux)
    keep = [ds_folder, cap_text, cap_extra, cap_prompt, cap_log, job_name, trigger, ctrl, prompts, dop_class,
            train_log, pause_time, test_prompts, test_log, dl_log, opt_ntfy, opt_comfy]
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
    threading.Thread(target=core.check_updates, daemon=True).start()   # mises a jour, sans bloquer
    app = FastAPI()

    @app.get("/")
    def _root():
        return RedirectResponse(f"/{i18n.saved_lang()}/")

    for code, name in i18n.LANGS:
        print(f"  interface {name} ...", flush=True)
        # ssr_mode=False : rendu dans le navigateur, jamais de serveur Node.js
        app = gr.mount_gradio_app(app, build(code).queue(), path=f"/{code}", show_error=True,
                                  allowed_paths=[core.ROOT], js=i18n.RESTORE_TAB_JS, ssr_mode=False,
                                  theme=UI_THEME, css=UI_CSS, head=UI_HEAD, footer_links=[])
    url = "http://127.0.0.1:7860/"
    print(f"UI : {url}   ({core.MODELS})", flush=True)
    if os.environ.get("H3_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")


if __name__ == "__main__":
    main()
