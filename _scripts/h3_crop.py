"""Rognage automatique des filigranes et textes d'un dataset.

Principe : un detecteur de texte repere les zones ecrites (filigrane, signature,
legende, adresse de site) ; pour chaque image on cherche le plus grand rectangle
qui n'en contient aucune. Rien n'est efface ni repeint : l'image est seulement
rognee, donc aucun pixel invente n'entre dans l'entrainement.

Detecteur : PP-OCRv3 (2,4 Mo, format ONNX, depot officiel opencv_zoo), execute
par OpenCV, deja installe avec ai-toolkit. S'il ne peut pas etre telecharge, un
detecteur de secours purement geometrique prend le relais (moins fiable).

Limites : un logo sans texte ou un filigrane en mosaique sur toute l'image ne
sont pas detectes ; un texte au centre de l'image ne peut pas etre rogne.
"""

import glob
import hashlib
import os
import shutil
import urllib.request

from PIL import Image, ImageDraw

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx")
MODEL_SHA256 = "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587"
MODEL_NAME = "text_detection_en_ppocrv3_2023may.onnx"

_detector = {"model": None, "kind": ""}   # charge une seule fois
_last = {"folder": None, "items": {}}     # derniere analyse : nom -> resultat


def backup_dir(folder):
    """Dossier des originaux : A COTE du dataset, jamais dedans (ai-toolkit lit
    les sous-dossiers du dataset et entrainerait aussi sur les sauvegardes)."""
    folder = os.path.abspath(folder).rstrip("\\/")
    return folder + "_originaux"


# ------------------------------------------------------------------ detecteur
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(models_dir):
    """Chemin du modele de detection, telecharge au besoin ; None si indisponible."""
    target_dir = os.path.join(models_dir, "outils")
    path = os.path.join(target_dir, MODEL_NAME)
    if os.path.isfile(path) and _sha256(path) == MODEL_SHA256:
        return path
    try:
        os.makedirs(target_dir, exist_ok=True)
        tmp = path + ".part"
        with urllib.request.urlopen(MODEL_URL, timeout=30) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if _sha256(tmp) != MODEL_SHA256:
            os.remove(tmp)
            return None
        os.replace(tmp, path)
        return path
    except Exception:
        return None


def _load_detector(models_dir):
    if _detector["kind"]:
        return
    try:
        import cv2

        path = ensure_model(models_dir)
        if path and hasattr(cv2.dnn, "TextDetectionModel_DB"):
            model = cv2.dnn.TextDetectionModel_DB(path)
            model.setBinaryThreshold(0.3)
            model.setPolygonThreshold(0.5)
            model.setMaxCandidates(200)
            model.setUnclipRatio(2.0)
            _detector.update(model=model, kind="ppocr")
            return
        _detector.update(model=None, kind="secours")
    except Exception:
        _detector.update(model=None, kind="indisponible")


def detector_label():
    return {"ppocr": "detecteur PP-OCRv3", "secours": "detecteur de secours (moins fiable)",
            "indisponible": "detection indisponible (OpenCV manquant)"}.get(_detector["kind"], "")


def _detect_ppocr(img, min_conf):
    import cv2
    import numpy as np

    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    H, W = arr.shape[:2]
    scale = min(1.0, 1280 / max(H, W))
    w = max(32, int(W * scale) // 32 * 32)   # le reseau exige des multiples de 32
    h = max(32, int(H * scale) // 32 * 32)
    model = _detector["model"]
    model.setInputParams(1 / 255.0, (w, h), (122.67891434, 116.66876762, 104.00698793))
    quads, confs = model.detect(cv2.resize(arr, (w, h)))
    boxes = []
    for quad, conf in zip(quads, confs):
        if float(conf) < min_conf:
            continue
        q = np.array(quad, dtype=np.float32)
        x0, x1 = q[:, 0].min() * W / w, q[:, 0].max() * W / w
        y0, y1 = q[:, 1].min() * H / h, q[:, 1].max() * H / h
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes


def _detect_fallback(img):
    """Sans modele : zones allongees a fort contraste local, typiques d'une ligne de texte."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    scale = min(1.0, 1280 / max(H, W))
    small = cv2.resize(gray, (int(W * scale), int(H * scale)))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        fill = cv2.contourArea(c) / max(1, w * h)
        if w >= 40 and 8 <= h <= 0.12 * small.shape[0] and w / h >= 3.5 and fill >= 0.45:
            boxes.append((int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale)))
    return boxes


def detect_text(img, models_dir, min_conf=0.5):
    _load_detector(models_dir)
    if _detector["kind"] == "ppocr":
        return _detect_ppocr(img, min_conf)
    if _detector["kind"] == "secours":
        return _detect_fallback(img)
    return []


# ------------------------------------------------------------------ meilleur rognage
def best_crop(W, H, boxes, margin=0):
    """Plus grand rectangle (l, t, r, b) ne touchant aucune boite, ou None.

    Un bord du rognage optimal passe toujours soit par un bord de l'image, soit
    juste au-dela d'une boite : on essaie toutes ces positions."""
    if not boxes:
        return (0, 0, W, H)
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:14]
    grown = [(max(0, x0 - margin), max(0, y0 - margin), min(W, x1 + margin), min(H, y1 + margin))
             for x0, y0, x1, y1 in boxes]
    lefts = sorted({0} | {b[2] for b in grown if b[2] < W})
    rights = sorted({W} | {b[0] for b in grown if b[0] > 0})
    tops = sorted({0} | {b[3] for b in grown if b[3] < H})
    bottoms = sorted({H} | {b[1] for b in grown if b[1] > 0})
    best, best_area = None, 0
    for l in lefts:
        for r in rights:
            if r - l <= 0 or (r - l) * H <= best_area:
                continue
            cols = [b for b in grown if b[0] < r and b[2] > l]   # boites dans la bande verticale
            for t in tops:
                for b_ in bottoms:
                    area = (r - l) * (b_ - t)
                    if b_ - t <= 0 or area <= best_area:
                        continue
                    if all(not (b[1] < b_ and b[3] > t) for b in cols):
                        best, best_area = (l, t, r, b_), area
    return best


# ------------------------------------------------------------------ analyse d'un dossier
def _preview(img, boxes, crop, size=420):
    """Vignette : zone retiree assombrie, texte detecte en rouge, rognage en vert."""
    im = img.convert("RGB")
    W, H = im.size
    if crop and crop != (0, 0, W, H):
        dark = Image.eval(im, lambda v: v // 3)
        dark.paste(im.crop(crop), crop[:2])
        im = dark
    draw = ImageDraw.Draw(im)
    line = max(3, max(W, H) // 250)
    for b in boxes:
        draw.rectangle(b, outline=(255, 40, 40), width=line)
    if crop and crop != (0, 0, W, H):
        draw.rectangle((crop[0], crop[1], crop[2] - 1, crop[3] - 1), outline=(60, 220, 90), width=line)
    im.thumbnail((size, size))
    return im


def analyze_folder(folder, models_dir, min_keep=0.65, margin_pct=1.5, min_conf=0.5, progress=None):
    """Retourne (rapport texte, [(vignette, legende)], noms rognables).

    min_keep : part minimale de l'image a conserver pour proposer le rognage."""
    if not folder or not os.path.isdir(folder):
        return "Dossier introuvable.", [], []
    if progress:
        progress(0, 1, "Chargement du detecteur de texte (telecharge au premier usage)")
    _load_detector(models_dir)
    if _detector["kind"] == "indisponible":
        return "Detection impossible : OpenCV n'est pas installe dans cet environnement.", [], []
    import h3_core as core
    names = core.dataset_images(folder)
    items, gallery = {}, []
    clean = croppable = hopeless = 0
    small_after = []
    for index, name in enumerate(names):
        if progress:
            progress(index, len(names), name)
        try:
            with Image.open(os.path.join(folder, name)) as im:
                im.load()
                img = im.copy()
        except Exception:
            continue
        W, H = img.size
        boxes = detect_text(img, models_dir, min_conf)
        if not boxes:
            clean += 1
            continue
        margin = int(max(W, H) * margin_pct / 100)
        crop = best_crop(W, H, boxes, margin)
        keep = ((crop[2] - crop[0]) * (crop[3] - crop[1]) / (W * H)) if crop else 0.0
        ok = crop is not None and keep >= min_keep
        items[name] = {"crop": crop if ok else None, "keep": keep, "size": (W, H)}
        if ok:
            croppable += 1
            cw, ch = crop[2] - crop[0], crop[3] - crop[1]
            label = f"{name} | garde {keep * 100:.0f} % -> {cw}x{ch}"
            if cw * ch < 500_000:
                small_after.append(name)
        else:
            hopeless += 1
            label = f"{name} | NON ROGNABLE (ne garderait que {keep * 100:.0f} %)"
        gallery.append((_preview(img, boxes, crop if ok else None), label))
    _last.update(folder=os.path.abspath(folder), items=items)
    lines = [f"{len(names)} image(s) analysee(s) avec le {detector_label()}.",
             f"- Sans texte detecte : {clean}",
             f"- Rognables : {croppable} (en vert sur les vignettes : la zone gardee)",
             f"- Texte trop central pour etre rogne : {hopeless}"]
    if hopeless:
        bad = [n for n, it in items.items() if it["crop"] is None]
        lines.append("  A retirer du dataset ou a nettoyer avec un outil de retouche : " + ", ".join(bad[:10]))
    if small_after:
        lines.append(f"- Attention : {len(small_after)} image(s) passeraient sous 0,5 megapixel apres rognage : "
                     + ", ".join(small_after[:8]))
    if croppable:
        lines.append("Verifiez les vignettes : un texte qui fait partie de la scene (enseigne, graffiti, "
                     "maillot) est detecte aussi. Decochez ces images avant d'appliquer.")
    return "\n".join(lines), gallery, [n for n, it in items.items() if it["crop"]]


def _drop_latent_cache(folder, name):
    """ai-toolkit met en cache l'image encodee sous <nom>_<hash>.safetensors : le
    hash ne depend pas du contenu, donc une image rognee garderait l'ancien cache."""
    sub, base = os.path.split(name)          # image d'un sous-dossier : cache dans ce sous-dossier
    stem = os.path.splitext(base)[0]
    for p in glob.glob(os.path.join(folder, sub, "_latent_cache", glob.escape(stem) + "_*.safetensors")):
        try:
            os.remove(p)
        except OSError:
            pass


def apply_crops(folder, names):
    """Rogne les images choisies ; les originaux sont copies a cote du dataset."""
    if not folder or os.path.abspath(folder) != _last["folder"]:
        return "Relancez d'abord l'analyse de ce dossier."
    if not names:
        return "Aucune image cochee."
    bdir = backup_dir(folder)
    os.makedirs(bdir, exist_ok=True)
    done, skipped = 0, []
    for name in names:
        item = _last["items"].get(name)
        src = os.path.join(folder, name)
        if not item or not item["crop"] or not os.path.isfile(src):
            skipped.append(name)
            continue
        try:
            with Image.open(src) as im:
                im.load()
                if im.size != item["size"]:      # fichier modifie depuis l'analyse
                    skipped.append(name)
                    continue
                keep = os.path.join(bdir, name)
                os.makedirs(os.path.dirname(keep), exist_ok=True)
                if not os.path.exists(keep):      # ne jamais ecraser le tout premier original
                    shutil.copy2(src, keep)
                out = im.crop(item["crop"])
                params = {"quality": 95, "subsampling": 0} if name.lower().endswith((".jpg", ".jpeg")) \
                    else {"quality": 95} if name.lower().endswith(".webp") else {}
                if out.mode in ("RGBA", "P") and name.lower().endswith((".jpg", ".jpeg")):
                    out = out.convert("RGB")
                out.save(src, **params)
            _drop_latent_cache(folder, name)
            item["crop"] = None                   # deja applique
            done += 1
        except Exception as exc:
            skipped.append(f"{name} ({exc})")
    msg = f"{done} image(s) rognee(s). Originaux conserves dans : {bdir}"
    if skipped:
        msg += "\nIgnorees : " + ", ".join(skipped[:10])
    if done:
        msg += ("\nLes legendes .txt ne sont pas modifiees : si l'une mentionnait le texte retire, "
                "corrigez-la. Le cache des images rognees a ete vide ; il se refera au prochain job.")
    return msg


def restore_originals(folder):
    bdir = backup_dir(folder)
    if not os.path.isdir(bdir):
        return "Aucune sauvegarde d'originaux pour ce dossier."
    n = 0
    for root, _dirs, files in os.walk(bdir):
        for base in files:
            if base.lower().endswith(IMG_EXT):
                name = os.path.relpath(os.path.join(root, base), bdir)
                os.makedirs(os.path.dirname(os.path.join(folder, name)), exist_ok=True)
                shutil.copy2(os.path.join(root, base), os.path.join(folder, name))
                _drop_latent_cache(folder, name)
                n += 1
    _last.update(folder=None, items={})
    return f"{n} original(aux) restaure(s) depuis {bdir}. Relancez l'analyse si besoin."
