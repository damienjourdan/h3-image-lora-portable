"""Audit du dataset par l'image : fonds unis non decrits, quasi-doublons, cadrages trop semblables.

- Fond uni : analyse des bords de l'image (sans modele). Un fond identique sur beaucoup d'images et
  absent des legendes s'attache au mot declencheur : le LoRA le reproduit partout (fond blanc).
- Doublons et images tres semblables : similarite DINOv2 entre images (le petit modele de l'onglet
  Ressemblance, ~90 Mo, sur le processeur).
"""

import os
import re

import h3_core as core

DUP = 0.95        # similarite DINOv2 : quasi-doublon (meme image recadree, retouchee, compressee)
SIMILAR = 0.90    # meme cadrage, meme pose, meme fond
BG_SHARE = 0.30   # part du dataset a partir de laquelle un meme fond devient un probleme
BG_WORDS = re.compile(r"\b(background|backdrop|backgrounds)\b", re.I)


def background_kind(path):
    """'white background', 'black background', 'gray background', 'simple background' ou None.
    On regarde une bande de 4 % sur les quatre bords : couleur quasi constante = fond uni."""
    from PIL import Image

    try:
        im = Image.open(path).convert("RGB")
    except OSError:
        return None
    im.thumbnail((256, 256))
    w, h = im.size
    b = max(2, int(min(w, h) * 0.04))
    px = im.load()
    pts = []
    for x in range(0, w, 2):
        for y in list(range(b)) + list(range(h - b, h)):
            pts.append(px[x, y])
    for y in range(b, h - b, 2):
        for x in list(range(b)) + list(range(w - b, w)):
            pts.append(px[x, y])
    if not pts:
        return None
    mean = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
    close = sum(1 for p in pts if max(abs(p[i] - mean[i]) for i in range(3)) <= 28)
    if close / len(pts) < 0.90:
        return None
    lum = sum(mean) / 3
    sat = max(mean) - min(mean)
    if lum > 225 and sat < 25:
        return "white background"
    if lum < 35:
        return "black background"
    if sat < 25:
        return "gray background"
    return "simple background"


def dhash(path, size=8):
    """Empreinte de 64 bits (gradient horizontal d'une miniature en niveaux de gris)."""
    from PIL import Image
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(im.getdata())
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return bits


def _caption(folder, img):
    path = os.path.join(folder, os.path.splitext(img)[0] + ".txt")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def audit(folder, progress=None):
    """Rend (rapport texte, paires a afficher [(chemin, legende)], corrections [(image, fond)])."""
    if not folder or not os.path.isdir(folder):
        return "Dossier introuvable.", [], []
    imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith(core.IMG_EXT))
    if len(imgs) < 2:
        return "Pas assez d'images a comparer.", [], []
    lines = [f"Audit de {len(imgs)} images :"]

    # ---- fonds unis
    kinds = {}
    for i, f in enumerate(imgs):
        if progress:
            progress(i, len(imgs) * 2, f)
        k = background_kind(os.path.join(folder, f))
        if k:
            kinds[f] = k
    fixes = []
    if kinds:
        counts = {}
        for k in kinds.values():
            counts[k] = counts.get(k, 0) + 1
        detail = ", ".join(f"{n} x {k}" for k, n in sorted(counts.items(), key=lambda x: -x[1]))
        undescribed = [f for f in kinds if not BG_WORDS.search(_caption(folder, f) or "")]
        fixes = [(f, kinds[f]) for f in undescribed if _caption(folder, f) is not None]
        if len(kinds) >= max(2, BG_SHARE * len(imgs)):
            lines.append(f"- Fond uni sur {len(kinds)}/{len(imgs)} image(s) ({detail}) : s'il n'est pas decrit, "
                         "il s'attache au mot declencheur et le LoRA le reproduit partout.")
            if undescribed:
                lines.append(f"- {len(undescribed)} legende(s) ne mentionnent pas ce fond : bouton 'Ajouter le fond "
                             "aux legendes', ou decrivez-le vous-meme.")
            else:
                lines.append("- Toutes ces legendes mentionnent le fond : rien a faire.")
        else:
            lines.append(f"- Fond uni sur {len(kinds)}/{len(imgs)} image(s) ({detail}) : minoritaire, sans risque "
                         "si c'est decrit dans les legendes.")
    else:
        lines.append("- Aucun fond uni detecte.")

    # ---- doublons et cadrages semblables (DINOv2)
    from PIL import Image

    import h3_likeness as likeness

    if progress:
        progress(len(imgs), len(imgs) * 2, "Chargement du modele d'analyse (telecharge au premier usage, ~90 Mo)")
    embs = []
    for i in range(0, len(imgs), 8):
        batch = [Image.open(os.path.join(folder, f)).convert("RGB") for f in imgs[i:i + 8]]
        embs.append(likeness.SCORER.embed(batch))
        if progress:
            progress(len(imgs) + min(i + 8, len(imgs)), len(imgs) * 2, imgs[min(i + 7, len(imgs) - 1)])
    import torch
    emb = torch.cat(embs)
    sim = (emb @ emb.T).tolist()
    hashes = [dhash(os.path.join(folder, f)) for f in imgs]
    off = [sim[a][b] for a in range(len(imgs)) for b in range(a + 1, len(imgs))]
    mean = sum(off) / len(off)
    std = (sum((v - mean) ** 2 for v in off) / len(off)) ** 0.5
    # "tres semblable" : nettement au-dessus de la similarite habituelle de CE dataset
    # (un meme personnage sur un meme fond est deja tres similaire partout)
    thr = max(SIMILAR, mean + 2 * std)
    dups, similar = [], []
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            near = bin(hashes[a] ^ hashes[b]).count("1") <= 10   # meme image a peu pres pixel pour pixel
            if sim[a][b] >= DUP and near:
                dups.append((a, b, sim[a][b]))
            elif sim[a][b] >= thr:
                similar.append((a, b, sim[a][b]))
    gallery = []
    if dups:
        lines.append(f"- {len(dups)} quasi-doublon(s) : un doublon compte double et pousse au sur-apprentissage. "
                     "Gardez une image par paire :")
        for a, b, v in sorted(dups, key=lambda x: -x[2])[:12]:
            lines.append(f"    {imgs[a]} = {imgs[b]} ({v * 100:.0f} %)")
            gallery += [(os.path.join(folder, imgs[a]), f"{imgs[a]} ({v * 100:.0f} %)"),
                        (os.path.join(folder, imgs[b]), imgs[b])]
    else:
        lines.append("- Aucun quasi-doublon.")
    if similar:
        # groupes d'images tres semblables (composantes connexes)
        parent = list(range(len(imgs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b, _v in similar + dups:
            parent[find(a)] = find(b)
        groups = {}
        for i in range(len(imgs)):
            groups.setdefault(find(i), []).append(imgs[i])
        groups = [g for g in groups.values() if len(g) >= 3]
        if groups:
            lines.append("- Groupes d'images tres semblables (meme cadrage, pose ou fond) : variez-les ou "
                         "gardez-en moins, sinon ce cadrage sera surappris :")
            for g in sorted(groups, key=len, reverse=True)[:6]:
                lines.append(f"    {len(g)} images : {', '.join(g[:8])}{' ...' if len(g) > 8 else ''}")
        else:
            lines.append(f"- {len(similar)} paire(s) d'images tres semblables, sans groupe important : acceptable.")
    else:
        lines.append("- Pas de groupe d'images trop semblables : bonne variete.")
    return "\n".join(lines), gallery, fixes


def add_background(folder, fixes):
    """Ajoute ', <fond>' a la fin des legendes qui ne le mentionnent pas encore."""
    done = 0
    for img, kind in fixes or []:
        path = os.path.join(folder, os.path.splitext(img)[0] + ".txt")
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
            if BG_WORDS.search(text):
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{text.rstrip(' ,.')}, {kind}")
            done += 1
        except OSError:
            continue
    # pas de cache a vider : ai-toolkit range les textes encodes sous une empreinte de la legende
    return f"{done} legende(s) completee(s) avec la description du fond."
