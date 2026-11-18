"""Correction automatique des legendes d'un LoRA de personnage : retire les traits permanents.

Ce qui est decrit dans une legende s'attache aux mots de la legende ; ce qui est tu s'attache au mot
declencheur. Les traits permanents (cheveux, barbe, carrure, cicatrices...) doivent donc etre tus.
Correction par regles, sans modele, sur les memes mots que le controle des legendes :
- un groupe nominal autour d'un nom de trait est retire ("with short spiky blond hair") ;
- un adjectif de trait est retire seul ("a muscular man" -> "a man", "white-haired" -> retire) ;
- les expressions variables sont gardees ("eyes closed", "skin-tight suit").
Chaque proposition est relue, modifiable et a cocher ; les originales sont sauvegardees a cote du dataset.
"""

import os
import re
import shutil

import h3_core as core

NOUNS = {"hair", "hairstyle", "mohawk", "beard", "goatee", "mustache", "moustache", "stubble",
         "muscles", "physique", "scar", "scars", "wrinkle", "wrinkles", "tattoo", "tattoos", "freckles",
         "skin", "eyes", "eyebrows", "jawline", "veins"}
ADJS = {"blond", "blonde", "brunette", "redhead", "bald", "balding", "bearded", "muscular", "muscled",
        "scarred", "wrinkled", "tattooed", "tanned", "middle-aged", "elderly", "veined", "haired"}
# mots qui arretent le groupe nominal en remontant vers la gauche (ils restent)
STOP = {"with", "and", "or", "has", "have", "having", "sporting", "featuring", "showing", "of", "in", "on",
        "is", "are", "wearing", "wears", "while", "who", "which", "that", "to", "from", "by", "under", "over",
        ",", ";", "."}
DETS = {"a", "an", "the", "his", "her", "their", "its", "some"}
# apres le nom : expression ou vetement, pas un trait permanent
KEEP_AFTER = {"eyes": ("closed", "open", "wide", "shut", "half", "squinting", "narrowed", "looking"),
              "skin": ("-tight", "tight", "suit")}
KEEP_BEFORE = {"eyes": ("closed", "covering", "covers", "shielding", "rubbing", "over", "shading", "tired",
                        "sleepy", "teary", "tearful", "wide", "narrowed", "squinting", "half-closed", "rolling")}
# complement de lieu d'un trait ("tattoos on his arms", "scar over the left eyebrow") : retire avec lui
TRAIT_PREP = {"on", "over", "across", "along", "covering", "running", "down", "around"}
STOP_FWD = {",", ";", ".", "and", "with", "while", "holding", "wearing", "standing", "sitting", "who"}
SUFFIX_HAIRED = re.compile(r"\b[\w]+-haired\b", re.I)


def _fix_clause(clause, extras):
    text = clause
    for ex in extras:   # elements permanents ajoutes par l'utilisateur : retires tels quels
        text = re.sub(r"\b(?:with |and )?(?:a |an |his |her |the )?" + re.escape(ex) + r"\b", " ", text,
                      flags=re.I)
    text = SUFFIX_HAIRED.sub(" ", text)
    words = re.findall(r"[\w'-]+|[^\w\s]", text)
    low = [w.lower() for w in words]
    drop = set()
    for i, w in enumerate(low):
        if w in NOUNS:
            after = low[i + 1] if i + 1 < len(low) else ""
            before = low[i - 1] if i > 0 else ""
            before2 = {low[k] for k in range(max(0, i - 2), i)}
            if any(after.startswith(k) for k in KEEP_AFTER.get(w, ())) or before2 & set(KEEP_BEFORE.get(w, ())):
                continue
            j = i
            while j - 1 >= 0 and low[j - 1] not in STOP and i - (j - 1) <= 5:
                j -= 1
                if low[j] in DETS:
                    break
            k = i
            if after in TRAIT_PREP or (after.endswith("ed") and len(after) > 3):   # "hair spiked up"
                k = i + 1
                while k + 1 < len(low) and low[k + 1] not in STOP_FWD and k - i <= 5:
                    k += 1
            drop.update(range(j, k + 1))
        elif w in ADJS:
            drop.add(i)
    kept = [w for i, w in enumerate(words) if i not in drop]
    out = " ".join(kept)
    out = re.sub(r"\s+([,;.!?])", r"\1", out)
    # liaisons orphelines : "with and", "and and", "with" ou "and" en bout de groupe
    for _ in range(3):
        out = re.sub(r"\b(with|and|or|has|having|sporting|featuring)\s+(and|or|with)\b", r"\1", out, flags=re.I)
        out = re.sub(r"\b(with|and|or|has|having|sporting|featuring|of)\s*$", "", out.strip(), flags=re.I)
        out = re.sub(r"^\s*(and|or|with)\b", "", out.strip(), flags=re.I)
        out = re.sub(r"\b(a|an|the|his|her)\s+(?=(and|with|or)\b)", "", out.strip(), flags=re.I)
        out = re.sub(r"\b(a|an|the|his|her|is|are|wearing|wears|has)\s*$", "", out.strip(), flags=re.I)
    out = re.sub(r"\b(a|an)\s+(?=[aeiou])", "an ", out, flags=re.I)
    out = re.sub(r"\ban\s+(?=[^aeiou\W])", "a ", out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if not re.search(r"[A-Za-z]{2,}", out) or out.lower() in DETS:
        return ""
    return out


def fix_caption(text, extras=(), trigger=""):
    """Legende sans traits permanents ; le mot declencheur n'est jamais touche."""
    trig = (trigger or "").strip()
    text = re.sub(r"\b[\w]+-haired\b\s*,?\s*", "", text, flags=re.I)
    # liste d'adjectifs de trait separes par des virgules ("a bald, muscular man")
    text = re.sub(r"\b(" + "|".join(sorted(ADJS, key=len, reverse=True)) + r")\s*,\s*", "", text, flags=re.I)
    parts = re.split(r"(\s*[,;]\s*|(?<=\.)\s+)", text.strip())
    out = []
    for k in range(0, len(parts), 2):
        clause = parts[k]
        if trig and clause.strip().lower() == trig.lower():
            fixed = clause.strip()
        else:
            fixed = _fix_clause(clause, extras)
        if fixed:
            out.append(fixed)
    res = ", ".join(c.rstrip(",;") for c in out if c)
    res = re.sub(r"\.\s*,", ".", res)
    return res


def proposals(folder, lora_type, trigger="", extra_traits=""):
    """Lignes [appliquer, fichier, legende actuelle, legende proposee] des legendes a corriger."""
    if not folder or not os.path.isdir(folder):
        return [], "Dossier introuvable."
    if lora_type != core.LORA_SUBJECT:
        return [], ("La correction automatique ne concerne que les LoRA de personnage / sujet : pour un "
                    "style, les traits du contenu doivent rester decrits.")
    extras = [t.strip() for t in re.split(r"[,;\n]", extra_traits or "") if t.strip()]
    rows = []
    for img in core.dataset_images(folder):
        f = os.path.splitext(img)[0] + ".txt"
        old = core.read_caption(folder, img)
        if not old:
            continue
        # une variante par ligne : chacune est corrigee
        new = "\n".join(x for x in (fix_caption(ln, extras, trigger) for ln in core.caption_lines(old)) if x)
        if new and new != "\n".join(core.caption_lines(old)):
            rows.append([True, f, old, new])
    if not rows:
        return [], "Aucune legende a corriger : aucune ne decrit de trait permanent reconnu."
    return rows, (f"{len(rows)} legende(s) a corriger. Relisez la colonne 'Proposee' (modifiable), decochez "
                  "celles a laisser, puis appliquez. Les originales sont sauvegardees a cote du dataset.")


def backup_dir(folder):
    return os.path.normpath(folder) + "_legendes_originales"


def apply(folder, rows):
    """Ecrit les legendes cochees ; la premiere fois, sauvegarde l'originale de chaque fichier."""
    bdir = backup_dir(folder)
    done = 0
    for row in rows or []:
        if len(row) < 4:
            continue
        on, name, _old, new = row[0], str(row[1]), row[2], str(row[3] or "").strip()
        rel = os.path.normpath(name)
        if str(on).lower() not in ("true", "1") or not new or os.path.isabs(rel) or rel.startswith(".."):
            continue
        path = os.path.join(folder, rel)
        if not os.path.isfile(path):
            continue
        keep = os.path.join(bdir, rel)
        os.makedirs(os.path.dirname(keep), exist_ok=True)
        if not os.path.isfile(keep):
            shutil.copy2(path, keep)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
        done += 1
    return f"{done} legende(s) corrigee(s). Originales dans : {bdir}"


def restore(folder):
    bdir = backup_dir(folder)
    if not os.path.isdir(bdir):
        return "Aucune sauvegarde de legendes pour ce dossier."
    n = 0
    for root, _dirs, files in os.walk(bdir):
        for f in files:
            if f.lower().endswith(".txt"):
                rel = os.path.relpath(os.path.join(root, f), bdir)
                os.makedirs(os.path.dirname(os.path.join(folder, rel)), exist_ok=True)
                shutil.copy2(os.path.join(root, f), os.path.join(folder, rel))
                n += 1
    return f"{n} legende(s) d'origine restauree(s)."
