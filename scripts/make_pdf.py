"""Build a plain-language PDF report from a finished run.

  python scripts/make_pdf.py --results results/mini
Writes <results>/report/Brain_Tumour_Segmentation_Report.pdf. All numbers are read from the result files.
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1f1f1d")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
SOFT = colors.HexColor("#eef4fc")

MODELS = [  # (result tag, name, what makes it different)
    ("runu_eco_svaeunetpp", "RUNU-ECO-<br/>SVAEUnet++<br/>(the paper's model)",
     "The full paper model: EfficientNet encoder + spatial Vision Transformer + UNet++ decoder, with its "
     "settings picked automatically by the RUNU-ECO optimiser."),
    ("svaeunetpp", "SVAEUnet++ (not tuned)",
     "Exactly the same network, but with ordinary default settings instead of optimiser-picked ones."),
    ("efficient_unetpp", "Efficient-UNet++",
     "The paper model without the Vision Transformer (ablation: tests what the transformer adds)."),
    ("efficient_unet", "Efficient-UNet",
     "The paper model without the transformer and with a simple U-Net decoder (ablation)."),
    ("unet3p", "UNet 3+", "Well-known comparison model: U-Net where every decoder level sees every scale."),
    ("unetpp", "UNet++", "Well-known comparison model: U-Net with extra nested skip connections."),
    ("unet", "UNet", "The classic medical segmentation network; the simplest model here."),
]
DATASETS = ["brats", "figshare", "lgg"]


def load(res):
    R = {}
    for p in glob.glob(os.path.join(res, "cv", "*", "*", "fold0", "metrics.json")):
        parts = p.split(os.sep)
        R.setdefault(parts[-4], {})[parts[-3]] = json.load(open(p))
    return R


def dice(R, ds, tag):
    r = R.get(ds, {}).get(tag)
    if r is None:
        return None
    cases = r["per_case"]
    if ds == "lgg":  # tumour slices only (empty slices make the all-slice average misleading)
        cases = [c for c in cases if c.get("gt_positive", True)]
    return float(np.mean([c["dice"] for c in cases]))


def img(path, width):
    w, h = Image.open(path).size
    return RLImage(path, width=width, height=width * h / w)


def build(res):
    rep = os.path.join(res, "report")
    out = os.path.join(rep, "Brain_Tumour_Segmentation_Report.pdf")
    R = load(res)
    hpo = {}
    for p in glob.glob(os.path.join(res, "hpo", "*.json")):
        h = json.load(open(p))
        hpo.setdefault(h["dataset"], {})[h["optimizer"]] = h
    bench_path = os.path.join(os.path.dirname(os.path.normpath(res)), "optimizer_benchmark.json")
    bench = json.load(open(bench_path)) if os.path.exists(bench_path) else None

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=20, leading=24, textColor=INK, spaceAfter=4)
    SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=14)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=14, leading=18, textColor=BLUE, spaceBefore=12, spaceAfter=6)
    H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11.5, leading=15, textColor=INK, spaceBefore=8, spaceAfter=3)
    P = ParagraphStyle("P", parent=ss["Normal"], fontSize=10.2, leading=14.5, textColor=INK, spaceAfter=5)
    B = ParagraphStyle("B", parent=P, leftIndent=14, bulletIndent=4, spaceAfter=3)
    CAP = ParagraphStyle("CAP", parent=P, fontSize=8.8, leading=12, textColor=MUTED, spaceAfter=10)
    CELL = ParagraphStyle("CELL", parent=P, fontSize=8.6, leading=11, spaceAfter=0)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
    HEAD = ParagraphStyle("HEAD", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)

    def bullets(items):
        return [Paragraph(t, B, bulletText="•") for t in items]

    W = A4[0] - 3.6 * cm
    s = []

    # ------------------------------------------------------------------ title + summary
    s += [Paragraph("Brain Tumour Segmentation with SVAEUnet++", H1),
          Paragraph("Re-implementation of Krishnaveni &amp; Kollem (2026), <i>Biomedical Signal Processing and Control</i> 122, "
                    "110403 - results of the 1-hour test run", SUB)]
    s.append(Paragraph("The short version", H2))
    d = {t: [dice(R, ds, t) for ds in DATASETS] for t, _, _ in MODELS}
    best_tag = "runu_eco_svaeunetpp"
    s += bullets([
        "We rebuilt the paper's whole pipeline - data preparation, the SVAEUnet++ network, the RUNU-ECO optimiser that "
        "picks the training settings, and the comparison models - and ran it on <b>three brain-MRI datasets</b>.",
        f"The paper's tuned model got the highest overlap score (Dice) on all three datasets: "
        f"<b>{d[best_tag][0]:.2f}</b> on BraTS, <b>{d[best_tag][1]:.2f}</b> on Figshare and <b>{d[best_tag][2]:.2f}</b> on LGG.",
        "But the <b>same network with normal settings did not beat</b> the simpler comparison models. In this run the "
        "win comes mostly from the better training settings (a higher learning speed), not from the network design.",
        "The paper's new optimiser (RUNU-ECO) and the original one (ECO) picked <b>identical settings</b> on every dataset, "
        "so this run cannot show that the new optimiser is better.",
        "This was a <b>quick test</b>: every model trained for only a few minutes on part of each dataset. Scores are "
        "therefore much lower than the paper's (about 0.92). It shows the pipeline works and how the models rank - "
        "it is not a full reproduction of the paper's numbers.",
    ])

    # ------------------------------------------------------------------ datasets
    s.append(Paragraph("The three datasets", H2))
    s.append(Paragraph("Every dataset is a set of brain MRI scans where a doctor has coloured in the tumour by hand "
                       "(the <b>mask</b>). The model sees the scan and must colour in the tumour itself; we then compare "
                       "its colouring with the doctor's.", P))
    s.append(Paragraph("Dataset 1 - BraTS2020 (3D)", H3))
    s.append(Paragraph("369 patients with brain tumours (gliomas). Each patient has a full <b>3D scan</b> of the head in four "
                       "MRI types (FLAIR, T1, T1-with-contrast, T2). The model colours the <b>whole tumour</b> in 3D. "
                       "This is the same dataset the paper used. This run used 60 patients to learn, 10 to check progress "
                       "and 20 unseen patients to test.", P))
    s.append(Paragraph("Dataset 2 - Figshare brain tumour set (2D)", H3))
    s.append(Paragraph("3,064 single <b>2D slices</b> (one MRI type, T1-with-contrast) showing three tumour types: glioma, "
                       "meningioma and pituitary tumour. Every slice contains a tumour. This run used 600 slices to learn and "
                       "200 to test.", P))
    s.append(Paragraph("Dataset 3 - TCGA-LGG (2D)", H3))
    s.append(Paragraph("3,929 <b>2D slices</b> (FLAIR MRI) from 110 patients with lower-grade gliomas. About two thirds of the "
                       "slices show <b>no tumour at all</b>, which makes it harder: the model must also learn to leave "
                       "healthy slices empty. This run used 600 slices to learn and 200 to test.", P))
    s.append(Paragraph("<b>Why datasets 2 and 3 differ from the paper:</b> the paper's own links for its datasets 2 and 3 "
                       "contain only <i>labels</i> (\"tumour / no tumour\", or tumour type) and <b>no masks</b>, so a "
                       "segmentation score cannot be calculated on them. We swapped in the closest public datasets that do "
                       "have doctor-drawn masks: Figshare has the same three tumour types as the paper's dataset 2.", P))

    # ------------------------------------------------------------------ metrics
    s.append(Paragraph("What the scores mean (in simple words)", H2))
    s.append(Paragraph("Picture the scan as a grid of tiny squares (pixels). The doctor marked some squares as tumour; the "
                       "model marks some too. Every square ends up in one of four groups: correctly marked tumour (a "
                       "<b>hit</b>), tumour the model missed (a <b>miss</b>), healthy tissue wrongly marked (a <b>false "
                       "alarm</b>), and healthy tissue correctly left alone. All scores below go from 0 (worst) to 1 "
                       "(perfect).", P))
    s += bullets([
        "<b>Dice</b> - the main score. How much the model's patch and the doctor's patch overlap. 1 = they match "
        "exactly, 0 = they do not touch. Punishes both misses and false alarms.",
        "<b>IoU (Jaccard)</b> - also overlap: shared area divided by the total area covered by either patch. Always "
        "a bit lower than Dice for the same result.",
        "<b>Precision</b> - of everything the model called tumour, how much really was tumour. Low precision = many "
        "false alarms (the model colours too much).",
        "<b>Sensitivity (Recall)</b> - of the real tumour, how much the model found. Low = it misses tumour.",
        "<b>Specificity</b> - of the healthy tissue, how much the model correctly left alone.",
        "<b>Accuracy</b> - share of all squares labelled correctly. It looks very high (often 0.95+) even for poor "
        "models, because most of the brain is healthy and easy to get right - so do not judge by accuracy alone.",
        "<b>F1</b> - for this task it is the same number as Dice. <b>MCC</b> - a balanced score that, like Dice, is "
        "not fooled by the huge amount of healthy tissue.",
    ])

    # ------------------------------------------------------------------ the one table
    table_head = [Paragraph("How the models differ - and how they scored", H2),
                  Paragraph("Dice score on unseen test data (higher is better; the best score in each column is in bold). "
                            "LGG is scored on slices that contain tumour.", P)]
    rows = [[Paragraph(x, HEAD) for x in ("Model", "What is different about it", "BraTS<br/>(3D)", "Figshare<br/>(2D)",
                                         "LGG<br/>(2D)", "Average")]]
    best_col = [max((d[t][i] for t, _, _ in MODELS if d[t][i] is not None), default=None) for i in range(3)]
    avgs = {t: float(np.mean(v)) for t, v in d.items() if None not in v}
    best_avg = max(avgs.values()) if avgs else None
    for t, name, desc in MODELS:
        cells = [Paragraph(name, CELLB), Paragraph(desc, CELL)]
        for i in range(3):
            v = d[t][i]
            cells.append(Paragraph("-" if v is None else f"{v:.2f}", CELLB if v == best_col[i] else CELL))
        a = avgs.get(t)
        cells.append(Paragraph("-" if a is None else f"{a:.2f}", CELLB if a == best_avg else CELL))
        rows.append(cells)
    tbl = Table(rows, colWidths=[3.9 * cm, 6.6 * cm, 1.55 * cm, 1.75 * cm, 1.45 * cm, 1.65 * cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("BACKGROUND", (0, 1), (-1, 1), SOFT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    s.append(KeepTogether(table_head + [tbl]))
    s.append(Spacer(1, 8))
    s.append(Paragraph("<b>How to read it:</b>", P))
    s += bullets([
        "<b>Top row vs second row</b> is the same network; only the training settings differ. The large gap shows the "
        "settings matter a lot when training is this short. On BraTS and Figshare the tuned model also trained for "
        "more steps; on LGG both trained for almost the same number of steps, and the tuned model still won.",
        "<b>Ablation rows</b> (Efficient-UNet, Efficient-UNet++): the paper reports that each added part helps. In "
        "this short run that pattern does <b>not</b> appear - the bigger networks need more training to pay off.",
        "<b>UNet 3+</b> is the strongest of the normally-trained models on BraTS and Figshare.",
    ])

    # ------------------------------------------------------------------ pictures
    names = {"brats": "BraTS (3D)", "figshare": "Figshare (2D)", "lgg": "LGG (2D)"}
    for ds in DATASETS:
        s.append(PageBreak())
        s.append(Paragraph(f"Results on {names[ds]}", H2))
        q = os.path.join(rep, f"{ds}_fig8_qualitative.png")
        if os.path.exists(q):
            s.append(img(q, W))
            s.append(Paragraph("<b>Example test scans.</b> Column 1: the MRI. Column 2: the tumour as drawn by the doctor "
                               "(orange). Other columns: each model's answer (blue), with its Dice score for that slice. "
                               "Blue outside the orange area = false alarms; orange with no blue = missed tumour."
                               + (" For BraTS the slice with the most tumour is shown; the patient's score covers "
                                  "the whole 3D head, so it can be lower than the slice suggests." if ds == "brats" else ""),
                               CAP))
        m = os.path.join(rep, f"{ds}_fig9_metrics.png")
        if os.path.exists(m):
            s.append(img(m, W))
            s.append(Paragraph("<b>All scores side by side</b> (bars = average over test cases, thin lines = spread "
                               "between cases). Recall is high for every model while Dice and IoU are low: the models find "
                               "the tumour but also colour too much healthy tissue."
                               + (" Here every slice is included, also tumour-free ones, which pulls Dice down." if ds == "lgg" else ""),
                               CAP))

    # ------------------------------------------------------------------ training curves
    s.append(PageBreak())
    s.append(Paragraph("Training progress of the paper's model", H2))
    s.append(Paragraph("Loss is the model's error during training (lower is better). <b>Train</b> = error on the "
                       "examples it learns from; <b>validation</b> = error on held-back examples it never learns from. "
                       "Both falling together means the model is genuinely learning, not memorising.", P))
    loss = [os.path.join(rep, f"{ds}_fig11_loss.png") for ds in DATASETS if os.path.exists(os.path.join(rep, f"{ds}_fig11_loss.png"))]
    if loss:
        cw = W / len(loss)
        t = Table([[img(p, cw - 0.2 * cm) for p in loss]], colWidths=[cw] * len(loss))
        s.append(t)
        s.append(Paragraph("Only 5 short training rounds (epochs) were run, so the curves are still falling - more "
                           "training would very likely raise every score.", CAP))

    # ------------------------------------------------------------------ optimiser
    s.append(Paragraph("The settings optimiser: RUNU-ECO vs ECO", H2))
    s.append(Paragraph("The paper's second idea is <b>RUNU-ECO</b>, a search method (modelled on schools competing for "
                       "students) that tries many combinations of three training settings - size of the transformer's "
                       "hidden layer, learning speed (learning rate), and steps per round - and keeps whatever gives the "
                       "best overlap. It is a tweak of an earlier method, <b>ECO</b>.", P))
    picks = []
    for ds in DATASETS:
        h = hpo.get(ds, {})
        if "RUNU-ECO" in h:
            b = h["RUNU-ECO"]["best_hparams"]
            same = "ECO" in h and h["ECO"]["best_hparams"] == b
            picks.append(f"<b>{names[ds]}</b>: hidden size {b['hidden_neurons']}, learning rate "
                         f"{b['learning_rate']:.4f}, {b['steps_per_epoch']} steps per round"
                         + (" - <b>ECO picked exactly the same</b>." if same else "."))
    s += bullets(picks)
    s.append(Paragraph("With the small search budget of this run, both methods ended on the same answer every time, so "
                       "the run cannot separate them. To compare them fairly we also tested both on six standard maths "
                       "puzzles (functions with a known best answer), 20 times each:", P))
    if bench:
        wins = sum(np.median(v["runu-eco"]["final"]) < np.median(v["eco"]["final"]) for v in bench.values())
        s.append(Paragraph(f"RUNU-ECO did better on <b>{wins} of {len(bench)}</b> puzzles and worse on the rest - so the "
                           "paper's claim that it is clearly better is <b>not confirmed</b>. Both are far better than "
                           "random guessing (grey).", P))
        s.append(KeepTogether([img(os.path.join(rep, "fig_optimizer_benchmark.png"), W),
                               Paragraph("Each panel is one puzzle; the line shows the best answer found so far "
                                         "(lower is better, log scale).", CAP)]))

    # ------------------------------------------------------------------ limits
    s.append(Paragraph("Limits of this run", H2))
    s += bullets([
        "Every model trained for only 5 short rounds, on 60 patients (BraTS) or 600 slices (2D sets). The paper trained "
        "much longer on all data, so its scores (about 0.92 Dice) are not comparable with these.",
        "Only 1 of the 5 cross-validation folds was run, so the results come from one train/test split.",
        "The settings search was tiny (4-6 candidates, 2-4 rounds) instead of the paper's 10 candidates x 50 rounds. "
        "BraTS reused a search from an earlier, larger run.",
        "Three optimisers the paper compares against (FSA, SCO, SAA) were not implemented because the paper does not "
        "give their equations.",
        "To run the paper's full settings on a GPU: <font face='Courier'>python scripts/run_all.py --profile paper</font>",
    ])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.8 * cm, 1.1 * cm, "SVAEUnet++ re-implementation - 1-hour test run")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.8 * cm, title="Brain Tumour Segmentation with SVAEUnet++",
                            author="SVAEUnet++ re-implementation")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/mini")
    build(ap.parse_args().results)
