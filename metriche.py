"""
Le metriche di ProTAS, per confrontarsi con le sue tabelle.

Segmentazione (le stesse di MS-TCN, seguendo `eval.py` di ProTAS riga per riga)
  Acc     frazione dei frame con l'etichetta giusta, contata su TUTTI i frame dello
          split insieme (non media per video). ProTAS la dà anche senza il background.
  Edit    distanza di Levenshtein normalizzata fra la sequenza dei segmenti predetti e
          quella vera (senza background): misura l'ordine delle azioni e punisce i
          segmenti spezzati. Media per video, da 0 a 100.
  F1@k    un segmento predetto è giusto se ha la classe giusta e si sovrappone (IoU)
          almeno al k% con un segmento vero non ancora usato; precisione e richiamo si
          contano su tutto lo split. k = 10, 25, 50.

Progresso (ProTAS, Fig. 6), sui segmenti d'azione VERI, frame per frame, nel canale
della classe VERA:
  MSE     errore quadratico medio fra stima e progresso vero (i+1)/L, media sui segmenti;
  E@C     errore all'ultimo frame del segmento, dove il vero vale 1: |p(fine) − 1|;
  CMD     quanto lontano dalla fine la stima tocca il suo massimo, in frazione della
          durata del segmento (0 = proprio all'ultimo frame).
"""
from itertools import groupby

import numpy as np

from dati import segmenti


def _segmenti_senza_bg(etichette, bg: int) -> list[tuple[int, int, int]]:
    """(classe, inizio, fine) dei segmenti, senza il background."""
    out, inizio = [], 0
    for k, g in groupby(etichette):
        n = len(list(g))
        if k != bg:
            out.append((int(k), inizio, inizio + n))
        inizio += n
    return out


def _levenshtein(p: list, y: list) -> float:
    D = np.zeros((len(p) + 1, len(y) + 1))
    D[:, 0], D[0, :] = np.arange(len(p) + 1), np.arange(len(y) + 1)
    for i in range(1, len(p) + 1):
        for j in range(1, len(y) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (y[j - 1] != p[i - 1]))
    return (1 - D[-1, -1] / max(len(p), len(y), 1)) * 100


def _conteggi_f1(pred, vero, soglia: float, bg: int) -> tuple[int, int, int]:
    """(veri positivi, falsi positivi, falsi negativi) di un video: ogni segmento predetto
    cerca il segmento vero della stessa classe con IoU più alta, e lo "consuma" se supera
    la soglia ed è ancora libero."""
    P, Y = _segmenti_senza_bg(pred, bg), _segmenti_senza_bg(vero, bg)
    usato = np.zeros(len(Y), dtype=bool)
    tp = fp = 0
    for k, s, e in P:
        migliore, j_migliore = 0.0, -1
        for j, (ky, sy, ey) in enumerate(Y):
            if ky != k:
                continue
            unione = max(e, ey) - min(s, sy)
            iou = max(0, min(e, ey) - max(s, sy)) / unione if unione else 0.0
            if iou > migliore:
                migliore, j_migliore = iou, j
        if migliore >= soglia and j_migliore >= 0 and not usato[j_migliore]:
            tp += 1
            usato[j_migliore] = True
        else:
            fp += 1
    return tp, fp, int((~usato).sum())


SOGLIE_F1 = (0.10, 0.25, 0.50)


def segmentazione(coppie: list[tuple[np.ndarray, np.ndarray]], bg: int) -> dict:
    """Acc, Acc senza background, Edit, F1@{10,25,50} da una lista di (predette, vere)
    etichette per frame, una coppia per video."""
    giusti = totali = giusti_nobg = totali_nobg = 0
    edit, conteggi = [], {s: [0, 0, 0] for s in SOGLIE_F1}
    for pred, vero in coppie:
        pred, vero = np.asarray(pred), np.asarray(vero)
        giusti += int((pred == vero).sum()); totali += len(vero)
        azione = vero != bg
        giusti_nobg += int((pred[azione] == vero[azione]).sum()); totali_nobg += int(azione.sum())
        edit.append(_levenshtein([k for k, _, _ in _segmenti_senza_bg(pred, bg)],
                                 [k for k, _, _ in _segmenti_senza_bg(vero, bg)]))
        for s in SOGLIE_F1:
            for i, x in enumerate(_conteggi_f1(pred, vero, s, bg)):
                conteggi[s][i] += x
    out = {"acc": 100 * giusti / max(totali, 1), "acc_wo_bg": 100 * giusti_nobg / max(totali_nobg, 1),
           "edit": float(np.mean(edit))}
    for s, (tp, fp, fn) in conteggi.items():
        precisione, richiamo = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"f1@{int(s * 100)}"] = 100 * 2 * precisione * richiamo / max(precisione + richiamo, 1e-12)
    return out


def progresso(P: dict, etichette: dict, video: list[str], bg: int) -> dict:
    """MSE, E@C, CMD (vedi in cima) e MAE. P[v] è la stima del progresso per classe [K, T]
    a 15 fps, etichette[v] le classi vere per frame. Accanto alla MSE quella con il
    bersaglio dell'Eq. 4 del paper, i/(L−1)."""
    r = {"mse": [], "mse_eq4": [], "e@c": [], "cmd": [], "mae": []}
    for v in video:
        for s, e, k in segmenti(etichette[v]):
            if k == bg:
                continue
            L = e - s
            q = P[v][k, s:e].astype(np.float64)
            vero = (np.arange(L) + 1) / L
            r["mse"].append(((q - vero) ** 2).mean()); r["mae"].append(np.abs(q - vero).mean())
            r["mse_eq4"].append(((q - np.arange(L) / max(L - 1, 1)) ** 2).mean())
            r["e@c"].append(abs(q[-1] - 1.0)); r["cmd"].append(abs(int(np.argmax(q)) - (L - 1)) / L)
    return {k: float(np.mean(x)) for k, x in r.items()}
