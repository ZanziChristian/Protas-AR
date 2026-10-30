"""
Due domande sul modello generativo, sui video di test (i numeri della relazione).

    python diagnosi.py --split 1

1. Prevede il token successivo meglio che copiando?
   Un passo dura 1/3 di secondo e la scena cambia poco: molti codici restano IDENTICI
   da un passo all'altro. La baseline è "il codice nello stesso posto del passo prima".
   Si riporta l'accuratezza del modello anche sui soli codici che CAMBIANO: è lì che
   deve davvero immaginare il futuro.

2. P([EoA]) sale dove l'azione finisce?
   Letta sui token veri a ogni confine fra due passi, con le azioni vere nel flusso.
   L'AUC dice quanto spesso un confine vero ha la P più alta di uno falso (0.5 = non li
   distingue, 1 = sempre): contro tutti gli altri confini, e contro i soli confini
   dentro un'azione (il caso difficile: sul background [EoA] non arriva mai).
"""
import argparse

import numpy as np
import torch

from addestra import finestre_di_lettura
from dati import RISULTATI, carica_videos, split
from flusso import BLK, EOA, V, costruisci
from modello import carica
from tokenizzatore import PEZZI


def auc(punteggi: np.ndarray, positivi: np.ndarray) -> float:
    """L'area sotto la curva ROC dai ranghi (statistica di Mann-Whitney), pareggi a metà."""
    punteggi, positivi = np.asarray(punteggi, float), np.asarray(positivi, bool)
    n_pos, n_neg = positivi.sum(), (~positivi).sum()
    ranghi = np.empty(len(punteggi))
    ranghi[np.argsort(punteggi, kind="mergesort")] = np.arange(1, len(punteggi) + 1)
    for x in np.unique(punteggi):
        uguali = punteggi == x
        if uguali.sum() > 1:
            ranghi[uguali] = ranghi[uguali].mean()
    return float((ranghi[positivi].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def letture(modello, token: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """A ogni posizione i: il codice più probabile in i + 1, e P([EoA]) in i + 1."""
    t = torch.from_numpy(token).to(device)
    scelto, p_eoa, lette = np.zeros(len(t), np.int64), np.zeros(len(t), np.float32), 0
    for inizio, fine in finestre_di_lettura(len(t)):
        da = max(inizio, lette)
        if da >= fine:
            continue
        logit = modello(t[None, inizio:fine])[0][0, da - inizio:].float()
        scelto[da:fine] = logit[:, :V].argmax(-1).cpu().numpy()
        p_eoa[da:fine] = np.exp((logit[:, EOA] - torch.logsumexp(logit, -1)).cpu().numpy())
        lette = fine
    return scelto, p_eoa


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--modello", default=None, help="default: il modello finale dello split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    modello = carica(a.modello or RISULTATI / f"split{a.split}" / "ar_sf" / "modello_finale.pt", a.device)
    copia = giusti = cambiano = giusti_cambiano = 0
    p_eoa, vero, dentro = [], [], []
    for v in carica_videos(split(a.split)[1], a.split):
        f = costruisci(v)
        scelto, p = letture(modello, f.token, a.device)
        # 1. i codici dal secondo passo: posizione q, predetto da q − 1, contro lo stesso posto del passo prima
        blk = np.flatnonzero(f.token == BLK)
        pos = np.concatenate([np.arange(b - PEZZI, b) for b in blk[1:]])
        prima = np.concatenate([f.token[b - PEZZI:b] for b in blk[:-1]])
        resta = f.token[pos] == prima
        ok = scelto[pos - 1] == f.token[pos]
        copia += int(resta.sum()); giusti += int(ok.sum())
        cambiano += int((~resta).sum()); giusti_cambiano += int(ok[~resta].sum())
        # 2. P([EoA]) a ogni confine: letta alla posizione del [PROG], vale per il token dopo
        p_eoa.append(p[f.pos_confine]); vero.append(f.eoa_al_confine)
        dentro.append(f.in_azione[:len(f.pos_confine)])
    n = copia + cambiano
    p_eoa, vero, dentro = np.concatenate(p_eoa), np.concatenate(vero), np.concatenate(dentro)
    print(f"token successivo sui codici del test ({n} codici)")
    print(f"  copia del passo prima       acc {copia / n:.3f}")
    print(f"  modello                     acc {giusti / n:.3f}")
    print(f"    sui codici che cambiano   acc {giusti_cambiano / max(cambiano, 1):.3f}   ({cambiano / n:.1%} dei codici)")
    print(f"P([EoA]) ai confini veri ({vero.sum()} su {len(vero)})")
    print(f"  AUC contro tutti gli altri confini   {auc(p_eoa, vero):.3f}")
    print(f"  AUC contro i confini dentro azioni   {auc(p_eoa[dentro], vero[dentro]):.3f}")


if __name__ == "__main__":
    main()
