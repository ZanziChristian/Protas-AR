"""
Passo 4 della catena: la stima del progresso a ogni passo, sul test e sul training.

    python stima.py --split 1
    -> risultati/split1/stime_test.npz     per ogni video: progresso, passi trascorsi, azione di ogni passo
    -> risultati/split1/stime_train.npz    lo stesso sui video di training (per la fusione)

Per ogni video il flusso si costruisce con le azioni del SEGMENTATORE (segmenta.py), non
con quelle vere: è il sistema reale, che non conosce le annotazioni. Per il training si
usano le azioni del cross-fitting, sbagliate come al test.

A ogni passo t, dal flusso fino al [PROG] di t (al più 1024 token), il rollout atteso
(rollout.py) dà la stima del progresso. L'inizio dell'azione è quello che dice il
segmentatore: l'ultimo passo in cui la sua azione è cambiata. Da lì si contano i passi
trascorsi, L_passati = t − inizio + 1.

Quattro catene per passo, tetto di 48 passi immaginati (16 s). I numeri casuali partono
dal seme 0 e i video si leggono sempre nello stesso ordine: le stime sono riproducibili.
"""
import argparse

import numpy as np
import torch

from dati import RISULTATI, carica_videos, dividi_validazione, split
from flusso import costruisci
from modello import carica
from rollout import immagina, progresso_atteso

FINESTRA, CAMPIONI, TETTO, CATENE = 1024, 4, 48, 32


def inizio_azione(azioni: np.ndarray) -> np.ndarray:
    """Per ogni passo, il passo in cui è cominciata la sua azione (secondo `azioni`)."""
    inizi = np.zeros(len(azioni), dtype=np.int64)
    for t in range(1, len(azioni)):
        inizi[t] = t if azioni[t] != azioni[t - 1] else inizi[t - 1]
    return inizi


def stima_video(modello, video, azioni: np.ndarray, device: str) -> dict:
    """Le stime di un video: progresso [T] (NaN dove non si stima), passi trascorsi [T]."""
    flusso = costruisci(video, azioni)
    token = torch.from_numpy(flusso.token).to(device)
    trascorsi = np.arange(video.n_passi) - inizio_azione(azioni) + 1
    progresso = np.full(video.n_passi, np.nan)
    # Si stima nei passi che stanno dentro un'azione secondo le annotazioni, dove il
    # progresso è definito. Nei passi di background la stima resta NaN, cioè 0 (vedi il
    # README, "Da sapere").
    da_stimare = [(t, int(flusso.pos_prog[t])) for t in range(video.n_passi) if flusso.in_azione[t]]
    # Prefissi lunghi uguali (1024 token, cioè dopo i primi ~54 passi) si immaginano 8 alla
    # volta, 32 catene in tutto; quelli dell'inizio, tutti di lunghezza diversa, uno alla volta.
    lunghi = [x for x in da_stimare if x[1] + 1 >= FINESTRA]
    gruppi = [lunghi[i:i + CATENE // CAMPIONI] for i in range(0, len(lunghi), CATENE // CAMPIONI)]
    gruppi += [[x] for x in da_stimare if x[1] + 1 < FINESTRA]
    for gruppo in gruppi:
        prefissi = torch.stack([token[max(0, pos + 1 - FINESTRA): pos + 1] for _, pos in gruppo])
        massa = immagina(modello, prefissi, TETTO, campioni=CAMPIONI)
        for (t, _), m in zip(gruppo, massa):
            progresso[t] = progresso_atteso(m, int(trascorsi[t]))
    return {"progresso": progresso, "trascorsi": trascorsi, "azione": azioni}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cartella = RISULTATI / f"split{a.split}"
    train, test = split(a.split)
    # i video di training nell'ordine dell'addestramento: prima i 18, poi i 3 di validazione
    for parte, nomi in (("test", test), ("train", sum(dividi_validazione(train), []))):
        torch.manual_seed(0)                    # ogni parte riparte dallo stesso seme
        modello = carica(cartella / "ar_sf" / "modello_finale.pt", a.device)
        azioni = np.load(cartella / "segmentazione" / f"azioni_{parte}.npz")
        out = {}
        for v in carica_videos(nomi, a.split):
            s = stima_video(modello, v, azioni[v.nome], a.device)
            out |= {f"{v.nome}__{k}": x for k, x in s.items()}
            print(f"  {parte} {v.nome:18s} {v.n_passi:4d} passi", flush=True)
        np.savez(cartella / f"stime_{parte}.npz", **out)
    print(f"-> {cartella}/stime_{{test,train}}.npz")


if __name__ == "__main__":
    main()
