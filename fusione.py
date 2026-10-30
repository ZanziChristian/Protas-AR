"""
Passo 5 della catena: il progresso dentro ProTAS, e le metriche finali.

    python fusione.py --split 1 [--calibra]
    -> risultati/split1/risultati[_cal].json     segmentazione e progresso, test e training
    -> risultati/split1/predizioni[_cal].npz     etichette e progresso per frame dei video di test (per le figure)

ProTAS (Shen & Elhamifar, CVPR 2024) ha quattro pezzi: il segmentatore causale, un
modulo che stima il progresso di OGNI classe (APP), la fusione progresso-probabilità e il
grafo del compito. Qui il segmentatore è la media di segmenta.py e il progresso è il
NOSTRO modello autoregressivo; fusione e grafo sono quelli di ProTAS.

Dal passo ai frame, in modo online
----------------------------------
La stima del passo t si conosce solo alla fine del passo (fini[t]). Nei frame che vengono
dopo, fino alla fine del passo successivo, la stima non resta ferma: cresce come cresce
il progresso vero se la durata prevista è giusta. Se alla fine del passo sono passati e
frame dall'inizio dell'azione e la stima è p = e / (e + r), dopo altri d frame il
progresso è (e + d) / (e + r) = p · (1 + d / e), al più 1. Usa solo il passato.

Il progresso per classe
-----------------------
Fusione e grafo vogliono un progresso per ogni classe, [K, T]. Il nostro è quello
dell'azione in corso secondo il segmentatore: va nel canale di quell'azione, gli altri
valgono 0. Se il segmentatore sbaglia la classe, il canale della classe vera resta a 0 (e
le metriche di ProTAS, lette nel canale della classe vera, contano l'errore pieno).

La fusione (residua, nostra variante)
-------------------------------------
logit_nuovi = logit + conv1x1([logit, progresso]), con la convoluzione che parte da
ZERO: all'inizio la fusione È il segmentatore, e il progresso può solo aggiungere una
correzione. ProTAS allena la convoluzione insieme al segmentatore; qui la si allena da
sola, dopo, su 21 video, e partendo da pesi casuali deve reimparare perfino a copiare i
logit (non ci riesce). Si allena sulle uscite del cross-fitting del training.

Il grafo del compito (come ProTAS)
----------------------------------
Dalle trascrizioni del training (solo training: lo script di ProTAS legge anche il
test) le matrici dei predecessori e dei successori di ogni azione. Il completamento di
un'azione è il massimo del suo progresso fino a t; un'azione i cui predecessori non sono
completi, o i cui successori sono già cominciati, viene penalizzata: nella loss (peso
0.1, grafo apprendibile con regolarizzazione 0.01) e all'inferenza (logit − 0.01 · penalità).
Come ProTAS: 50 epoche, Adam lr 5e-4, un video per passo.

La calibrazione (--calibra, il confronto della relazione)
---------------------------------------------------------
Una regressione isotonica imparata sul training (cross-fitting) corregge la scala della
stima, uguale per il test. La variante finale è quella SENZA, scelta sul training: la
somma MSE + E@C + CMD del training è più bassa.
"""
import argparse
import json
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cas import perdita_stadio
from dati import (FRAME_PER_PASSO, RISULTATI, SEME_PROTAS, background, classi, etichette_frame, griglia, segmenti,
                  split)
from metriche import progresso as metriche_progresso
from metriche import segmentazione
from segmenta import l_min, post_elaborazione, softmax


# --- dal passo ai frame ---------------------------------------------------------------

def ai_frame(per_passo: np.ndarray, fini: np.ndarray, n: int) -> np.ndarray:
    """[T] un valore per passo -> [n] per frame: ogni frame ha l'ultimo valore già noto."""
    out = np.zeros(n, dtype=np.float64)
    for t in range(len(per_passo)):
        a, b = int(fini[t]), int(fini[t + 1]) if t + 1 < len(per_passo) else n
        out[min(a, n):min(b, n)] = per_passo[t]
    return out


def progresso_ai_frame(p: np.ndarray, trascorsi: np.ndarray, fini: np.ndarray, n: int) -> np.ndarray:
    """[T] stima per passo -> [n] per frame, crescente fra un passo e l'altro (vedi in cima).
    `trascorsi` sono i frame dall'inizio dell'azione alla fine di ogni passo."""
    out = np.zeros(n, dtype=np.float64)
    for t in range(len(p)):
        a, b = int(fini[t]), int(fini[t + 1]) if t + 1 < len(p) else n
        if a >= n:
            break
        d = np.arange(min(b, n) - a)
        out[a:min(b, n)] = np.minimum(1.0, p[t] * (1 + d / max(trascorsi[t], 1)))
    return out


def stime_ai_frame(file, n_frame: dict) -> dict:
    """Le stime di stima.py -> per ogni video (progresso per frame, azione per frame)."""
    z = np.load(file)
    out = {}
    for v in dict.fromkeys(k.split("__")[0] for k in z.files):
        p = np.clip(np.nan_to_num(z[f"{v}__progresso"].astype(np.float64)), 0, 1)
        _, fini = griglia(n_frame[v])
        trascorsi = z[f"{v}__trascorsi"].astype(np.float64) * FRAME_PER_PASSO
        out[v] = (progresso_ai_frame(p, trascorsi, fini, n_frame[v]),
                  ai_frame(z[f"{v}__azione"].astype(np.float64), fini, n_frame[v]).astype(np.int64))
    return out


def per_classe(S: dict, K: int) -> dict:
    """(progresso, azione) per frame -> [K, T]: il progresso nel canale dell'azione, 0 altrove."""
    out = {}
    for v, (pf, af) in S.items():
        P = np.zeros((K, len(pf)), np.float32)
        P[af, np.arange(len(pf))] = pf
        out[v] = P
    return out


# --- la calibrazione ---------------------------------------------------------------------

def isotonica(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """La funzione NON decrescente g che rende minima la somma di (g(x) − y)² (pool
    adjacent violators): si scorrono i punti in ordine di x e, ogni volta che un gruppo
    ha media più bassa del precedente, i due si fondono nella media pesata. Restituisce i
    nodi (x, g(x)) per np.interp."""
    xu, inv = np.unique(x, return_inverse=True)
    pesi0 = np.bincount(inv).astype(np.float64)
    medie0 = np.bincount(inv, weights=y) / pesi0
    medie, pesi, inizio, fine = [], [], [], []
    for xi, mi, wi in zip(xu, medie0, pesi0):
        medie.append(mi); pesi.append(wi); inizio.append(xi); fine.append(xi)
        while len(medie) > 1 and medie[-2] > medie[-1]:
            w = pesi[-2] + pesi[-1]
            m = (medie[-2] * pesi[-2] + medie[-1] * pesi[-1]) / w
            medie[-2:], pesi[-2:], inizio[-2:], fine[-2:] = [m], [w], [inizio[-2]], [fine[-1]]
    return np.ravel(np.column_stack([inizio, fine])), np.repeat(medie, 2)


def calibra(S_train: dict, S_test: dict, etichette: dict, bg: int) -> tuple[dict, dict]:
    """La calibrazione si impara sui frame del training in cui la stima sta nel canale
    giusto, e si applica uguale a training e test. Più 0.001 · stima: la funzione
    isotonica ha tratti piatti, e il termine piccolo tiene le stime nell'ordine di prima,
    così non si sposta il frame in cui la stima tocca il massimo (il CMD)."""
    xs, ys = [], []
    for v, (pf, af) in S_train.items():
        for s, e, k in segmenti(etichette[v]):
            if k != bg:
                giusto = af[s:e] == k
                xs.append(pf[s:e][giusto]); ys.append(((np.arange(e - s) + 1) / (e - s))[giusto])
    nodi = isotonica(np.concatenate(xs), np.concatenate(ys))
    return [{v: ((1 - 1e-3) * np.interp(pf, *nodi) + 1e-3 * pf, af) for v, (pf, af) in S.items()}
            for S in (S_train, S_test)]


# --- fusione e grafo --------------------------------------------------------------------

def grafo(train: list[str], K: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    """(predecessori, successori) [K, K] dalle trascrizioni del training: quante volte
    l'azione i viene subito prima (dopo) della j, diviso quante volte compare la j."""
    pre, suc, conta = np.zeros((K, K)), np.zeros((K, K)), np.zeros(K)
    for v in train:
        trascrizione = [k for k, _ in groupby(c for c in etichette_frame(v) if c != bg)]
        for k in trascrizione:
            conta[k] += 1
        for a, b in zip(trascrizione[:-1], trascrizione[1:]):
            pre[a, b] += 1; suc[b, a] += 1
    return pre / np.maximum(conta[None], 1e-5), suc / np.maximum(conta[None], 1e-5)


class FusioneGrafo(nn.Module):
    """La fusione residua progresso-probabilità e il grafo apprendibile (vedi in cima)."""

    def __init__(self, K: int, pre: np.ndarray, suc: np.ndarray, reg: float = 0.01, eta: float = 0.01):
        super().__init__()
        self.conv = nn.Conv1d(2 * K, K, 1)
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)
        self.pre = nn.Parameter(torch.from_numpy(pre).float())
        self.suc = nn.Parameter(torch.from_numpy(suc).float())
        self.register_buffer("pre0", torch.from_numpy(pre).float())
        self.reg, self.eta = reg, eta

    def forward(self, logit, prog):                       # [1, K, T] ciascuno
        return logit + self.conv(torch.cat([logit, prog], dim=1))

    def penalita(self, prog):
        completo, _ = torch.cummax(torch.clamp(prog, 0, 1), dim=-1)
        return torch.einsum("bkt,kK->bKt", 1 - completo, self.pre) + torch.einsum("bkt,kK->bKt", completo, self.suc)

    def perdita_grafo(self, logit, prog):
        return (self.penalita(prog) * F.softmax(logit, dim=1)).mean() + self.reg * ((self.pre - self.pre0) ** 2).mean()

    def inferenza(self, logit, prog):
        return logit - self.eta * self.penalita(prog)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--calibra", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.manual_seed(SEME_PROTAS)
    cartella = RISULTATI / f"split{a.split}"
    train, test = split(a.split)
    K, bg = len(classi()), background()
    etichette = {v: etichette_frame(v) for v in train + test}
    logit = dict(np.load(cartella / "segmentazione" / "logit_train.npz")) | dict(np.load(cartella / "segmentazione" / "logit_test.npz"))
    n_frame = {v: L.shape[1] for v, L in logit.items()}
    S_train = stime_ai_frame(cartella / "stime_train.npz", n_frame)
    S_test = stime_ai_frame(cartella / "stime_test.npz", n_frame)
    if a.calibra:
        S_train, S_test = calibra(S_train, S_test, etichette, bg)
    P = per_classe(S_train, K) | per_classe(S_test, K)

    # --- l'addestramento di fusione e grafo, sulle uscite del cross-fitting
    fg = FusioneGrafo(K, *grafo(train, K, bg)).to(a.device)
    opt = torch.optim.Adam(fg.parameters(), lr=5e-4)
    rng = np.random.RandomState(SEME_PROTAS)
    T = lambda d, v: torch.from_numpy(d[v])[None].to(a.device)
    for _ in range(50):
        for v in rng.permutation(train):
            o = fg(T(logit, v), T(P, v))
            loss = perdita_stadio(o, torch.from_numpy(etichette[v]).to(a.device)) + 0.1 * fg.perdita_grafo(o, T(P, v))
            opt.zero_grad(); loss.backward(); opt.step()
    fg.eval()

    # --- le metriche, sul test e (per le scelte) sul training, ognuna anche con la post-elaborazione
    minimo = l_min(train)
    risultati, etichette_test = {}, {}
    with torch.no_grad():
        for nome, fn in (("CAS", lambda v: T(logit, v)),
                         ("CAS + AR", lambda v: fg(T(logit, v), T(P, v))),
                         ("CAS + AR + grafo", lambda v: fg.inferenza(fg(T(logit, v), T(P, v)), T(P, v)))):
            for prefisso, video in (("", test), ("train: ", train)):
                L = {v: fn(v)[0].cpu().numpy() for v in video}
                risultati[prefisso + nome] = segmentazione([(L[v].argmax(0), etichette[v]) for v in video], bg)
                pp = {v: post_elaborazione(softmax(L[v]), minimo) for v in video}
                risultati[prefisso + nome + " + pp"] = segmentazione([(pp[v], etichette[v]) for v in video], bg)
                if prefisso == "":
                    etichette_test[nome] = pp
    risultati["progresso"] = metriche_progresso(P, etichette, test, bg)
    risultati["progresso_train"] = metriche_progresso(P, etichette, train, bg)

    nome = "risultati" + ("_cal" if a.calibra else "")
    (cartella / f"{nome}.json").write_text(json.dumps(risultati, indent=1))
    predizioni = {}
    for v in test:
        predizioni[f"{v}__segmentatore"] = etichette_test["CAS"][v]
        predizioni[f"{v}__completo"] = etichette_test["CAS + AR + grafo"][v]
        predizioni[f"{v}__stima"], predizioni[f"{v}__canale"] = S_test[v]
    np.savez(cartella / f"{nome.replace('risultati', 'predizioni')}.npz", **predizioni)

    for nome_riga in ("CAS + pp", "CAS + AR + grafo + pp"):
        print(f"{nome_riga:24s} " + "  ".join(f"{k} {x:.1f}" for k, x in risultati[nome_riga].items()))
    for parte in ("progresso", "progresso_train"):
        m = risultati[parte]
        print(f"{parte:24s} MSE {m['mse']:.3f}  E@C {m['e@c']:.3f}  CMD {m['cmd']:.3f}   (somma {m['mse'] + m['e@c'] + m['cmd']:.3f})")


if __name__ == "__main__":
    main()
