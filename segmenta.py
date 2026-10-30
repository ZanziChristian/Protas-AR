"""
Passo 2 della catena: quale azione è in corso, frame per frame, in modo online.

    python segmenta.py --split 1
    -> risultati/split1/segmentazione/logit_{test,train}.npz    logit [K, T] di ogni video
    -> risultati/split1/segmentazione/azioni_{test,train}.npz   l'azione di ogni passo

Il nostro modello stima il progresso dell'AZIONE IN CORSO, quindi deve sapere quale
azione è in corso: gliela dice un segmentatore online, e la si scrive nel suo flusso di
token. Il segmentatore qui è la MEDIA di due segmentatori online diversi, la CAS di
ProTAS (cas.py) e OnlineTAS (onlinetas.py): due architetture diverse sbagliano in punti
diversi, e la media delle loro log-probabilità sbaglia meno di ciascuna. La scelta fra
questa media e le alternative (ciascuno dei due da solo, una rete che li combina) è
stata fatta sul cross-fitting del training, mai sul test.

Training e test
---------------
  * Per il TEST ciascun segmentatore si addestra su tutti i video di training.
  * Per il TRAINING servono uscite "da test" (vedi dati.py, cross-fitting): ogni video
    di training riceve i logit di un segmentatore addestrato sugli altri tre quarti.
    Sui video che ha visto un segmentatore indovina il 97% dei passi, sul test il
    67-77%: un modello che imparasse dalle sue uscite di training imparerebbe a
    fidarsene troppo. Le uscite del cross-fitting sono sbagliate come al test.

La post-elaborazione di OnlineTAS
---------------------------------
I modelli causali spezzano spesso un'azione con pochi frame incerti di un'altra classe.
La post-elaborazione di OnlineTAS (Alg. 2 del paper) accetta un'etichetta solo se la
rete ne è sicura (probabilità massima >= 0.6); altrimenti il frame tiene l'etichetta
del frame prima, ma per al più l_min frame di fila (1/16 del video di training più
lungo). Usa solo il presente e il passato: resta online. I due valori sono quelli del
loro paper per GTEA, non scelti da noi.

L'azione di un passo è quella che il segmentatore (dopo la post-elaborazione) dice
all'ultimo frame del passo, cioè quando il passo finisce.
"""
import argparse

import numpy as np
import torch

import cas
import onlinetas
from dati import RISULTATI, background, classi, etichette_frame, feature_frame, fold_di_training, griglia, split
from metriche import segmentazione

SOGLIA_PP, FRAZIONE_PP = 0.6, 1 / 16


def softmax(L: np.ndarray) -> np.ndarray:
    """Softmax sulle classi di logit [K, T]."""
    e = np.exp(L - L.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)


def post_elaborazione(prob: np.ndarray, l_min: float) -> np.ndarray:
    """Probabilità [K, T] -> etichette [T] con la post-elaborazione di OnlineTAS (vedi in cima)."""
    sicura, y = prob.max(0), prob.argmax(0)
    out, di_fila = y.copy(), 0
    for t in range(1, len(y)):
        if sicura[t] < SOGLIA_PP and di_fila < l_min:
            out[t] = out[t - 1]
            di_fila += 1
        else:
            di_fila = 0
    return out


def l_min(train: list[str]) -> float:
    return FRAZIONE_PP * max(len(etichette_frame(v)) for v in train)


def azioni_per_passo(etichette: np.ndarray) -> np.ndarray:
    """Etichette per frame [T] -> l'etichetta all'ultimo frame di ogni passo."""
    _, fini = griglia(len(etichette))
    return etichette[np.minimum(fini, len(etichette)) - 1].astype(np.int64)


def media(*logit: dict) -> dict:
    """La media delle log-probabilità di più segmentatori, video per video."""
    out = {}
    for v in logit[0]:
        T = min(L[v].shape[1] for L in logit)
        lp = [torch.log_softmax(torch.from_numpy(L[v][:, :T]), 0) for L in logit]
        out[v] = torch.stack(lp).mean(0).numpy()
    return out


def addestra_e_predici(modulo, dati, train, test, K, device) -> tuple[dict, dict]:
    """(logit del test da un modello addestrato su tutto il training,
        logit del training dal cross-fitting)."""
    pieno = modulo.addestra(dati, train, K, device)
    logit_test = modulo.logit(pieno, dati, test, device)
    logit_train = {}
    for i, gruppo in enumerate(fold_di_training(train)):
        print(f"  cross-fitting {i + 1}/4: predetti {len(gruppo)} video", flush=True)
        m = modulo.addestra(dati, [v for v in train if v not in gruppo], K, device)
        logit_train |= modulo.logit(m, dati, gruppo, device)
    return logit_test, logit_train


def stampa(nome: str, logit: dict, minimo: float, bg: int) -> None:
    coppie = [(L.argmax(0), etichette_frame(v)[:L.shape[1]]) for v, L in logit.items()]
    coppie_pp = [(post_elaborazione(softmax(L), minimo), y) for (_, y), L in zip(coppie, logit.values())]
    for nome_pp, c in (("", coppie), (" + pp", coppie_pp)):
        print(f"{nome + nome_pp:34s} " + "  ".join(f"{k} {x:.1f}" for k, x in segmentazione(c, bg).items()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    train, test = split(a.split)
    dati = {v: (feature_frame(v), etichette_frame(v)) for v in train + test}
    K, bg, minimo = len(classi()), background(), l_min(train)

    uscite = {}
    for nome, modulo in (("OnlineTAS", onlinetas), ("CAS", cas)):
        print(f"{nome}: addestramento sul training e cross-fitting", flush=True)
        uscite[nome] = addestra_e_predici(modulo, dati, train, test, K, a.device)
    logit_test = media(uscite["OnlineTAS"][0], uscite["CAS"][0])
    logit_train = media(uscite["OnlineTAS"][1], uscite["CAS"][1])

    out = RISULTATI / f"split{a.split}" / "segmentazione"
    out.mkdir(parents=True, exist_ok=True)
    for parte, L in (("test", logit_test), ("train", logit_train)):
        np.savez(out / f"logit_{parte}.npz", **{v: x.astype(np.float32) for v, x in L.items()})
        np.savez(out / f"azioni_{parte}.npz", **{v: azioni_per_passo(post_elaborazione(softmax(x), minimo))
                                                 for v, x in L.items()})
    for nome, (te, tr) in uscite.items():
        stampa(f"{nome} sul test", te, minimo, bg)
        stampa(f"{nome} sul training (cross-fitting)", tr, minimo, bg)
    stampa("media sul test", logit_test, minimo, bg)
    stampa("media sul training (cross-fitting)", logit_train, minimo, bg)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
