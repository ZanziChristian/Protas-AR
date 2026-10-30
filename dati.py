"""
I dati di GTEA: dove stanno, come si leggono, e come diventano "passi".

GTEA (Fathi et al., 2011) ha 28 video di cucina in prima persona, 11 classi (10 azioni
più il background), etichette a 15 frame al secondo e feature I3D da 2048 numeri per
frame (quelle di MS-TCN). Servono, in `data/gtea/` dentro questa cartella:

    features/<video>.npy        feature I3D [2048, T], una colonna per frame
    groundTruth/<video>.txt     un'etichetta per frame (il nome della classe)
    mapping.txt                 "<id> <nome>" per ogni classe
    splits/{train,test}.splitN.bundle   i 4 split ufficiali (un soggetto di test per split)

Il passo
--------
Il modello autoregressivo non lavora sui frame ma su PASSI da 5 frame (1/3 di secondo):
il passo t copre i frame [5t, 5t + 5), l'ultimo quelli che restano. Di ogni passo si
tengono:
  * la feature: la media delle feature I3D dei suoi frame;
  * l'etichetta: la classe che occupa più frame del passo (se un passo sta a cavallo di
    due azioni, vince quella che ne occupa di più);
  * il progresso dell'azione in corso, misurato sul tempo vero (vedi `progresso`).

Il cross-fitting
----------------
Più avanti molti pezzi (segmentatori, fusione, calibrazione) hanno bisogno di uscite
"da test" sui video di training: uscite di un modello che quel video NON l'ha visto.
Si dividono allora i video di training in 4 gruppi (`fold_di_training`) e ogni gruppo
viene predetto da un modello addestrato sugli altri tre. I gruppi sono sempre gli
stessi, così tutti i pezzi del sistema usano la stessa divisione.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
DATI = QUI / "data" / "gtea"                                    # i dati originali di GTEA
RISULTATI = Path(os.environ.get("RISULTATI", QUI / "risultati"))      # tutto quello che la catena produce

FRAME_PER_PASSO = 5
SEME_PROTAS = 1538574472                     # il seme di main.py di ProTAS, usato ovunque lo usavano loro


# --- classi, split, etichette --------------------------------------------------------

def classi() -> dict[str, int]:
    """nome della classe -> id, da `mapping.txt`."""
    righe = (DATI / "mapping.txt").read_text().split("\n")
    return {nome: int(i) for i, nome in (r.split() for r in righe if r.strip())}


def background() -> int:
    return classi()["background"]


def split(numero: int) -> tuple[list[str], list[str]]:
    """(video di training, video di test) dello split, nell'ordine dei file `.bundle`.
    L'ordine conta: da lui dipendono le estrazioni casuali."""
    leggi = lambda nome: [r.removesuffix(".txt") for r in (DATI / "splits" / nome).read_text().split() if r.strip()]
    train, test = leggi(f"train.split{numero}.bundle"), leggi(f"test.split{numero}.bundle")
    assert not set(train) & set(test)
    return train, test


def etichette_frame(video: str) -> np.ndarray:
    """[T] la classe di ogni frame."""
    nomi = classi()
    return np.array([nomi[n] for n in (DATI / "groundTruth" / f"{video}.txt").read_text().split()], dtype=np.int64)


def feature_frame(video: str) -> np.ndarray:
    """[2048, T] le feature I3D di ogni frame."""
    return np.load(DATI / "features" / f"{video}.npy").astype(np.float32)


def dividi_validazione(train: list[str], n_val: int = 3, seme: int = 0) -> tuple[list[str], list[str]]:
    """Tiene fuori `n_val` video di training per scegliere il checkpoint del modello AR
    (il test non si guarda mai). Restituisce (training, validazione), entrambi ordinati."""
    ordinati = sorted(train)
    perm = np.random.RandomState(seme).permutation(len(ordinati))
    return sorted(ordinati[i] for i in perm[n_val:]), sorted(ordinati[i] for i in perm[:n_val])


def fold_di_training(train: list[str], n: int = 4) -> list[list[str]]:
    """I video di training in n gruppi fissi per il cross-fitting (vedi in cima)."""
    perm = np.random.RandomState(SEME_PROTAS).permutation(sorted(train))
    return [sorted(g.tolist()) for g in np.array_split(perm, n)]


def segmenti(etichette: np.ndarray) -> np.ndarray:
    """Le etichette divise in segmenti: righe (inizio, fine esclusa, classe). Un segmento
    è una fila massimale di posizioni con la stessa etichetta."""
    etichette = np.asarray(etichette)
    cambi = np.flatnonzero(etichette[1:] != etichette[:-1]) + 1
    inizi = np.concatenate([[0], cambi])
    fini = np.concatenate([cambi, [len(etichette)]])
    return np.stack([inizi, fini, etichette[inizi]], axis=1).astype(np.int64)


# --- i passi ---------------------------------------------------------------------------

def griglia(n_frame: int) -> tuple[np.ndarray, np.ndarray]:
    """(primo frame, frame dopo l'ultimo) di ogni passo."""
    inizi = np.arange(0, n_frame, FRAME_PER_PASSO, dtype=np.int64)
    return inizi, np.minimum(inizi + FRAME_PER_PASSO, n_frame)


def passi(video: str) -> dict:
    """Feature, etichetta e confini di ogni passo di un video (vedi in cima)."""
    F, y = feature_frame(video), etichette_frame(video)
    assert F.shape[1] == len(y), f"{video}: feature ed etichette di lunghezza diversa"
    inizi, fini = griglia(len(y))
    return {"feature": np.stack([F[:, a:b].mean(axis=1) for a, b in zip(inizi, fini)]),
            "etichette": np.array([np.bincount(y[a:b]).argmax() for a, b in zip(inizi, fini)], dtype=np.int64),
            "inizi": inizi, "fini": fini}


def progresso(etichette: np.ndarray, inizi: np.ndarray, fini: np.ndarray, bg: int) -> np.ndarray:
    """Il progresso dell'azione in corso alla FINE di ogni passo, sul tempo vero: la
    frazione dei frame dell'azione già trascorsi. NaN sul background, dove non c'è
    un'azione di cui misurarlo.

    Si conta in frame e non in passi perché i confini delle azioni cadono a metà dei
    passi: contando i passi un'azione di 59 frame riceverebbe 0.33 dove il vero è 0.47."""
    p = np.full(len(etichette), np.nan, dtype=np.float32)
    for s, e, k in segmenti(etichette):
        if k == bg:
            continue
        primo, ultimo = int(inizi[s]), int(fini[e - 1])
        durata = max(ultimo - primo, 1)
        for t in range(s, e):
            p[t] = min((int(fini[t]) - primo) / durata, 1.0)
    return p


@dataclass
class Video:
    """Un video come lo vede il modello AR: un passo dopo l'altro."""
    nome: str
    codici: np.ndarray        # [T, 16] i token del passo (id 0..1023, vedi tokenizzatore.py)
    azioni: np.ndarray        # [T] la classe di ogni passo
    progresso: np.ndarray     # [T] progresso dell'azione in corso (NaN sul background)
    fini: np.ndarray          # [T] il frame dopo l'ultimo di ogni passo (per tornare ai frame)
    bg: int

    @property
    def n_passi(self) -> int:
        return len(self.azioni)

    @property
    def segmenti(self) -> np.ndarray:
        return segmenti(self.azioni)

    def segmento_di(self, t: int) -> np.ndarray:
        """Il segmento (inizio, fine, classe) che contiene il passo t."""
        for seg in self.segmenti:
            if seg[0] <= t < seg[1]:
                return seg
        raise IndexError(t)


def carica_video(nome: str, numero_split: int) -> Video:
    """Un video con i token dello split (vedi tokenizzatore.py)."""
    p = np.load(RISULTATI / "passi" / f"{nome}.npz")
    codici = np.load(RISULTATI / f"split{numero_split}" / "token" / f"{nome}.npy")
    bg = background()
    return Video(nome, codici, p["etichette"], progresso(p["etichette"], p["inizi"], p["fini"], bg), p["fini"], bg)


def carica_videos(nomi: list[str], numero_split: int) -> list[Video]:
    return [carica_video(v, numero_split) for v in nomi]
