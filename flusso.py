"""
Dal video al flusso di token che il transformer legge, come un testo.

Il vocabolario (1041 token)
---------------------------
    0 .. 1023        i codici visivi (tokenizzatore.py): 16 per passo
    1024  [BOT]      inizio del video
    1025  [EoA]      fine dell'azione in corso
    1026  [PROG]     "qui si stima il progresso"
    1027  [PAD]      riempimento del batch, mai nella loss
    1028  [BLK]      fine dei codici del passo
    1029 .. 1039     [A_k], l'azione in corso (11 classi, background compreso)
    1040  [A_?]      "azione sconosciuta": il modello finale non lo usa mai; resta perché
                     fa parte della forma dei pesi addestrati

Il flusso
---------
Per ogni passo t, nell'ordine:

    c_t^0 ... c_t^15  i 16 codici visivi del passo
    [BLK]             il passo è completo
    [A_k]             l'azione in corso, secondo il segmentatore (in addestramento: le
                      etichette vere, con i confini spostati a caso, vedi `confini_rumorosi`)
    [PROG]            la testa del progresso si legge qui
    [EoA]             solo se il passo t chiude un'azione (non il background)

cioè:  [BOT] c c ... c [BLK] [A_3] [PROG] c ... [BLK] [A_3] [PROG] [EoA] c ...   (16 c per passo)

[EoA] cade quindi solo fra due passi: il modello, dopo ogni [PROG], decide fra "l'azione
finisce qui" ([EoA]) e "continua" (il primo codice del passo dopo). È la decisione su
cui si regge la stima del progresso (rollout.py).

Cosa impara il modello e dove (la posizione i predice il token i + 1)
--------------------------------------------------------------------
  * testa visiva: il token successivo, ovunque tranne dove il successivo è [PROG] (lo
    schema lo impone, non c'è niente da imparare) e oltre la fine;
  * fine dell'azione: a ogni confine fra due passi (la posizione del [PROG]), se segue
    [EoA]. È una loss a parte (addestra.py) perché nella testa visiva questa decisione
    sarebbe un bersaglio su ~20;
  * testa del progresso: alle posizioni [PROG] dentro un'azione, bersaglio il progresso
    vero del passo (dati.progresso);
  * testa dell'azione: alle posizioni [BLK], bersaglio l'azione vera del passo.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from dati import Video, segmenti
from tokenizzatore import CODICI, PEZZI

V = PEZZI * CODICI                             # codici visivi (1024)
BOT, EOA, PROG, PAD, BLK = V, V + 1, V + 2, V + 3, V + 4
N_AZIONI = 11
AZIONE_0 = V + 5                               # [A_k] = AZIONE_0 + k
DIM_VOCABOLARIO = V + 5 + N_AZIONI + 1         # + [A_?]


@dataclass
class Flusso:
    token: np.ndarray            # [L] il flusso
    pos_prog: np.ndarray         # [T] dove sta il [PROG] di ogni passo
    progresso: np.ndarray        # [T] il progresso vero del passo (0 sul background)
    in_azione: np.ndarray        # [T] il passo sta dentro un'azione (lì la testa del progresso ha la loss)
    bersaglio_azione: np.ndarray # [L] l'azione vera alle posizioni [BLK], −1 altrove
    pos_confine: np.ndarray      # [C] i confini fra due passi (la posizione del [PROG])
    eoa_al_confine: np.ndarray   # [C] lì segue [EoA]?

    def __len__(self):
        return len(self.token)


def costruisci(video: Video, azioni_nel_flusso: np.ndarray | None = None) -> Flusso:
    """Il flusso di un video. `azioni_nel_flusso` [T] sono le azioni scritte nei token
    [A_k] (di default quelle vere): con quelle di un segmentatore anche gli [EoA] seguono
    i suoi segmenti, perché è lì che il sistema crede che un'azione finisca. I bersagli
    delle loss restano sempre quelli veri."""
    azioni = video.azioni if azioni_nel_flusso is None else azioni_nel_flusso
    fine_azione = {int(e) - 1 for s, e, k in segmenti(azioni) if k != video.bg}
    T = video.n_passi
    token = [BOT]
    pos_prog, pos_confine, eoa_al_confine = [], [], []
    bersaglio_azione = []                                   # (posizione del [BLK], azione vera)
    for t in range(T):
        token.extend(int(c) for c in video.codici[t])
        token.append(BLK)
        bersaglio_azione.append(len(token) - 1)
        token.append(AZIONE_0 + int(azioni[t]))
        pos_prog.append(len(token))
        token.append(PROG)
        # Il confine dopo il passo t: da qui il modello predice [EoA] o il primo codice
        # del passo dopo. Dopo l'ultimo passo di un video che finisce sul background non
        # viene niente, e quel confine non ha bersaglio.
        if t in fine_azione or t < T - 1:
            pos_confine.append(len(token) - 1)
            eoa_al_confine.append(t in fine_azione)
        if t in fine_azione:
            token.append(EOA)
    token = np.asarray(token, dtype=np.int64)
    bersaglio = np.full(len(token), -1, dtype=np.int64)
    bersaglio[bersaglio_azione] = video.azioni
    in_azione = video.azioni != video.bg
    return Flusso(token, np.asarray(pos_prog, dtype=np.int64), np.nan_to_num(video.progresso).astype(np.float32),
                  in_azione, bersaglio, np.asarray(pos_confine, dtype=np.int64), np.asarray(eoa_al_confine, dtype=bool))


# --- l'augmentation dell'addestramento -------------------------------------------------

def confini_rumorosi(azioni: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Le azioni con ogni cambio spostato di d passi, d uniforme fra −1 e +3: d > 0 il
    cambio arriva in ritardo, d < 0 in anticipo. A test le azioni nel flusso vengono dal
    segmentatore, che sbaglia proprio i confini, quasi sempre in ritardo: così il modello
    impara a non fidarsi ciecamente del token d'azione."""
    sporche = azioni.copy()
    T = len(azioni)
    for b in np.flatnonzero(azioni[1:] != azioni[:-1]) + 1:      # il passo in cui l'azione cambia
        d = rng.randint(-1, 4)
        if d > 0:
            sporche[b: min(b + d, T)] = azioni[b - 1]
        elif d < 0:
            sporche[max(b + d, 0): b] = azioni[b]
    return sporche


def cambia_velocita(video: Video, rng: np.random.RandomState, minima: float = 0.8, massima: float = 1.25) -> Video:
    """Una copia del video con ogni azione eseguita a una velocità diversa.

    Il modello potrebbe stimare il progresso contando i passi e dividendo per una durata
    tipica. Se la stessa azione dura a volte 8 passi e a volte 12, contare non basta più:
    deve guardare cosa succede nella scena. Per ogni azione si estrae una velocità s fra
    `minima` e `massima` (uniforme in scala logaritmica) e l'azione da N passi ne diventa
    round(N / s), ripetendo o saltando passi; il primo e l'ultimo restano sempre. Il
    progresso si rilegge sulla nuova durata dalla curva originale. Il background non cambia."""
    indici, progresso = [], []
    for s, e, k in video.segmenti:
        s, e = int(s), int(e)
        N = e - s
        if k == video.bg:
            indici.extend(range(s, e))
            progresso.extend(video.progresso[s:e])
            continue
        velocita = float(np.exp(rng.uniform(np.log(minima), np.log(massima))))
        N_nuovo = max(1, int(round(N / velocita)))
        indici.extend(s + (np.array([N - 1]) if N_nuovo == 1 else np.round(np.linspace(0, N - 1, N_nuovo)).astype(np.int64)))
        # la curva passa per (−1, 0) e per i valori veri ai passi 0 .. N−1; il passo nuovo j
        # cade al punto (j + 1) / N_nuovo dell'azione
        x = np.arange(-1, N, dtype=np.float64)
        y = np.concatenate([[0.0], np.asarray(video.progresso[s:e], dtype=np.float64)])
        progresso.extend(np.interp((np.arange(N_nuovo) + 1) * N / N_nuovo - 1, x, y).astype(np.float32))
    indici = np.asarray(indici, dtype=np.int64)
    azioni = video.azioni[indici]
    progresso = np.asarray(progresso, dtype=np.float32)
    progresso[azioni == video.bg] = np.nan
    return Video(video.nome, video.codici[indici], azioni, progresso, video.fini[indici], video.bg)


# --- finestre e batch ------------------------------------------------------------------

def finestre_dell_epoca(flussi: list[Flusso], lunghezza: int, rng: np.random.RandomState) -> list[tuple[int, int]]:
    """Le finestre di un'epoca, (quale flusso, dove inizia), in ordine casuale.

    Un flusso è lungo ~2.300 token e il modello ne legge 1.024 per volta: un flusso di L
    token si copre con finestre affiancate, e a ogni epoca tutta la fila si sposta di un
    tratto casuale, così i tagli non cadono sempre negli stessi punti. Un'epoca vede ogni
    token una volta."""
    elementi = []
    for i, f in enumerate(flussi):
        L = len(f)
        if L <= lunghezza:
            elementi.append((i, 0))
            continue
        spostamento = rng.randint(0, lunghezza)
        for k in range(math.ceil((L + spostamento) / lunghezza)):
            elementi.append((i, min(max(0, k * lunghezza - spostamento), L - lunghezza)))
    ordine = rng.permutation(len(elementi))
    return [elementi[j] for j in ordine]


def batch(flussi: list[Flusso], inizi: list[int], lunghezza: int) -> dict:
    """Le finestre [inizio, inizio + lunghezza) di più flussi, una sotto l'altra, riempite
    a destra con [PAD]. Le posizioni dei [PROG] e dei confini diventano coppie (riga del
    batch, posizione nella finestra)."""
    finestre = [(s, min(len(f), s + lunghezza)) for f, s in zip(flussi, inizi)]
    B, L = len(flussi), max(e - s for s, e in finestre)
    token = np.full((B, L), PAD, dtype=np.int64)
    bersaglio_azione = np.full((B, L), -1, dtype=np.int64)
    prog_b, prog_pos, progresso, in_azione = [], [], [], []
    confine_b, confine_pos, confine_eoa = [], [], []
    for b, (f, (s, e)) in enumerate(zip(flussi, finestre)):
        token[b, :e - s] = f.token[s:e]
        bersaglio_azione[b, :e - s] = f.bersaglio_azione[s:e]
        dentro = (f.pos_prog >= s) & (f.pos_prog < e)
        prog_b += [b] * int(dentro.sum()); prog_pos += list(f.pos_prog[dentro] - s)
        progresso += list(f.progresso[dentro]); in_azione += list(f.in_azione[dentro])
        # un confine conta se nella finestra c'è anche il token che lo segue
        dentro = (f.pos_confine >= s) & (f.pos_confine < e - 1)
        confine_b += [b] * int(dentro.sum()); confine_pos += list(f.pos_confine[dentro] - s)
        confine_eoa += list(f.eoa_al_confine[dentro])
    # la testa visiva impara dove il token successivo esiste, è vero e non è [PROG]
    impara_visivo = np.zeros((B, L), dtype=bool)
    for b, (s, e) in enumerate(finestre):
        impara_visivo[b, :e - s - 1] = token[b, 1:e - s] != PROG
    t = lambda x, tipo: torch.as_tensor(np.asarray(x), dtype=tipo)
    return {"token": torch.from_numpy(token), "impara_visivo": torch.from_numpy(impara_visivo),
            "bersaglio_azione": torch.from_numpy(bersaglio_azione),
            "prog_b": t(prog_b, torch.long), "prog_pos": t(prog_pos, torch.long),
            "progresso": t(progresso, torch.float32), "in_azione": t(in_azione, torch.bool),
            "confine_b": t(confine_b, torch.long), "confine_pos": t(confine_pos, torch.long),
            "confine_eoa": t(confine_eoa, torch.bool)}
