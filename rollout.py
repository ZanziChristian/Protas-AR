"""
Il rollout atteso: il progresso letto da quanto il modello immagina che manchi alla fine.

Il modello non regredisce il progresso. Dato il flusso vero fino al [PROG] del passo t,
IMMAGINA il seguito un passo alla volta, e a ogni confine fra due passi immaginati legge
la probabilità che l'azione finisca lì, h_j = P([EoA]). Lungo la catena si porta avanti
S, la probabilità di essere ancora dentro l'azione:

    P(mancano j passi) = S_j · h_j,     S_{j+1} = S_j · (1 − h_j),     S_0 = 1

Così da una catena esce una DISTRIBUZIONE del numero di passi che mancano, invece del
numero che si otterrebbe tirando a sorte [EoA] (è la stessa quantità con meno rumore:
la moneta di [EoA] non si tira, se ne legge la probabilità). Se dall'inizio dell'azione
sono passati L_passati passi e ne mancano j, il progresso è

    P = L_passati / (L_passati + j)

e la stima è la mediana di P sotto la distribuzione, mediata su 4 catene (4 futuri
immaginati diversi). Nulla è addestrato a produrre questo numero: viene dalla capacità
del modello di prevedere quando arriva [EoA].

Cosa si immagina
----------------
Un passo immaginato ha le stesse regole di un passo vero: al confine il primo codice
(campionato fra i soli codici, perché la fine si legge e non si tira), poi esattamente 16
codici, poi [BLK], poi l'azione (campionata fra le 11), poi il [PROG], che non si
campiona: si scrive. La catena si ferma quando S scende sotto `soglia` (il resto va al
tetto) o dopo `tetto` passi immaginati (48, cioè 16 secondi: quello che resta di S va
al tetto, come un'azione che non è ancora finita).

Molte catene insieme
--------------------
Generare un token alla volta con un batch da 1 lascia la GPU quasi ferma. Qui si
generano molte catene insieme, una per riga del batch, e ognuna ha il suo stato (in che
punto del passo si trova). Le catene finite continuano a ricevere [PAD] finché almeno un
quarto del batch ha finito; allora escono dal batch (la cache si accorcia) e non costano più.
"""
import numpy as np
import torch

from flusso import AZIONE_0, BLK, EOA, N_AZIONI, PAD, PROG, V
from tokenizzatore import PEZZI

CODICI_PER_PASSO = PEZZI                # 16
# le fasi di una catena
CONFINE, DENTRO, AZIONE, METTI_PROG = 0, 1, 2, 3


def _maschere(dim: int, device) -> torch.Tensor:
    """Una riga per situazione: 0 sui token ammessi, −inf sugli altri (si sommano ai logit)."""
    m = torch.full((5, dim), float("-inf"), device=device)
    m[0, :V] = 0.0                         # 0: confine (o [PROG] da mettere): il primo codice del passo
    m[1, :V] = 0.0                         # 1: dentro il passo: un altro codice
    m[2, BLK] = 0.0                        # 2: il passo ha i suoi 16 codici: [BLK]
    m[3, AZIONE_0:AZIONE_0 + N_AZIONI] = 0.0   # 3: dopo [BLK]: l'azione
    m[4, :V] = 0.0
    m[4, EOA] = 0.0                        # 4: per LEGGERE P([EoA]) al confine: [EoA] contro i codici
    return m


@torch.no_grad()
def immagina(modello, prefissi: torch.Tensor, tetto: int, campioni: int = 1, soglia: float = 0.01,
             con_token: bool = False):
    """Catene di passi immaginati dopo ogni prefisso.

    prefissi [Q, L]: flussi veri che finiscono al [PROG] di un passo, tutti lunghi uguali
    (le catene avanzano insieme e condividono le posizioni). Ogni prefisso dà `campioni`
    catene (catena b = prefisso b // campioni).

    Restituisce massa [Q, campioni, tetto + 1]: per ogni catena P(mancano j passi),
    j = 0 .. tetto. Con `con_token` anche i token immaginati da ogni catena (servono al
    self-forcing, addestra.py)."""
    device = prefissi.device
    Q, L0 = prefissi.shape
    B = Q * campioni
    # ogni passo immaginato aggiunge al più 16 codici + [BLK] + azione + [PROG]; se lo
    # spazio da preparare è troppo rispetto al prefisso si torna a concatenare
    capienza = L0 + tetto * (CODICI_PER_PASSO + 3)
    cache = modello.nuova_cache(capienza if capienza <= 4 * L0 else None)
    logit, _, _ = modello(prefissi, cache, inizio=0, solo_ultima=True)
    for c in cache:                        # il prefisso si calcola una volta, la cache si copia per i campioni
        for k, v in c.items():
            if torch.is_tensor(v):
                c[k] = v.repeat_interleave(campioni, dim=0)
    logit = logit[:, -1].repeat_interleave(campioni, dim=0)                     # [B, vocabolario]
    pos = L0
    maschere = _maschere(logit.shape[-1], device)

    fase = np.full(B, CONFINE)
    nel_passo = np.zeros(B, dtype=np.int64)          # codici già immaginati nel passo in corso
    passi = np.zeros(B, dtype=np.int64)              # passi immaginati finora
    finita = np.zeros(B, dtype=bool)
    S = np.ones(B)                                   # P(l'azione non è ancora finita)
    massa = np.zeros((B, tetto + 1))                 # P(mancano j passi)
    token = [[] for _ in range(B)]
    righe = np.arange(B)                             # quale catena sta in ogni riga del batch

    while True:
        f, n = fase[righe], nel_passo[righe]
        situazione = np.where(f == DENTRO, np.where(n < CODICI_PER_PASSO, 1, 2), np.where(f == AZIONE, 3, 0))
        h = torch.softmax(logit + maschere[4], dim=-1)[:, EOA].float().cpu().numpy()
        scelti = torch.multinomial(torch.softmax(logit + maschere[torch.from_numpy(situazione).to(device)], dim=-1), 1)
        scelti = scelti[:, 0].cpu().numpy()

        prossimo = np.full(len(righe), PAD, dtype=np.int64)
        for r, b in enumerate(righe):
            if finita[b]:
                continue
            if fase[b] == METTI_PROG:                  # il [PROG] dello schema: si scrive, non si campiona
                prossimo[r], fase[b] = PROG, CONFINE
                continue
            tok = int(scelti[r])
            if fase[b] == CONFINE:
                # con probabilità h l'azione finisce qui, dopo `passi` passi immaginati;
                # altrimenti la catena continua con il codice campionato
                hr = float(h[r])                      # in doppia precisione, come S
                massa[b, passi[b]] += S[b] * hr
                S[b] *= 1 - hr
                if S[b] < soglia:                      # quello che resta è trascurabile: va al tetto
                    massa[b, tetto] += S[b]
                    finita[b] = True
                    continue
                fase[b], nel_passo[b] = DENTRO, 1
            elif fase[b] == DENTRO:
                if tok == BLK:
                    fase[b] = AZIONE
                else:
                    nel_passo[b] += 1
            elif fase[b] == AZIONE:                    # il passo immaginato è completo
                passi[b] += 1
                if passi[b] >= tetto:
                    massa[b, tetto] += S[b]
                    finita[b] = True
                    continue
                fase[b] = METTI_PROG
            prossimo[r] = tok

        if con_token:
            for r, b in enumerate(righe):
                if prossimo[r] != PAD:
                    token[b].append(int(prossimo[r]))
        finite_qui = finita[righe]
        if finite_qui.all():
            break
        if finite_qui.mean() >= 0.25:                  # le catene finite escono dal batch
            tieni = np.flatnonzero(~finite_qui)
            indici = torch.from_numpy(tieni).to(device)
            for c in cache:
                for k, v in c.items():
                    if torch.is_tensor(v):
                        c[k] = v.index_select(0, indici)
            righe, prossimo = righe[tieni], prossimo[tieni]
        logit = modello(torch.from_numpy(prossimo)[:, None].to(device), cache, inizio=pos)[0][:, -1]
        pos += 1
    massa = massa.reshape(Q, campioni, tetto + 1)
    return (massa, [np.asarray(t, dtype=np.int64) for t in token]) if con_token else massa


def mediana_pesata(valori: np.ndarray, pesi: np.ndarray) -> float:
    """La mediana di una distribuzione discreta: il primo valore, in ordine, in cui la
    massa cumulata arriva a metà."""
    ordine = np.argsort(valori)
    cumulata = np.cumsum(pesi[ordine])
    return float(valori[ordine][np.searchsorted(cumulata, 0.5 * cumulata[-1])])


def progresso_atteso(massa: np.ndarray, passati: int) -> float:
    """massa [campioni, tetto + 1] di un prefisso -> la stima del progresso: le catene si
    mescolano con lo stesso peso e si prende la mediana di P = passati / (passati + j)."""
    j = np.arange(massa.shape[1])
    return mediana_pesata(passati / (passati + j), massa.mean(0))
