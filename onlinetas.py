"""
OnlineTAS (Zhong, Ding & Yao, NeurIPS 2024): il secondo segmentatore online.

È il metodo più recente pensato apposta per la segmentazione online, e su GTEA riporta
più della CAS di ProTAS (75.8 contro 74.0 di accuratezza). Il codice non è pubblico: è
riscritto dalla descrizione del paper, e dove il paper non dice tutto la scelta è
scritta accanto.

Come funziona
-------------
Il video si divide in clip di w = 64 frame. Per ogni clip:
  1. una GRU, che scorre tutto il video in ordine, dà a ogni frame un riassunto del
     passato (c_GRU): è causale per costruzione;
  2. il modulo CFA arricchisce i frame della clip con una MEMORIA del passato, due volte
     di seguito (con gli stessi pesi):
       - auto-attenzione dentro la clip, alla Swin: prima dentro ciascuna metà, poi fra
         le due metà, con un bias appreso per la distanza fra i frame;
       - un decoder Transformer (2 strati) che aggiorna la memoria guardando la clip;
       - un'attenzione incrociata: la clip interroga la memoria, più un residuo con c_GRU;
  3. una TCN causale a uno stadio (10 strati dilatati, come la CAS) dà i logit;
  4. la memoria si aggiorna: ha sempre w posti, divisi in memoria LUNGA (un token per
     ogni clip passata, al più 2/3 dei posti, oltre si scarta il più vecchio) e memoria
     CORTA (la coda della clip appena elaborata, nei posti che restano).

Online in senso stretto
-----------------------
Per il frame t si elabora la clip che FINISCE in t e si tiene solo la sua ultima
predizione. Le clip che finiscono in t, t − w, t − 2w, ... sono una divisione del video
in clip spostata di (t mod w): si fanno quindi tutte le w divisioni possibili insieme,
come un batch di w sequenze, e per ogni frame si legge la clip giusta. Il frame t non
vede nulla dopo t.

Scelte dove il paper non è esplicito:
  * la GRU lavora su una proiezione delle I3D a 64 canali (convoluzione 1x1);
  * il primo stato della memoria è c_GRU della prima clip;
  * in addestramento le clip partono da uno spostamento casuale (uno per video e per
    epoca), così il modello vede tutte le divisioni che l'inferenza usa; i frame di
    riempimento (prima e dopo il video) copiano il primo e l'ultimo e non entrano nella loss;
  * le attenzioni sono pre-norm (LayerNorm prima, residuo dopo).
Si allena come nel paper: 50 epoche, Adam lr 5e-4, la loss di MS-TCN.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cas import Stadio
from dati import SEME_PROTAS

IGNORA = -100                # etichetta dei frame di riempimento: fuori dalla loss


class Attenzione(nn.Module):
    """Attenzione a più teste: softmax(Q Kᵀ / √d + B) V, dove B è un numero appreso per
    ogni testa e per ogni distanza i − j fra query e chiave (il bias relativo di Swin)."""

    def __init__(self, D: int, teste: int, lato: int):
        super().__init__()
        self.h, self.lato = teste, lato
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.bias = nn.Parameter(torch.zeros(teste, 2 * lato - 1))

    def forward(self, q, kv):                                        # q [B, Lq, D], kv [B, Lk, D]
        B, Lq, D = q.shape
        Lk, d = kv.shape[1], D // self.h
        Q = self.q(q).view(B, Lq, self.h, d).transpose(1, 2)        # [B, h, Lq, d]
        K = self.k(kv).view(B, Lk, self.h, d).transpose(1, 2)
        V = self.v(kv).view(B, Lk, self.h, d).transpose(1, 2)
        distanza = torch.arange(Lq, device=q.device)[:, None] - torch.arange(Lk, device=q.device)[None] + self.lato - 1
        punteggi = Q @ K.transpose(-1, -2) / d ** 0.5 + self.bias[:, distanza]
        return self.o((punteggi.softmax(-1) @ V).transpose(1, 2).reshape(B, Lq, D))


class StratoDecoder(nn.Module):
    """La memoria guarda se stessa, poi la clip, poi una rete a due strati."""

    def __init__(self, D: int, teste: int, w: int):
        super().__init__()
        self.sa, self.ca = Attenzione(D, teste, w), Attenzione(D, teste, w)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))
        self.n1, self.n2, self.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, M, c):
        m = self.n1(M)
        M = M + self.sa(m, m)
        M = M + self.ca(self.n2(M), c)
        return M + self.ffn(self.n3(M))


class CFA(nn.Module):
    """Context-aware feature augmentation: la clip c [B, w, D] arricchita dalla memoria M [B, w, D]."""

    def __init__(self, D: int, w: int, teste_sa: int = 4, teste_td: int = 8, ripetizioni: int = 2):
        super().__init__()
        self.w, self.ripetizioni = w, ripetizioni
        self.locale = Attenzione(D, teste_sa, w // 2)       # dentro ciascuna metà della clip
        self.fra_meta = Attenzione(D, teste_sa, w)          # fra le due metà
        self.decoder = nn.ModuleList(StratoDecoder(D, teste_td, w) for _ in range(2))
        self.incrociata = Attenzione(D, teste_sa, w)
        self.n1, self.n2, self.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)

    def auto_attenzione(self, c):
        B, w, D = c.shape
        x = c.reshape(B * 2, w // 2, D)                     # le due metà come due sequenze
        n = self.n1(x)
        x = (x + self.locale(n, n)).reshape(B, w, D)
        n = self.n2(x)
        return x + self.fra_meta(n, n)

    def forward(self, c, M):
        h = c
        for _ in range(self.ripetizioni):
            c_sa = self.auto_attenzione(h)
            memoria = M
            for strato in self.decoder:
                memoria = strato(memoria, h)
            h = self.incrociata(self.n3(c_sa), memoria) + h
        return h


class OnlineTAS(nn.Module):
    def __init__(self, D_in: int, K: int, w: int = 64, D: int = 64, maschera_canali: float = 0.5):
        super().__init__()
        self.w, self.maschera_canali = w, maschera_canali
        self.proiezione = nn.Conv1d(D_in, D, 1)
        self.gru = nn.GRU(D, D, batch_first=True)
        self.cfa = CFA(D, w)
        self.token = nn.Conv1d(D, D, w)                     # una clip -> un token di memoria lunga
        self.tcn = Stadio(D, D, K)

    def contesto(self, X):
        """Feature [1, D_in, T] -> c_GRU [T, D]. In addestramento prima il channel masking
        (vedi cas.addestra)."""
        if self.training and self.maschera_canali > 0:
            X = F.dropout1d(X, self.maschera_canali)
        return self.gru(self.proiezione(X).transpose(1, 2))[0][0]

    def per_clip(self, c):
        """c [B, Tp, D], Tp multiplo di w -> logit [B, K, Tp]: una clip dopo l'altra, con la
        memoria che passa dall'una all'altra."""
        B, Tp, D = c.shape
        w = self.w
        lunga, M, uscite = [], c[:, :w], []
        for k in range(Tp // w):
            ct = self.cfa(c[:, k * w:(k + 1) * w], M)                       # [B, w, D]
            uscite.append(self.tcn(ct.transpose(1, 2)))                     # [B, K, w]
            m = self.token(ct.transpose(1, 2)).transpose(1, 2)              # [B, 1, D]
            lunga = (lunga if len(lunga) <= 2 * w / 3 else lunga[1:]) + [m]
            M = torch.cat(lunga + [ct[:, len(lunga):]], dim=1)              # sempre w posti
        return torch.cat(uscite, dim=2)

    def riempi(self, c, spostamento: int, n_clip: int):
        """c [T, D] -> [n_clip * w, D]: davanti w − 1 − spostamento copie del primo frame
        (così le clip finiscono nei frame t con t mod w = spostamento), dietro copie
        dell'ultimo. Restituisce anche quanti frame sono stati aggiunti davanti."""
        davanti = self.w - 1 - spostamento
        dietro = n_clip * self.w - davanti - len(c)
        return torch.cat([c[:1].expand(davanti, -1), c, c[-1:].expand(dietro, -1)]), davanti

    @torch.no_grad()
    def online(self, X):
        """Feature [1, D_in, T] -> logit [K, T] in senso strettamente online (vedi in cima)."""
        c = self.contesto(X)
        T, w = len(c), self.w
        n_clip = -(-(T + w - 1) // w)
        divisioni = torch.stack([self.riempi(c, o, n_clip)[0] for o in range(w)])   # [w, n_clip*w, D]
        L = self.per_clip(divisioni)
        t = torch.arange(T, device=c.device)
        o = t % w
        return L[o, :, (w - 1 - o) + t].T


def perdita_mascherata(L: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """La loss di MS-TCN (vedi cas.perdita_stadio) senza i frame di riempimento.
    L [1, K, Tp], y [Tp] con IGNORA dove non c'è video."""
    ce = F.cross_entropy(L[0].T, y, ignore_index=IGNORA)
    lp = F.log_softmax(L, dim=1)
    vero = y != IGNORA
    coppie = (vero[1:] & vero[:-1]).float()                 # frame consecutivi entrambi veri
    d = torch.clamp((lp[0, :, 1:] - lp.detach()[0, :, :-1]) ** 2, min=0, max=16)
    return ce + 0.15 * (d * coppie).sum() / (coppie.sum() * L.shape[1]).clamp(min=1)


def addestra(dati: dict, video: list[str], K: int, device: str, maschera_canali: float = 0.5,
             epoche: int = 50) -> OnlineTAS:
    """50 epoche, un video per passo, clip con uno spostamento casuale (vedi in cima)."""
    torch.manual_seed(SEME_PROTAS)
    D_in = next(iter(dati.values()))[0].shape[0]
    m = OnlineTAS(D_in, K, maschera_canali=maschera_canali).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    rng = np.random.RandomState(SEME_PROTAS)
    w = m.w
    for epoca in range(epoche):
        m.train()
        totale = 0.0
        for v in rng.permutation(video):
            X, y = dati[v]
            c = m.contesto(torch.from_numpy(X)[None].to(device))
            spostamento = rng.randint(w)
            n_clip = -(-(len(c) + w - 1 - spostamento) // w)
            seq, davanti = m.riempi(c, spostamento, n_clip)
            yy = torch.full((len(seq),), IGNORA, dtype=torch.long)
            yy[davanti:davanti + len(y)] = torch.from_numpy(y)
            loss = perdita_mascherata(m.per_clip(seq[None]), yy.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            totale += loss.item()
        if (epoca + 1) % 10 == 0:
            print(f"  OnlineTAS epoca {epoca + 1}: loss {totale / len(video):.3f}", flush=True)
    return m.eval()


def logit(m: OnlineTAS, dati: dict, video: list[str], device: str) -> dict:
    """I logit online [K, T] di ogni video."""
    return {v: m.online(torch.from_numpy(dati[v][0])[None].to(device)).cpu().numpy() for v in video}
