"""
La CAS di ProTAS: il segmentatore causale a 4 stadi (una MS-TCN resa online).

ProTAS (Shen & Elhamifar, CVPR 2024) segmenta il video con una MS-TCN (Farha & Gall,
CVPR 2019) in cui ogni convoluzione guarda solo il passato: l'etichetta del frame t si
decide con i frame fino a t. Il codice di ProTAS è stato letto come riferimento e
riscritto qui (il loro repository non ha licenza).

    feature I3D [1, 2048, T] -> stadio 1 -> logit [1, 11, T]
                             -> softmax -> stadio 2 -> ... -> stadio 4

Ogni stadio: una convoluzione 1x1 porta l'ingresso a 64 canali, 10 strati dilatati
residui (dilatazione 1, 2, 4, ..., 512: il campo visivo raddoppia a ogni strato), e una
convoluzione 1x1 dà gli 11 logit. Gli stadi dopo il primo leggono le probabilità dello
stadio prima e le correggono (soprattutto la sovra-segmentazione).

La loss di ogni stadio è quella di MS-TCN: entropia incrociata più 0.15 volte un
termine che penalizza i salti delle log-probabilità fra frame vicini (troncato a 16).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dati import SEME_PROTAS


class StratoDilatato(nn.Module):
    """Uno strato residuo con convoluzione dilatata CAUSALE: si aggiungono 2d valori a
    entrambi i lati (copiando il bordo) e si tagliano gli ultimi 2d dell'uscita, così
    l'uscita in t vede i frame t, t − d, t − 2d."""

    def __init__(self, dilatazione: int, canali: int):
        super().__init__()
        self.d = dilatazione
        self.conv = nn.Conv1d(canali, canali, 3, padding=2 * dilatazione, padding_mode="replicate", dilation=dilatazione)
        self.uno = nn.Conv1d(canali, canali, 1)
        self.drop = nn.Dropout()                     # 0.5, come in ProTAS

    def forward(self, x):
        h = F.relu(self.conv(x))[..., :-2 * self.d]
        return x + self.drop(self.uno(h))


class Stadio(nn.Module):
    def __init__(self, in_dim: int, canali: int, K: int, strati: int = 10):
        super().__init__()
        self.ingresso = nn.Conv1d(in_dim, canali, 1)
        self.strati = nn.ModuleList(StratoDilatato(2 ** i, canali) for i in range(strati))
        self.uscita = nn.Conv1d(canali, K, 1)

    def forward(self, x):
        h = self.ingresso(x)
        for s in self.strati:
            h = s(h)
        return self.uscita(h)


class CAS(nn.Module):
    """[1, D, T] -> logit di ogni stadio [stadi, 1, K, T]."""

    def __init__(self, D: int, K: int, stadi: int = 4, canali: int = 64):
        super().__init__()
        self.stadi = nn.ModuleList([Stadio(D, canali, K)] + [Stadio(K, canali, K) for _ in range(stadi - 1)])

    def forward(self, x):
        uscite = [self.stadi[0](x)]
        for s in self.stadi[1:]:
            uscite.append(s(F.softmax(uscite[-1], dim=1)))
        return torch.stack(uscite)


def perdita_stadio(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """La loss di MS-TCN per uno stadio. logit [1, K, T], y [T]."""
    ce = F.cross_entropy(logit[0].T, y)
    lp = F.log_softmax(logit, dim=1)
    liscio = torch.clamp((lp[:, :, 1:] - lp.detach()[:, :, :-1]) ** 2, min=0, max=16).mean()
    return ce + 0.15 * liscio


def addestra(dati: dict, video: list[str], K: int, device: str, maschera_canali: float = 0.5,
             epoche: int = 50) -> CAS:
    """La ricetta di ProTAS: 50 epoche, Adam lr 5e-4, un video per passo, la loss su ogni
    stadio. In più (non in ProTAS) il "channel masking" di ASFormer su GTEA: in
    addestramento si spegne a caso metà dei 2048 canali I3D, lo stesso canale in tutti i
    frame, così con 21 video la rete non si affida a poche feature molto informative."""
    torch.manual_seed(SEME_PROTAS)
    D = next(iter(dati.values()))[0].shape[0]
    cas = CAS(D, K).to(device)
    opt = torch.optim.Adam(cas.parameters(), lr=5e-4)
    rng = np.random.RandomState(SEME_PROTAS)
    for epoca in range(epoche):
        cas.train()
        totale = 0.0
        for v in rng.permutation(video):
            X, y = dati[v]
            X = F.dropout1d(torch.from_numpy(X)[None].to(device), maschera_canali)
            loss = sum(perdita_stadio(l, torch.from_numpy(y).to(device)) for l in cas(X))
            opt.zero_grad(); loss.backward(); opt.step()
            totale += loss.item()
        if (epoca + 1) % 10 == 0:
            print(f"  CAS epoca {epoca + 1}: loss {totale / len(video):.3f}", flush=True)
    return cas.eval()


@torch.no_grad()
def logit(cas: CAS, dati: dict, video: list[str], device: str) -> dict:
    """I logit dell'ultimo stadio [K, T] di ogni video."""
    return {v: cas(torch.from_numpy(dati[v][0])[None].to(device))[-1][0].cpu().numpy() for v in video}
