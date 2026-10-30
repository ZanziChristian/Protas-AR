"""
Passo 3 della catena: l'addestramento del modello autoregressivo.

    python addestra.py --split 1
    -> risultati/split1/ar_base/best_nt.pt                   fase 1
    -> risultati/split1/ar_sf/epoca{10..100}.pt               fase 2
    -> risultati/split1/ar_sf/modello_finale.pt               fase 3: la media dei pesi

Fase 1: addestramento (2000 epoche)
-----------------------------------
Il modello legge finestre di 1024 token del flusso (flusso.py) e impara quattro cose,
tutte entropie incrociate:

    L =       CE(token successivo)           immaginare il futuro
      + 2   · BCE(P([EoA]) a ogni confine)   l'azione finisce qui o continua?
      +       BCE(progresso)                 a che punto dell'azione siamo, ai [PROG]
      + 0.1 · CE(azione in corso)            quale azione è in corso, ai [BLK]

La prima è la domanda del progetto: un modello che impara a prevedere il token successivo
sa stimare il progresso GRAZIE a quell'informazione? La seconda allena la decisione da
cui si legge il progresso (rollout.py): nella prima sarebbe un bersaglio su ~20, qui ha
un peso suo, mediata sui confini.

Due aumenti dei dati, nuovi a ogni epoca:
  * metà dei video cambia la velocità delle sue azioni (×0.8 – ×1.25), così contare i
    passi non basta a stimare il progresso (flusso.cambia_velocita);
  * in metà dei video i token d'azione hanno i confini spostati di −1..+3 passi, come li
    sbaglia un segmentatore online (flusso.confini_rumorosi).

Il checkpoint: quello con l'errore più basso sul token successivo nei 3 video di
validazione (`best_nt.pt`), cioè il miglior modello generativo. Si sceglie sulla
validazione, mai sul test.

Fase 2: self-forcing (100 epoche, dal checkpoint della fase 1)
-------------------------------------------------------------
In fase 1 il modello impara P([EoA]) dopo passi VERI; il rollout però la legge dopo
passi che ha immaginato LUI. Il self-forcing (Huang et al., NeurIPS 2025, qui solo sulla
fine) lo allena nella situazione del rollout. Ogni 4 passi di ottimizzazione:
  1. si scelgono 8 istanti t dentro un'azione, di cui si sa quanti passi mancano davvero
     alla fine, r;
  2. da ognuno il modello immagina, senza gradiente e con le regole del rollout, i passi
     che servono;
  3. un forward con gradiente su prefisso vero + futuro immaginato, e a ogni confine j
     una BCE su P([EoA]) con il bersaglio del tempo vero: 0 per j < r, 1 per j = r (se
     l'azione dura oltre 48 passi, tutti 0 fino al 48°).
Sommate, le BCE di una catena sono −log della probabilità che il rollout atteso dà alla
durata vera: abbassarle vuol dire alzare quella probabilità.

Fase 3: media dei pesi
----------------------
Il modello finale è la media dei pesi delle epoche 50, 60, ..., 100 del self-forcing
(come Stochastic Weight Averaging, Izmailov et al., 2018): i pesi di fine addestramento
oscillano intorno a una zona buona, e la media ci cade più vicino. Così nessun checkpoint
si sceglie sul rumore di 3 video di validazione. La decisione è stata presa prima di
vedere i risultati.
"""
import argparse
import json
import math
import os
import time

# Memoria della GPU a segmenti espandibili, prima di importare torch: il self-forcing
# alterna forward di lunghezze molto diverse e l'allocatore di default frammenta la memoria.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

from dati import RISULTATI, carica_videos, dividi_validazione, split
from flusso import EOA, PAD, PROG, V, batch, cambia_velocita, confini_rumorosi, costruisci, finestre_dell_epoca
from modello import ModelloAR
from rollout import immagina

FINESTRA, BATCH = 1024, 4
PESO_VISIVO, PESO_FINE, PESO_PROGRESSO, PESO_AZIONE = 1.0, 2.0, 1.0, 0.1
SF_OGNI, SF_PARTENZE, SF_ORIZZONTE = 4, 8, 48
EPOCHE_MEDIA = (50, 60, 70, 80, 90, 100)


# --- le loss -----------------------------------------------------------------------------

def log_fine(logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """logit [N, vocabolario] -> (log P([EoA]), log (1 − P([EoA]))). Tutti e due con
    logsumexp: P([EoA]) può valere 1e-4, e log(1 − P) calcolato come log1p(−P)
    perderebbe precisione proprio dove serve."""
    tutti = torch.logsumexp(logit, dim=-1)
    altri = logit.clone()
    altri[:, EOA] = float("-inf")
    return logit[:, EOA] - tutti, torch.logsumexp(altri, dim=-1) - tutti


def perdita(modello, b: dict) -> tuple[torch.Tensor, dict]:
    """La loss di un batch (vedi in cima) e le sue quattro parti."""
    logit, progresso, logit_azione = modello(b["token"])
    zero = logit.sum() * 0.0                                  # uno zero con il gradiente
    # --- il token successivo, dove si impara
    m = b["impara_visivo"][:, :-1]
    visivo = F.cross_entropy(logit[:, :-1][m], b["token"][:, 1:][m])
    # --- la fine dell'azione, a ogni confine fra due passi
    fine = zero
    if len(b["confine_pos"]):
        log_p, log_non = log_fine(logit[b["confine_b"], b["confine_pos"]].float())
        y = b["confine_eoa"].float()
        fine = -(y * log_p + (1 - y) * log_non).mean()
    # --- il progresso, ai [PROG] dentro un'azione
    p = progresso[b["prog_b"], b["prog_pos"]]
    dentro = b["in_azione"]
    prog = F.binary_cross_entropy(p[dentro].clamp(1e-6, 1 - 1e-6), b["progresso"][dentro]) if dentro.any() else progresso.sum() * 0.0
    # --- l'azione in corso, ai [BLK]
    azione = F.cross_entropy(logit_azione.reshape(-1, logit_azione.shape[-1]), b["bersaglio_azione"].reshape(-1), ignore_index=-1)
    totale = PESO_VISIVO * visivo + PESO_FINE * fine + PESO_PROGRESSO * prog + PESO_AZIONE * azione
    return totale, {"visivo": visivo.item(), "fine": fine.item(), "progresso": prog.item(), "azione": azione.item()}


def self_forcing(modello, flussi, videos, rng: np.random.RandomState, device: str) -> float:
    """La loss del self-forcing (vedi in cima), con il suo backward già fatto.

    Le catene si trattano una alla volta: 8 sequenze da ~1.600 token con il gradiente non
    stanno insieme in una GPU da 12 GB. Ogni catena fa subito il backward di
    loss / 8 e libera la memoria; i gradienti si sommano a quelli del batch."""
    candidati = []                                     # (flusso, posizione del [PROG], passo, passi che mancano)
    for i, (f, v) in enumerate(zip(flussi, videos)):
        fine_azione = np.zeros(v.n_passi, dtype=np.int64)      # il passo dopo la fine del segmento di ogni passo
        for s, e, _ in v.segmenti:
            fine_azione[s:e] = e
        for t in np.flatnonzero(f.in_azione):
            candidati.append((i, int(f.pos_prog[t]), int(t), int(fine_azione[t]) - 1 - int(t)))
    if not candidati:
        return 0.0
    partenze = [candidati[k] for k in rng.choice(len(candidati), size=min(SF_PARTENZE, len(candidati)), replace=False)]
    # bersagli ai confini j = 0, 1, ...: r zeri e un 1, o 48 zeri se l'azione dura di più
    bersagli = []
    for *_, r in partenze:
        e = np.zeros(SF_ORIZZONTE if r > SF_ORIZZONTE else r + 1, dtype=np.float32)
        if r <= SF_ORIZZONTE:
            e[r] = 1.0
        bersagli.append(e)
    prefissi = [torch.from_numpy(flussi[i].token[max(0, pos + 1 - FINESTRA): pos + 1]) for i, pos, _, _ in partenze]

    # --- il futuro immaginato: senza gradiente e senza dropout, come nel rollout. Le
    # partenze con prefissi lunghi uguali si immaginano insieme, ognuna poi tiene i suoi passi
    futuri = [None] * len(partenze)
    modello.eval()
    gruppi = {}
    for k, pre in enumerate(prefissi):
        gruppi.setdefault(len(pre), []).append(k)
    for ks in gruppi.values():
        da_immaginare = max(len(bersagli[k]) - 1 for k in ks)
        if da_immaginare == 0:
            for k in ks:
                futuri[k] = np.zeros(0, dtype=np.int64)
            continue
        _, token = immagina(modello, torch.stack([prefissi[k] for k in ks]).to(device), da_immaginare + 1,
                            soglia=0.0, con_token=True)
        for k, tok in zip(ks, token):
            n = len(bersagli[k]) - 1                    # passi che servono: fino all'n-esimo [PROG]
            futuri[k] = tok[:np.flatnonzero(tok == PROG)[n - 1] + 1] if n else tok[:0]
    modello.train()

    # --- per ogni catena: forward con gradiente e BCE ai confini (fine del prefisso, poi i [PROG] immaginati)
    perdite = []
    for pre, fut, e in zip(prefissi, futuri, bersagli):
        seq = torch.from_numpy(np.concatenate([pre.numpy(), fut]))[None].to(device)
        confini = np.concatenate([[len(pre) - 1], len(pre) + np.flatnonzero(fut == PROG)])
        logit, _, _ = modello(seq)
        log_p, log_non = log_fine(logit[0, torch.from_numpy(confini).to(device)].float())
        y = torch.from_numpy(e).to(device)
        l = -(y * log_p + (1 - y) * log_non).sum()
        (l / len(partenze)).backward()
        perdite.append(l.item())
    return float(np.mean(perdite))


@torch.no_grad()
def errore_token_successivo(modello, flussi, device: str) -> float:
    """Entropia incrociata media sul token successivo, sulle posizioni in cui il token da
    prevedere è un codice. Flussi più lunghi della finestra si leggono a finestre che
    avanzano di mezza finestra; ogni posizione si conta una volta."""
    somma, n = 0.0, 0
    modello.eval()
    for f in flussi:
        token = torch.from_numpy(f.token).to(device)
        L = len(token)
        da_prevedere = np.zeros(L, dtype=bool)
        da_prevedere[:-1] = (f.token[1:] != PROG) & (f.token[1:] < V)
        da_prevedere = torch.from_numpy(da_prevedere).to(device)
        contate = 0
        for inizio, fine in finestre_di_lettura(L):
            logit = modello(token[None, inizio:fine])[0]
            pos = torch.arange(max(inizio, contate), min(fine, L - 1), device=device)
            if len(pos) == 0:
                continue
            contate = int(pos[-1]) + 1
            ok = da_prevedere[pos]
            if ok.any():
                somma += F.cross_entropy(logit[0, pos - inizio][ok].float(), token[pos + 1][ok], reduction="sum").item()
                n += int(ok.sum())
    modello.train()
    return somma / n


def finestre_di_lettura(L: int):
    """[inizio, fine) delle finestre con cui si legge un flusso lungo L: la prima intera,
    poi a passi di mezza finestra, l'ultima che finisce sulla fine del flusso."""
    if L <= FINESTRA:
        yield 0, L
        return
    yield 0, FINESTRA
    for fine in range(FINESTRA + FINESTRA // 2, L + FINESTRA // 2, FINESTRA // 2):
        fine = min(fine, L)
        yield fine - FINESTRA, fine


# --- il ciclo di addestramento -------------------------------------------------------------

def tasso(passo: int, totale: int, riscaldamento: int, lr: float) -> float:
    """Riscaldamento lineare, poi un coseno fino a lr/10."""
    if passo < riscaldamento:
        return lr * (passo + 1) / riscaldamento
    frazione = (passo - riscaldamento) / max(1, totale - riscaldamento)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frazione)))


def allena(numero_split: int, out, epoche: int, lr: float, epoche_riscaldamento: float, ogni: int,
           da: str | None, con_self_forcing: bool, device: str) -> None:
    """Una fase di addestramento (vedi in cima). Fase 1: salva `best_nt.pt`, il migliore
    sul token successivo in validazione. Fase 2: salva i pesi ogni `ogni` epoche."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    out.mkdir(parents=True, exist_ok=True)
    train, val = dividi_validazione(split(numero_split)[0])
    video_train, video_val = carica_videos(train, numero_split), carica_videos(val, numero_split)
    flussi_val = [costruisci(v) for v in video_val]
    codebook = torch.from_numpy(np.load(RISULTATI / f"split{numero_split}" / "token" / "codebook.npy").astype(np.float32))
    modello = ModelloAR(codebook).to(device)
    if da is not None:
        modello.load_state_dict(torch.load(da, map_location="cpu", weights_only=False)["model"])
    opt = torch.optim.AdamW(modello.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    # Quanti passi di ottimizzazione fa un'epoca: si contano sui flussi senza aumenti
    n_finestre = len(finestre_dell_epoca([costruisci(v) for v in video_train], FINESTRA, np.random.RandomState(0)))
    passi_epoca = math.ceil(n_finestre / BATCH)
    totale, riscaldamento, passo = passi_epoca * epoche, max(1, int(epoche_riscaldamento * passi_epoca)), 0
    print(f"split {numero_split}: {len(train)} video di training, {len(val)} di validazione, "
          f"{sum(p.numel() for p in modello.parameters()) / 1e6:.2f} M parametri, {passi_epoca} passi per epoca, "
          f"device {device}", flush=True)
    migliore, t0 = float("inf"), time.time()

    for epoca in range(1, epoche + 1):
        # --- gli aumenti dell'epoca: prima la velocità di tutti i video, poi i confini
        videos = [cambia_velocita(v, rng) if rng.rand() < 0.5 else v for v in video_train]
        flussi = [costruisci(v, confini_rumorosi(v.azioni, rng) if rng.rand() < 0.5 else None) for v in videos]
        modello.train()
        finestre = finestre_dell_epoca(flussi, FINESTRA, rng)
        for k in range(passi_epoca):
            gruppo = finestre[k * BATCH:(k + 1) * BATCH]
            if not gruppo:
                break
            quali = [i for i, _ in gruppo]
            b = {n: x.to(device) for n, x in batch([flussi[i] for i in quali], [s for _, s in gruppo], FINESTRA).items()}
            for g in opt.param_groups:
                g["lr"] = tasso(passo, totale, riscaldamento, lr)
            loss, parti = perdita(modello, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if con_self_forcing and passo % SF_OGNI == 0:
                parti["self-forcing"] = self_forcing(modello, [flussi[i] for i in quali], [videos[i] for i in quali],
                                                     rng, device)
            torch.nn.utils.clip_grad_norm_(modello.parameters(), 1.0)
            opt.step()
            passo += 1

        if epoca % ogni == 0 or epoca == epoche:
            minuti = (time.time() - t0) / 60
            riga = f"epoca {epoca:4d}  " + "  ".join(f"{n} {x:.3f}" for n, x in parti.items())
            if con_self_forcing:
                torch.save({"model": modello.state_dict(), "epoca": epoca}, out / f"epoca{epoca}.pt")
            else:
                errore = errore_token_successivo(modello, flussi_val, device)
                riga += f"  | validazione: token successivo {errore:.3f}"
                if errore < migliore:
                    migliore = errore
                    torch.save({"model": modello.state_dict(), "epoca": epoca}, out / "best_nt.pt")
                    riga += " *"
            print(riga + f"  | {minuti:.0f} min", flush=True)


def media_dei_pesi(checkpoint: list, uscita) -> None:
    """La media dei pesi di più checkpoint (vedi in cima): si mediano i tensori a virgola
    mobile, il resto viene dall'ultimo."""
    ck = [torch.load(c, map_location="cpu", weights_only=False) for c in checkpoint]
    media = {k: torch.stack([c["model"][k].float() for c in ck]).mean(0).to(v.dtype) if v.is_floating_point() else v
             for k, v in ck[-1]["model"].items()}
    torch.save({"model": media, "epoca": f"media delle epoche {[c['epoca'] for c in ck]}"}, uscita)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cartella = RISULTATI / f"split{a.split}"
    base, sf = cartella / "ar_base", cartella / "ar_sf"
    if not (base / "best_nt.pt").exists():
        print("fase 1: addestramento", flush=True)
        allena(a.split, base, epoche=2000, lr=3e-4, epoche_riscaldamento=20, ogni=100, da=None,
               con_self_forcing=False, device=a.device)
    if not (sf / f"epoca{EPOCHE_MEDIA[-1]}.pt").exists():
        print("fase 2: self-forcing", flush=True)
        allena(a.split, sf, epoche=100, lr=1e-4, epoche_riscaldamento=1, ogni=10, da=base / "best_nt.pt",
               con_self_forcing=True, device=a.device)
    media_dei_pesi([sf / f"epoca{e}.pt" for e in EPOCHE_MEDIA], sf / "modello_finale.pt")
    print(f"fase 3: media dei pesi delle epoche {list(EPOCHE_MEDIA)} -> {sf / 'modello_finale.pt'}")


if __name__ == "__main__":
    main()
