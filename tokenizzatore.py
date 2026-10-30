"""
Passo 1 della catena: dal video ai token.

    python tokenizzatore.py --split 1
    -> risultati/passi/<video>.npz             feature, etichetta e confini di ogni passo (uguali per tutti gli split)
    -> risultati/split1/token/<video>.npy      [T, 16] i 16 token di ogni passo
    -> risultati/split1/token/codebook.npy     [1024, 128] il vettore di ogni token

Il modello autoregressivo legge token DISCRETI, come un modello di linguaggio legge
parole. La feature di un passo è invece un vettore di 2048 numeri reali: qui diventa 16
token con la quantizzazione a prodotto (Jégou et al., 2011).

La quantizzazione a prodotto (PQ-16, 64 codici per pezzo)
---------------------------------------------------------
Un solo dizionario per vettori da 2048 numeri dovrebbe avere milioni di parole per
descriverli bene, e ognuna si vedrebbe una volta sola. Si taglia invece il vettore in 16
pezzi da 128 numeri, e ogni pezzo ha il suo dizionario di 64 "codici", imparato con il
k-means: il codice di un pezzo è il centro del k-means più vicino. Un passo diventa così
16 token, uno per pezzo, e le combinazioni possibili sono 64^16 = 2^96.

Gli id: il codice k del pezzo g diventa il token g * 64 + k, quindi il vocabolario dei
codici ha 16 * 64 = 1024 token e il token nel posto g del passo appartiene sempre al
pezzo g. Il codebook salvato ha una riga per token: il centro corrispondente (128 numeri).
Il modello legge e scrive i token attraverso questi vettori (vedi modello.py).

Prima di tagliare, ogni dimensione si porta a media 0 e deviazione 1 (con le statistiche
del training): senza, i pezzi con i numeri più grandi dominerebbero il k-means.

Perché uno per split
--------------------
Il k-means impara i codici guardando i dati: se guardasse anche i video di test, il
tokenizer saprebbe già qualcosa di loro. I codici si imparano quindi solo sui video di
training dello split, e ogni split ha i suoi token.

Perché 16 pezzi da 64 codici
----------------------------
Il modello impara a prevedere il codice successivo, e il problema dei codici è il
tremolio: due passi quasi uguali finiscono spesso su centri diversi, solo perché stanno
vicino al confine fra due centri. Quel cambio non dice niente sul video e non si può
prevedere. Con pochi codici per pezzo ogni centro copre una zona più larga: il codice
cambia meno spesso senza motivo, e ogni codice si vede più volte in addestramento (uno
split ha circa 4.700 passi di training: ~70 per codice con 64 codici, ~18 con 256). Più
pezzi recuperano l'informazione che i dizionari piccoli perdono: 16 × 6 bit = 96 bit per
passo, contro gli 8 × 8 = 64 di 8 pezzi da 256 codici.

Scelto sul training (cross-fitting), contro 8 pezzi da 256 codici: sui video tenuti fuori
i codici che cambiano fra un passo e il successivo scendono dal 48% al 43%, il modello
indovina il 13% dei codici che cambiano invece dell'11%, P([EoA]) separa meglio i confini
veri (AUC 0.864 contro 0.853), e la stima del progresso migliora (somma MSE + E@C + CMD
sul training 0.709 contro 0.726).
"""
import argparse

import numpy as np
import torch

from dati import RISULTATI, passi, split

PEZZI, CODICI = 16, 64          # token per passo, codici per pezzo (flusso.py ne ricava il vocabolario)


def prepara_passi() -> None:
    """Feature, etichetta e confini di ogni passo, per tutti i video (vedi dati.passi).
    Le feature si salvano a mezza precisione: metà dello spazio, e sono quelle che il
    k-means legge."""
    cartella = RISULTATI / "passi"
    cartella.mkdir(parents=True, exist_ok=True)
    tutti = sorted(set(sum(split(1), [])))
    for v in tutti:
        p = passi(v)
        np.savez(cartella / f"{v}.npz", feature=p["feature"].astype(np.float16),
                 etichette=p["etichette"], inizi=p["inizi"], fini=p["fini"])
    print(f"{len(tutti)} video divisi in passi da 5 frame -> {cartella}")


def kmeans(X: torch.Tensor, k: int, seme: int, iterazioni: int = 25) -> torch.Tensor:
    """I k centri del k-means di Lloyd: [N, D] -> [k, D].

    Si parte da k punti a caso; poi, ripetendo: ogni punto va al centro più vicino, e
    ogni centro si sposta nella media dei suoi punti. Un centro rimasto senza punti
    riparte da un punto a caso."""
    g = torch.Generator().manual_seed(seme)
    centri = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(iterazioni):
        vicino = piu_vicino(X, centri)
        somma = torch.zeros_like(centri).index_add_(0, vicino, X)
        conta = torch.zeros(k, device=X.device).index_add_(0, vicino, torch.ones_like(vicino, dtype=torch.float32))
        vuoti = conta == 0
        centri = somma / conta.clamp(min=1)[:, None]
        if vuoti.any():
            centri[vuoti] = X[torch.randint(len(X), (int(vuoti.sum()),), generator=g).to(X.device)]
    return centri


def piu_vicino(x: torch.Tensor, centri: torch.Tensor) -> torch.Tensor:
    """L'indice del centro più vicino a ogni riga di x. La distanza al quadrato è
    |x|² − 2 x·c + |c|²; |x|² è uguale per tutti i centri e non serve."""
    return ((centri * centri).sum(1)[None] - 2 * x @ centri.T).argmin(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if not (RISULTATI / "passi").exists():
        prepara_passi()
    train, test = split(a.split)
    feature = {v: np.load(RISULTATI / "passi" / f"{v}.npz")["feature"].astype(np.float32) for v in train + test}

    # standardizzazione con le statistiche del solo training
    X = np.concatenate([feature[v] for v in train])
    media, scala = X.mean(0), X.std(0) + 1e-6
    A = torch.from_numpy((X - media) / scala).to(a.device)

    # un k-means per pezzo: [16, 64 codici, 128 numeri]
    D = A.shape[1] // PEZZI
    codebook = torch.stack([kmeans(A[:, g * D:(g + 1) * D], CODICI, seme=g) for g in range(PEZZI)])

    out = RISULTATI / f"split{a.split}" / "token"
    out.mkdir(parents=True, exist_ok=True)
    for v in train + test:
        Z = torch.from_numpy((feature[v] - media) / scala).to(a.device)
        codici = torch.stack([piu_vicino(Z[:, g * D:(g + 1) * D], codebook[g]) for g in range(PEZZI)], dim=1)
        token = codici + torch.arange(PEZZI, device=codici.device)[None] * CODICI      # pezzo g -> id g*64 + k
        np.save(out / f"{v}.npy", token.cpu().numpy().astype(np.int64))
    np.save(out / "codebook.npy", codebook.reshape(PEZZI * CODICI, D).cpu().numpy())
    print(f"split {a.split}: k-means su {len(X)} passi di {len(train)} video di training -> {out}")


if __name__ == "__main__":
    main()
