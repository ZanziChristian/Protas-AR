"""
Il transformer autoregressivo, scritto per esteso (niente nn.TransformerDecoder).

È un GPT-2 piccolo (Radford et al., 2019): 4 blocchi, d = 64, 4 teste, 0.25 M parametri.
Legge il flusso di token (flusso.py) e a ogni posizione produce tre uscite:

    token     [B, L]          id dei token
    x         [B, L, 64]      il vettore di ogni token (+ il suo posto nel passo)
    4 blocchi: x = x + Attenzione(LN(x));  x = x + MLP(LN(x))      (pre-LayerNorm)
    logit     [B, L, 1041]    testa visiva: quale token viene dopo
    progresso [B, L]          testa del progresso: un numero in (0, 1)
    azione    [B, L, 11]      testa dell'azione: quale azione è in corso

Un solo tronco per tre teste: lo stesso vettore deve dire "cosa viene dopo", "a che
punto siamo" e "cosa sto facendo".

I codici passano dal codebook
-----------------------------
Il codebook del tokenizzatore dà a ogni codice il suo vettore da 128 (il centro del
k-means), e codici con vettori vicini descrivono scene simili. Il modello ne approfitta
dai due lati: un codice ENTRA come il suo vettore proiettato a 64, e in USCITA la testa
produce un vettore da 128 il cui prodotto scalare con ogni codice è il logit di quel
codice. Così un codice quasi giusto costa quasi come quello giusto. Il codebook resta
fisso (normalizzato a lunghezza 1); i simboli speciali e le azioni hanno vettori appresi.

Il posto nel passo
------------------
Il codice nel posto g di un passo è il pezzo g della feature, e somiglia soprattutto al
codice nello STESSO posto del passo prima, 19 token indietro (20 se in mezzo c'è [EoA]).
Un vettore appreso per il posto (0, ..., 15) dice al modello quale pezzo sta guardando.

Maschera causale
----------------
La posizione i guarda solo le posizioni <= i: è ciò che rende il modello autoregressivo
e la stima del progresso online (al [PROG] del passo t il modello non ha visto niente
dopo t).

Posizioni rotatorie, RoPE (Su et al., 2021)
-------------------------------------------
La posizione assoluta qui non significa niente: le finestre di addestramento partono da
punti a caso del video. Conta la distanza: il token più utile per prevedere un codice
sta 19 posizioni indietro. La RoPE ruota query e chiavi di ogni testa, a coppie di
dimensioni, di un angolo proporzionale alla posizione: il prodotto fra una query in i e
una chiave in j ruotate dipende allora solo da i − j, e l'attenzione vede "quanto
indietro", mai "dove". Scelta sul training (validazione e cross-fitting) contro la
sinusoide sommata all'ingresso del primo modello.

KV-cache
--------
Nella generazione (rollout.py) si aggiunge un token alla volta: le chiavi e i valori
dei token già visti si tengono in memoria, e il token nuovo calcola solo la propria riga
dell'attenzione invece di rifare tutto il prefisso.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flusso import DIM_VOCABOLARIO, N_AZIONI, V

D_MODELLO, N_BLOCCHI, N_TESTE, DROPOUT = 64, 4, 4, 0.1
# Righe della tabella del posto nel passo: ne servono 16, ma il modello è nato con i token
# EVATok (fino a 512 per passo) e la tabella è rimasta così. Le righe in più non si usano;
# restano per poter caricare i pesi addestrati.
MAX_POSTO = 512


def angoli_rope(pos: torch.Tensor, d_testa: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, L] posizioni -> (cos, sin) [B, 1, L, d_testa/2]: un angolo per coppia di
    dimensioni. Frequenze geometriche: la prima coppia gira di un radiante a posizione
    (distanze brevi), l'ultima quasi non si muove (distanze lunghe)."""
    i = torch.arange(0, d_testa, 2, dtype=torch.float32, device=pos.device)
    freq = torch.exp(-math.log(10000.0) * i / d_testa)
    angoli = pos[..., None].float() * freq
    return angoli.cos()[:, None], angoli.sin()[:, None]


def ruota(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """[B, H, L, d_testa]: ogni coppia (a, b) di dimensioni ruotata del suo angolo θ,
    (a cosθ − b sinθ, a sinθ + b cosθ)."""
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


def posti_nel_passo(token: torch.Tensor, partenza: torch.Tensor) -> torch.Tensor:
    """[B, L] -> [B, L] il posto di ogni codice nel suo passo: quanti codici consecutivi
    lo precedono. Ai simboli vale 0 (e non si usa).

    `partenza` [B] dice quanti codici del passo in corso precedevano il primo token: zero
    senza cache; nella generazione i token arrivano uno alla volta e il conto riprende.
    Il conto senza cicli: `conta` sono i codici visti fino a ogni posizione; tolto il
    valore che aveva all'ultimo simbolo restano quelli venuti dopo, cioè il posto."""
    codice = token < V
    conta = torch.cumsum(codice.long(), dim=1)
    all_ultimo_simbolo = torch.cummax(torch.where(codice, torch.zeros_like(conta), conta), dim=1).values
    dopo_un_simbolo = torch.cummax((~codice).long(), dim=1).values.bool()
    posto = conta - all_ultimo_simbolo - 1 + torch.where(dopo_un_simbolo, 0, partenza[:, None])
    return torch.where(codice, posto, torch.zeros_like(posto))


class Attenzione(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_teste, self.d_testa = N_TESTE, D_MODELLO // N_TESTE
        self.qkv = nn.Linear(D_MODELLO, 3 * D_MODELLO)     # query, chiavi e valori in una proiezione
        self.proj = nn.Linear(D_MODELLO, D_MODELLO)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor, rot, cache: dict | None = None) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        # [B, L, d] -> [B, teste, L, d_testa]: ogni testa lavora su una fetta di d
        q, k, v = (t.view(B, L, self.n_teste, self.d_testa).transpose(1, 2) for t in (q, k, v))
        # RoPE su query e chiavi, PRIMA della cache: ogni chiave resta ruotata alla sua posizione
        q, k = ruota(q, *rot), ruota(k, *rot)

        if cache is not None and cache["capienza"]:
            # Spazio allocato una volta: si scrivono solo le righe nuove. Concatenare
            # ricopierebbe tutta la cache a ogni token generato.
            if cache["k"] is None:
                forma = (B, self.n_teste, cache["capienza"], self.d_testa)
                cache["k"], cache["v"], cache["n"] = k.new_zeros(forma), v.new_zeros(forma), 0
            n = cache["n"]
            cache["k"][:, :, n:n + L], cache["v"][:, :, n:n + L] = k, v
            cache["n"] = n + L
            k, v = cache["k"][:, :, :n + L], cache["v"][:, :, :n + L]
        elif cache is not None:
            if cache["k"] is not None:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        L_tot = k.shape[2]

        punteggi = q @ k.transpose(2, 3) / math.sqrt(self.d_testa)       # [B, H, L, L_tot]
        # le L query sono le ultime L delle L_tot posizioni: ognuna vede le chiavi fino a sé
        pos_q = torch.arange(L_tot - L, L_tot, device=x.device)
        pos_k = torch.arange(L_tot, device=x.device)
        punteggi = punteggi.masked_fill(pos_k[None, :] > pos_q[:, None], float("-inf"))
        pesi = self.dropout(torch.softmax(punteggi, dim=-1))
        return self.proj((pesi @ v).transpose(1, 2).reshape(B, L, d))


class Blocco(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODELLO)
        self.attn = Attenzione()
        self.ln2 = nn.LayerNorm(D_MODELLO)
        self.mlp = nn.Sequential(nn.Linear(D_MODELLO, 4 * D_MODELLO), nn.GELU(), nn.Linear(4 * D_MODELLO, D_MODELLO))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, rot, cache=None):
        x = x + self.dropout(self.attn(self.ln1(x), rot, cache))
        return x + self.dropout(self.mlp(self.ln2(x)))


class ModelloAR(nn.Module):
    def __init__(self, codebook: torch.Tensor):
        super().__init__()
        # Il codebook a lunghezza 1. È normalizzato due volte, come nella versione che ha
        # addestrato il modello finale: la seconda divisione cambia solo l'ultima cifra
        # dei numeri, ma serve a riprodurre esattamente gli stessi risultati.
        cb = F.normalize(codebook.float(), dim=1)
        self.register_buffer("codebook", cb / cb.norm(dim=1, keepdim=True))       # [1024, 128]
        dim = codebook.shape[1]
        n_simboli = DIM_VOCABOLARIO - V
        # I nomi (in_codice, embed, ...) e l'ordine in cui i pezzi si creano sono quelli
        # dei pesi addestrati: l'ordine decide anche i numeri casuali dell'inizializzazione.
        self.in_codice = nn.Linear(dim, D_MODELLO)              # codice -> vettore del modello
        self.embed = nn.Embedding(n_simboli, D_MODELLO)         # simboli e azioni
        self.out_codice = nn.Linear(D_MODELLO, dim)             # vettore del modello -> spazio dei codici
        self.head_speciali = nn.Linear(D_MODELLO, n_simboli)    # logit dei simboli e delle azioni
        self.embed_posto = nn.Embedding(MAX_POSTO, D_MODELLO)
        self.dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Blocco() for _ in range(N_BLOCCHI)])
        self.ln_f = nn.LayerNorm(D_MODELLO)
        self.head_progress = nn.Linear(D_MODELLO, 1)
        self.head_action = nn.Linear(D_MODELLO, N_AZIONI)
        self.apply(self._inizializza)

    @staticmethod
    def _inizializza(m):
        """Come GPT-2: pesi piccoli, così all'inizio i logit sono quasi uniformi."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def nuova_cache(self, capienza: int | None = None) -> list[dict]:
        """Una cache vuota per blocco. Con `capienza` lo spazio per chiavi e valori si
        alloca una volta per quel numero di posizioni. La prima tiene anche il posto
        raggiunto nel passo in corso."""
        cache = [{"k": None, "v": None, "capienza": capienza} for _ in self.blocks]
        cache[0]["posto"] = None
        return cache

    def forward(self, token: torch.Tensor, cache: list | None = None, inizio: int = 0, solo_ultima: bool = False):
        """token [B, L] -> (logit [B, L, 1041], progresso [B, L], logit dell'azione [B, L, 11]).

        Con la cache `token` contiene solo i token nuovi e `inizio` è la posizione del
        primo (quanti token la cache ha già). Con `solo_ultima` le teste si calcolano
        solo sull'ultima posizione: serve a chi riempie la cache con un prefisso lungo e
        poi genera."""
        B, L = token.shape
        codice = token < V
        # --- il vettore di ogni token: dal codebook per i codici, appreso per i simboli
        x = torch.where(codice[..., None], self.in_codice(self.codebook[token.clamp(max=V - 1)]),
                        self.embed((token - V).clamp(min=0)))
        # --- più il posto nel passo, solo per i codici
        partenza = torch.zeros(B, dtype=torch.long, device=token.device)
        if cache is not None and cache[0]["posto"] is not None:
            partenza = cache[0]["posto"]
        posto = posti_nel_passo(token, partenza)
        x = x + self.embed_posto(posto.clamp(max=MAX_POSTO - 1)) * codice[..., None]
        if cache is not None:            # il passo in corso ha un codice in più, oppure è chiuso
            cache[0]["posto"] = torch.where(codice[:, -1], posto[:, -1] + 1, 0)
        x = self.dropout(x)
        # --- i blocchi, con la RoPE sull'indice del token
        pos = torch.arange(inizio, inizio + L, device=token.device)[None].expand(B, L)
        rot = angoli_rope(pos, D_MODELLO // N_TESTE)
        for i, blocco in enumerate(self.blocks):
            x = blocco(x, rot, None if cache is None else cache[i])
        if solo_ultima:
            x = x[:, -1:]
        x = self.ln_f(x)
        # --- le tre teste. Logit di un codice = somiglianza fra il vettore predetto e il suo
        logit = torch.cat([self.out_codice(x) @ self.codebook.T, self.head_speciali(x)], dim=-1)
        return logit, torch.sigmoid(self.head_progress(x))[..., 0], self.head_action(x)


def carica(percorso, device: str = "cpu") -> ModelloAR:
    """Un modello salvato da addestra.py."""
    pesi = torch.load(percorso, map_location="cpu", weights_only=False)["model"]
    m = ModelloAR(torch.zeros_like(pesi["codebook"]))
    m.load_state_dict(pesi)
    return m.to(device).eval()
