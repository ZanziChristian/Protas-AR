# ProTAS-AR: stima del progresso delle azioni con un modello autoregressivo

Project work del corso di Generative Models, Università di Firenze (Christian Zanzi, 2026).

## Sinossi

**La domanda.** Un modello generativo addestrato solo a prevedere il token successivo sa
stimare il progresso di un'azione? Se sa come continua il video, deve sapere anche quando
l'azione finirà.

**Il problema.** Segmentazione online delle azioni e stima del loro progresso in video
procedurali egocentrici (GTEA, persone che cucinano). A ogni istante, guardando solo il
passato, il sistema deve dire quale azione è in corso e a che punto è.

**Il metodo.** Si parte dalla struttura di ProTAS (Shen & Elhamifar, CVPR 2024):
segmentatore causale, modulo del progresso, fusione e grafo del compito. Il modulo del
progresso, che in ProTAS è una rete di regressione, è sostituito da un modello
autoregressivo:

1. il video si divide in passi da 5 frame (0.33 s); la feature I3D di ogni passo diventa
   16 token con la quantizzazione a prodotto (16 pezzi, 64 codici per pezzo, k-means sul
   solo training);
2. per ogni passo un piccolo transformer causale (4 blocchi, d = 64, RoPE, 0.25 M
   parametri) legge i 16 codici, l'azione in corso (data dal segmentatore), un token di
   query `[PROG]` e, a fine azione, `[EoA]`. Impara a prevedere il token successivo;
3. per stimare il progresso il modello **immagina il futuro**: continua il flusso da solo
   e a ogni confine fra due passi immaginati legge h_j = P(`[EoA]`). Da qui la
   distribuzione del tempo che manca e la stima del progresso:

       P(L_pred = j) = h_j · ∏_{i<j} (1 − h_i)
       p̂ = mediana di  L_past / (L_past + L_pred)

   Nessuna parte del modello è addestrata a produrre questo numero: viene dalla capacità
   di prevedere quando arriva `[EoA]`.

Il riconoscimento dell'azione è la media di due segmentatori causali (la CAS di ProTAS e
OnlineTAS, riscritto dal paper) con la post-elaborazione online di OnlineTAS. Ogni scelta
(varianti, calibrazione, tokenizzatore) è fatta con un cross-fitting del training: il test
non decide niente.

**I risultati** (GTEA, media dei 4 split, stesse feature, split e metriche di ProTAS; i
numeri di ProTAS sono quelli del paper).

Segmentazione online:

| | Acc | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| ProTAS completo | **74.3** | 69.2 | 77.1 | 74.1 | 59.7 |
| ProTAS-AR (sistema completo) | 73.7 | **80.3** | **85.1** | **82.4** | **65.4** |

Progresso (più basso è meglio):

| | MSE | E@C | CMD |
|---|---|---|---|
| ProTAS | **0.111** | 0.525 | **0.203** |
| Frame-counting (stesso inizio stimato) | 0.125 | 0.561 | 0.285 |
| ProTAS-AR, rollout | 0.132 | **0.410** | 0.255 |
| ProTAS-AR, solo le azioni con la classe giusta (71%) | 0.075 | 0.171 | 0.124 |

- **Segmentazione:** ProTAS-AR supera ProTAS su Edit (+11.1) e su tutti gli F1, e perde
  0.6 punti di accuratezza.
- **Progresso:** il rollout batte ProTAS sull'errore al completamento (E@C) e il
  frame-counting su E@C e CMD. Resta sotto ProTAS su MSE e CMD.
- **Lo scarto viene soprattutto dal segmentatore.** Il progresso sta nel canale
  dell'azione riconosciuta: quando la classe è sbagliata, la metrica (letta nel canale
  della classe vera) conta un errore pieno. Nel 71% delle azioni la classe all'ultimo
  frame è giusta, e lì la stima è migliore di ProTAS su tutte e tre le metriche.
- **Il modello generativo ha imparato la fine delle azioni.** Sul test indovina il
  46–48% dei codici successivi e il 13–14% di quelli che cambiano da un passo all'altro.
  P(`[EoA]`) distingue le fini vere dagli altri confini con un'AUC di 0.83–0.90, contro
  0.71–0.77 del solo tempo trascorso.

**La risposta.** Sì: un modello addestrato a prevedere il token successivo stima il
progresso e sa quando un'azione sta per finire meglio del tempo trascorso. Batte ProTAS
sull'errore al completamento e resta sotto su MSE e CMD, soprattutto a causa delle azioni
in cui il segmentatore sbaglia classe.

## I dati

Le feature I3D, le annotazioni e gli split di GTEA sono quelli di MS-TCN (Abu Farha & Gall,
CVPR 2019), gli stessi usati da ProTAS. Si scaricano da Zenodo:

https://zenodo.org/records/3625992 (`data.zip`, circa 30 GB: contiene anche 50Salads e
Breakfast, serve solo la cartella `gtea`)

La cartella `gtea` va messa in `data/gtea/` dentro questa repository:

    data/gtea/
      features/<video>.npy        feature I3D [2048, T], una colonna per frame (15 fps)
      groundTruth/<video>.txt     un'etichetta per frame
      mapping.txt                 "<id> <nome>" per ogni classe
      splits/{train,test}.splitN.bundle

## Come si esegue

```bash
pip install -r requirements.txt
mkdir -p risultati
./esegui_split.sh 1 0 > risultati/split1.log 2>&1     # tutta la catena sullo split 1, sulla GPU 0
```

Su una RTX A2000 uno split richiede circa 3 ore (segmentatori ~1 h, addestramento ~1 h 15,
stime ~45 min). Ogni passo scrive in `risultati/` e si salta se è già fatto. Dopo i
quattro split:

```bash
python riassunto.py       # la tabella finale, media sui 4 split
python diagnosi.py --split 1   # token successivo e AUC di [EoA] sul test
```

## I file

La catena, nell'ordine in cui `esegui_split.sh` la esegue:

| file | cosa fa |
|---|---|
| `dati.py` | legge GTEA, divide i video in passi da 5 frame, calcola il progresso vero, definisce split e cross-fitting |
| `tokenizzatore.py` | passo 1: la feature di ogni passo diventa 16 token (quantizzazione a prodotto, k-means sul solo training) |
| `segmenta.py` | passo 2: addestra i due segmentatori online, ne fa la media e la post-elaborazione; sul training con il cross-fitting |
| `cas.py` | la CAS di ProTAS: MS-TCN causale a 4 stadi |
| `onlinetas.py` | OnlineTAS: GRU, attenzione con memoria e TCN causale |
| `flusso.py` | il vocabolario e il flusso di token che il transformer legge, con gli aumenti dei dati |
| `modello.py` | il transformer autoregressivo (RoPE, KV-cache), scritto per esteso |
| `addestra.py` | passo 3: addestramento sul token successivo e su `[EoA]`, poi self-forcing e media dei pesi |
| `rollout.py` | il rollout atteso: dalle probabilità di `[EoA]` sui futuri immaginati alla stima del progresso |
| `stima.py` | passo 4: la stima del progresso a ogni passo, sul test e (cross-fitting) sul training |
| `fusione.py` | passo 5: il progresso dentro ProTAS (fusione residua e grafo del compito) e le metriche finali |
| `metriche.py` | le metriche di ProTAS: Acc, Edit, F1@k per la segmentazione; MSE, E@C, CMD per il progresso |
| `riassunto.py` | la tabella finale, media sui 4 split |
| `diagnosi.py` | il modello prevede il token successivo meglio che copiando? P(`[EoA]`) sale dove l'azione finisce? |
| `esegui_split.sh` | tutta la catena su uno split |

Ogni file spiega nei commenti cosa fa e perché.

## Da sapere

- **Il rollout si fa solo dove le annotazioni dicono che c'è un'azione.** In `stima.py` i
  passi su cui si stima sono quelli dentro un'azione vera; sul background vero la stima
  resta 0. Per misurare il progresso va bene, perché sul background non è definito. A valle
  però la stima entra anche nella fusione, cioè nella segmentazione, e lì è
  un'informazione che un sistema reale non avrebbe. Per toglierla basta stimare a ogni
  passo (`da_stimare` in `stima.py`), e i risultati cambiano.
- **Due addestramenti non sono mai identici sulla GPU.** Rieseguendo la catena i numeri
  cadono nel rumore fra due addestramenti, non coincidono bit per bit.
- **Il confronto con ProTAS** usa i numeri del paper, non una nostra esecuzione. Il
  sistema ha circa 1.83 M parametri (0.80 M + 0.78 M per i segmentatori, 0.25 M per il
  modello autoregressivo) contro circa 0.9 M di ProTAS.

## Riferimenti

- Y. Shen, E. Elhamifar. *Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos.* CVPR 2024.
- Q. Zhong, G. Ding, A. Yao. *OnlineTAS: An Online Baseline for Temporal Action Segmentation.* NeurIPS 2024.
- Y. Abu Farha, J. Gall. *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.* CVPR 2019.
- A. Fathi, X. Ren, J. M. Rehg. *Learning to Recognize Objects in Egocentric Activities.* CVPR 2011.
- H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 2011.
