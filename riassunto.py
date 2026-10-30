"""
La tabella finale: la media sui 4 split di GTEA, come la riporta ProTAS.

    python riassunto.py            # dopo esegui_split.sh su tutti e 4 gli split

Per ogni variante (senza e con calibrazione): il progresso sul test (MSE, E@C, CMD), la
segmentazione con la post-elaborazione (solo il segmentatore, e il sistema completo con
il progresso e il grafo), e il criterio con cui si sceglie fra le varianti, la somma
MSE + E@C + CMD sul training (cross-fitting). Il test non decide niente.
"""
import json

import numpy as np

from dati import RISULTATI

SEG = ("acc", "edit", "f1@10", "f1@25", "f1@50")


def main():
    print(f"{'':36s} {'MSE':>6s} {'E@C':>6s} {'CMD':>6s} | {'Acc':>5s} {'Edit':>5s} {'F1@10':>5s} {'F1@25':>5s} "
          f"{'F1@50':>5s} | criterio sul training")
    for nome in ("risultati", "risultati_cal"):
        r = [json.loads((RISULTATI / f"split{s}" / f"{nome}.json").read_text()) for s in (1, 2, 3, 4)]
        m = lambda f: np.mean([f(x) for x in r])
        prog = [m(lambda x: x["progresso"][k]) for k in ("mse", "e@c", "cmd")]
        criterio = m(lambda x: x["progresso_train"]["mse"] + x["progresso_train"]["e@c"] + x["progresso_train"]["cmd"])
        for riga in ("CAS + pp", "CAS + AR + grafo + pp"):
            seg = [m(lambda x: x[riga][k]) for k in SEG]
            testa = " ".join(f"{x:6.3f}" for x in prog) if riga != "CAS + pp" else " " * 20
            coda = f"{criterio:.3f}" if riga != "CAS + pp" else ""
            print(f"{nome.replace('risultati', 'finale') + ': ' + riga:36s} {testa} | "
                  + " ".join(f"{x:5.1f}" for x in seg) + f" | {coda}")


if __name__ == "__main__":
    main()
