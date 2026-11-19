# Caso medio em Python, sem o binario Bend: as mesmas 400 execucoes de caso_medio.py
# (n = 2^14 pontos, k = 8) e a curva do custo J, com numpy.
# Mesma aritmetica F32 e mesma ordem de somas (arvore) de referencia.py, entao os
# resultados sao identicos; o script confere isso contra referencia.py e contra os CSVs do Bend.
import csv, os, statistics, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.join(os.path.dirname(AQUI), "dados")
sys.path.insert(0, AQUI)
import referencia

F = np.float32
M = np.uint64(2**32)

def rnd(h):
    a = (h * np.uint64(2654435761)) % M
    b = a ^ (a >> np.uint64(15))
    z = (b * np.uint64(2246822519)) % M
    return (((z ^ (z >> np.uint64(13))) & np.uint64(65535)).astype(np.float64) / 65536.0).astype(F)

def pontos(n, modo, s):
    i = np.arange(n, dtype=np.uint64)
    off = np.uint64((s * 1000003) % 2**32)
    ux, uy = rnd((i * 2 + 11 + off) % M), rnd((i * 2 + 12 + off) % M)
    if modo == 1:
        return ux * F(100.0), uy * F(100.0)
    b, o2 = i % np.uint64(8), np.uint64((s * 7919) % 2**32)
    cx, cy = rnd((b * 7 + 1 + o2) % M) * F(100.0), rnd((b * 7 + 3 + o2) % M) * F(100.0)
    return cx + (ux - F(0.5)) * F(12.0), cy + (uy - F(0.5)) * F(12.0)

def arvore(v):
    # soma em arvore no eixo 0, a mesma ordem da recursao de referencia.somar (n potencia de 2)
    while v.shape[0] > 1:
        v = v[0::2] + v[1::2]
    return v[0]

def kmeans(d, k, modo, maxit, s=0, hist=None):
    n = 2**d
    x, y = pontos(n, modo, s)
    cs = np.stack(pontos(k, modo, s), 1)
    it = 0
    for _ in range(maxit):
        dx, dy = x[:, None] - cs[None, :, 0], y[:, None] - cs[None, :, 1]
        dd = dx * dx + dy * dy                      # n x k distancias ao quadrado
        if hist is not None:
            hist.append(float(arvore(dd.min(1))))   # custo J
        m = dd.argmin(1)[:, None] == np.arange(k)   # passo 1: centro mais perto
        sx, sy = arvore(np.where(m, x[:, None], F(0))), arvore(np.where(m, y[:, None], F(0)))
        c = m.sum(0)
        novo = cs.copy()                            # passo 2: media de cada grupo
        for j in range(k):
            if c[j]:
                novo[j] = (sx[j] / F(c[j]), sy[j] / F(c[j]))
        mov = F(0)
        for j in range(k):
            ex, ey = novo[j, 0] - cs[j, 0], novo[j, 1] - cs[j, 1]
            mov = F(ex * ex + ey * ey) + mov
        cs, it = novo, it + 1
        if mov < F(0.000001):
            break
    return it, cs

# 1) igual a referencia.py: iteracoes, centros e J, bit a bit
for d, k, modo, s in [(10, 8, 0, 0), (12, 8, 1, 0), (10, 8, 1, 3), (12, 8, 0, 7), (11, 5, 1, 42)]:
    h1, h2 = [], []
    (it1, c1), (it2, c2) = kmeans(d, k, modo, 200, s, h1), referencia.kmeans(d, k, modo, 200, s, h2)
    assert it1 == it2 and np.array_equal(c1, np.array(c2, dtype=F)) and h1 == h2, (d, k, modo, s)
print("igual a referencia.py em 5 casos")

# 2) 400 execucoes
bend = {(int(r["modo"]), int(r["semente"])): int(r["iteracoes"])
        for r in csv.DictReader(open(os.path.join(DADOS, "iteracoes.csv")))}
with open(os.path.join(DADOS, "iteracoes_python.csv"), "w") as f:
    f.write("modo,semente,iteracoes\n")
    for modo, nome in ((0, "com grupos"), (1, "uniformes")):
        its = []
        for s in range(1, 201):
            it, _ = kmeans(14, 8, modo, 1000, s)
            its.append(it); f.write(f"{modo},{s},{it}\n")
        iguais = sum(its[s - 1] == bend.get((modo, s)) for s in range(1, 201))
        print(f"{nome}: media {statistics.mean(its):.1f}, min {min(its)}, max {max(its)}, "
              f"iguais ao Bend: {iguais}/200", flush=True)

# 3) curva de J na execucao mediana entre as sementes 1..20 (dados uniformes), como caso_medio.py
lin = sorted((int(r["iteracoes"]), int(r["semente"])) for r in csv.DictReader(open(os.path.join(DADOS, "iteracoes_python.csv")))
             if r["modo"] == "1" and int(r["semente"]) <= 20)
it, s = lin[len(lin) // 2]
j = []; kmeans(14, 8, 1, 1000, s, j)
with open(os.path.join(DADOS, "custo_j_python.csv"), "w") as f:
    f.write("iteracao,J\n")
    for i, v in enumerate(j, 1): f.write(f"{i},{v}\n")
jb = [float(r["J"]) for r in csv.DictReader(open(os.path.join(DADOS, "custo_j.csv")))]
erro = max(abs(a - b) / b for a, b in zip(j, jb)) if len(j) == len(jb) else float("inf")
print(f"curva J: semente {s}, {it} iteracoes, J de {j[0]:.0f} para {j[-1]:.0f}, "
      f"sempre cai: {all(b <= a for a, b in zip(j, j[1:]))}, erro relativo ao Bend: {erro:.1e}")
