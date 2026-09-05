"""Executa o cliente N vezes e consolida as acuracias em uma tabela.

O servidor e quem calcula a acuracia do modelo federado (so ele tem X_test),
e ele apenas escreve o resultado no log. Entao este script:

  1. sobe o server.py como subprocesso, redirecionando o log para um arquivo;
  2. roda o client.py N vezes em sequencia, guardando a saida de cada execucao;
  3. le a acuracia de cada rodada no log do servidor e o veredito
     honesto/bizantino de cada cliente no log do cliente;
  4. agrega por numero de clientes bizantinos e imprime a tabela.

Uso:
    python run_experiment.py                # N=10, sobe o servidor sozinho
    python run_experiment.py -n 30
    python run_experiment.py --server-log caminho/server.log   # servidor ja no ar

Saidas em --outdir (default: resultados/): server.log, client_runNN.log,
acuracias.csv, tabela_ptbr.tsv, tabela_en.tsv.
"""

import argparse
import csv
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = "localhost", 50051

RE_SCENARIO = re.compile(r"Cenario")
RE_VERDICT = re.compile(
    r"Client\s+(?P<cid>\d+)\s+\(\s*(?P<behavior>\w+)\s*\):.*?"
    r"(?P<verdict>defined as a Byzantine|classified as honest)"
)
RE_ROUND = re.compile(r"All \d+ clients received")
RE_SCORE = re.compile(r"Final model score on test data: (?P<score>[0-9.]+)")
RE_HONEST_COUNT = re.compile(r"Model trained with (?P<n>\d+) honest client")


# ---------------------------------------------------------------- servidor


def port_is_open():
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def start_server(log_path):
    """Sobe o servidor em uma sessao propria (para poder matar o grupo inteiro).

    Recusa subir se a porta ja estiver ocupada: o gRPC usa SO_REUSEPORT, entao
    dois servidores conseguem escutar a mesma porta ao mesmo tempo e as
    requisicoes se dividem silenciosamente entre eles, corrompendo o
    experimento (cada servidor tem seu proprio contador de clientes por rodada).
    """
    if port_is_open():
        sys.exit(
            f"ERRO: ja existe algo escutando em {HOST}:{PORT}. Encerre o servidor "
            f"antigo, ou use --server-log para aproveitar o que ja esta no ar."
        )

    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=HERE,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"ERRO: servidor morreu ao iniciar. Veja {log_path}")
        if "Server started" in open(log_path, errors="replace").read():
            print(f"servidor no ar (pid {proc.pid})")
            return proc
        time.sleep(1)

    stop_server(proc)
    sys.exit(f"ERRO: servidor nao subiu em 180s. Veja {log_path}")


def stop_server(proc):
    """Mata o grupo de processos: o Ray deixa processos filhos para tras."""
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            continue
    for _ in range(10):
        if not port_is_open():
            break
        time.sleep(1)
    print("servidor encerrado")


# ----------------------------------------------------------------- parsing


def parse_client_log(text):
    """-> lista de cenarios, cada um com a lista de (behavior, foi_marcado_biz)."""
    scenarios = []
    for line in text.splitlines():
        if RE_SCENARIO.search(line):
            scenarios.append([])
        m = RE_VERDICT.search(line)
        if m and scenarios:
            scenarios[-1].append(
                (m["behavior"], m["verdict"].startswith("defined"))
            )
    return scenarios


def parse_server_log(text):
    """-> lista de rodadas na ordem em que ocorreram: (acuracia|None, n_honestos).

    A rodada em que nenhum cliente foi aceito nao tem linha de acuracia; ela
    entra como None para nao desalinhar as rodadas seguintes.
    """
    rounds = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not RE_ROUND.search(line):
            continue
        honest = RE_HONEST_COUNT.search(line)
        score = RE_SCORE.search(lines[i + 1]) if i + 1 < len(lines) else None
        rounds.append(
            (
                float(score["score"]) if score else None,
                int(honest["n"]) if honest else 0,
            )
        )
    return rounds


# --------------------------------------------------------------- execucao


def run_client(run_idx, outdir):
    log_path = os.path.join(outdir, f"client_run{run_idx:02d}.log")
    proc = subprocess.run(
        [sys.executable, "client.py"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    with open(log_path, "w") as f:
        f.write(output)
    if proc.returncode != 0:
        sys.exit(f"ERRO: execucao {run_idx} falhou (rc={proc.returncode}). Veja {log_path}")
    return parse_client_log(output)


# --------------------------------------------------------------- relatorio


def build_table(records):
    """records: lista de dicts com n_byz, accuracy, erros, total."""
    by_nbyz = {}
    for r in records:
        by_nbyz.setdefault(r["n_byz"], []).append(r)

    cols = sorted(by_nbyz)
    rows = {"Média": [], "Mínimo": [], "Máximo": [], "Desvio padrão": [],
            "% erro na classificação": []}

    for n in cols:
        group = by_nbyz[n]
        accs = [r["accuracy"] for r in group if r["accuracy"] is not None]
        rows["Média"].append(statistics.mean(accs) if accs else None)
        rows["Mínimo"].append(min(accs) if accs else None)
        rows["Máximo"].append(max(accs) if accs else None)
        rows["Desvio padrão"].append(statistics.stdev(accs) if len(accs) > 1 else None)
        erros = sum(r["erros"] for r in group)
        total = sum(r["total"] for r in group)
        rows["% erro na classificação"].append(100 * erros / total if total else None)

    return cols, rows, by_nbyz


def fmt(value, is_pct):
    if value is None:
        return "-"
    return f"{value:.1f}%" if is_pct else f"{value:.4f}"


def print_table(cols, rows):
    header = ["Métrica"] + [
        f"{n} bizantino{'' if n == 1 else 's'}" for n in cols
    ]
    body = [
        [name] + [fmt(v, name.startswith("%")) for v in values]
        for name, values in rows.items()
    ]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]

    def line(cells):
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |")

    line(header)
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in body:
        line(r)


def write_tsv(path, cols, rows, decimal_comma):
    def cell(value, is_pct):
        s = fmt(value, is_pct)
        return s.replace(".", ",") if decimal_comma else s

    with open(path, "w") as f:
        f.write("Métrica\t" + "\t".join(
            f"{n} bizantino{'' if n == 1 else 's'}" for n in cols) + "\n")
        for name, values in rows.items():
            f.write(name + "\t" + "\t".join(
                cell(v, name.startswith("%")) for v in values) + "\n")


def write_csv(path, records):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "run", "cenario", "n_byz", "accuracy", "n_honestos_no_treino",
            "erros", "total", "falso_positivo", "falso_negativo"])
        w.writeheader()
        for r in records:
            w.writerow(r)


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--runs", type=int, default=10,
                    help="numero de execucoes do cliente (default: 10)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "resultados"),
                    help="diretorio de saida (default: resultados/)")
    ap.add_argument("--server-log", default=None,
                    help="usa um servidor ja no ar e le este log em vez de subir um novo")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    server = None
    if args.server_log:
        server_log = args.server_log
        if not port_is_open():
            sys.exit(f"ERRO: nada escutando em {HOST}:{PORT}. Suba o servidor primeiro.")
        offset = len(parse_server_log(open(server_log, errors="replace").read()))
        print(f"usando servidor ja no ar; {offset} rodadas ja no log serao ignoradas")
    else:
        server_log = os.path.join(args.outdir, "server.log")
        server = start_server(server_log)
        offset = 0

    try:
        all_scenarios = []
        for i in range(1, args.runs + 1):
            print(f"execucao {i}/{args.runs}...", flush=True)
            all_scenarios.append(run_client(i, args.outdir))
    finally:
        stop_server(server)

    rounds = parse_server_log(open(server_log, errors="replace").read())[offset:]
    expected = sum(len(s) for s in all_scenarios)
    if len(rounds) != expected:
        sys.exit(
            f"ERRO: {expected} cenarios executados mas {len(rounds)} rodadas no log do "
            f"servidor. Havia outro cliente ou servidor rodando em paralelo?"
        )

    records, k = [], 0
    for run_idx, scenarios in enumerate(all_scenarios, start=1):
        for scen_idx, clients in enumerate(scenarios):
            accuracy, n_honestos = rounds[k]
            k += 1
            fp = sum(1 for b, pred in clients if b == "honest" and pred)
            fn = sum(1 for b, pred in clients if b != "honest" and not pred)
            records.append({
                "run": run_idx,
                "cenario": scen_idx,
                "n_byz": sum(1 for b, _ in clients if b != "honest"),
                "accuracy": accuracy,
                "n_honestos_no_treino": n_honestos,
                "erros": fp + fn,
                "total": len(clients),
                "falso_positivo": fp,
                "falso_negativo": fn,
            })

    cols, rows, by_nbyz = build_table(records)
    print()
    print_table(cols, rows)

    print("\nDetalhe dos erros (falso positivo = honesto marcado bizantino):")
    for n in cols:
        g = by_nbyz[n]
        fp, fn = sum(r["falso_positivo"] for r in g), sum(r["falso_negativo"] for r in g)
        honestos = sum(r["total"] - r["n_byz"] for r in g)
        byz = sum(r["n_byz"] for r in g)
        print(f"  {n} bizantino(s): falso-positivo {fp}/{honestos}, falso-negativo {fn}/{byz}")

    write_csv(os.path.join(args.outdir, "acuracias.csv"), records)
    write_tsv(os.path.join(args.outdir, "tabela_ptbr.tsv"), cols, rows, True)
    write_tsv(os.path.join(args.outdir, "tabela_en.tsv"), cols, rows, False)
    print(f"\nArquivos gravados em {args.outdir}/")


if __name__ == "__main__":
    main()
