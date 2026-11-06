"""Re-test one configuration of experiment E3 on every sequence (or on a disjoint subset).

With --nist-max-seq, run_experiments.py tests evenly spaced sequences only. When one
uniformity check (P-value_T) falls near the 1e-4 threshold, this script repeats the
chosen tests on all sequences of that configuration, or on the sequences that the
capped run did not use, so that a chance result can be told apart from a real defect.

Example:
    python3 nist_recheck.py --data ~/Dataset/silesia --config "UHC-EtM (AES-CTR)" \
        --tests Serial-1,Serial-2 --subset unused --max 1000 --jobs 8 --out recheck.csv
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as R  # noqa: E402
from uhc import metrics  # noqa: E402
from uhc import variants as V  # noqa: E402
from uhc.primitives import ExperimentDRBG  # noqa: E402


def _worker(args):
    chunk, tests = args
    bits = metrics.to_bits(chunk)
    res = metrics.nist_battery(bits)
    return {t: res[t] for t in tests}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--config", required=True, help='configuration name as in e3_nist_summary.csv')
    ap.add_argument("--tests", default=",".join(metrics.NIST_TESTS))
    ap.add_argument("--subset", choices=["all", "unused"], default="unused",
                    help="all sequences, or those not used by a run with --nist-max-seq")
    ap.add_argument("--used-max", type=int, default=1000, help="the --nist-max-seq of the earlier run")
    ap.add_argument("--max", type=int, default=0, help="test at most this many of the selected sequences")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--nist-keys", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("nist_recheck.csv"))
    a = ap.parse_args()
    tests = a.tests.split(",")

    corpus = R.load_corpus(a.data)
    root = ExperimentDRBG(a.seed, "root")
    rsa_priv = V.c_recipient_keypair()
    schemes = R.build_schemes(root.child("schemes"), rsa_priv)
    m_all = sum(len(d) for _, d in corpus) * a.nist_keys // R.SEQ_BYTES
    used = set(int(i) for i in R.nist_selection(m_all, a.used_max))
    pool = [i for i in range(m_all) if a.subset == "all" or i not in used]
    if a.max and len(pool) > a.max:
        pool = [pool[int(j)] for j in np.unique(np.linspace(0, len(pool) - 1, a.max).round().astype(int))]
    wanted = set(pool)

    chunks, idx, buf = [], 0, b""
    for cfg, segments in R._nist_sources(corpus, schemes, root.child("e3"), a.nist_keys, rsa_priv):
        if cfg != a.config:  # other configurations are skipped without being computed;
            continue          # the experiment DRBG is consumed in the same order as in E3
        for seg in segments:
            buf += seg
            full = len(buf) // R.SEQ_BYTES
            for j in range(full):
                if idx + j in wanted:
                    chunks.append((idx + j, buf[j * R.SEQ_BYTES:(j + 1) * R.SEQ_BYTES]))
            idx += full
            buf = buf[full * R.SEQ_BYTES:]
        break

    work = [(c, tests) for _, c in chunks]
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(_worker, work, chunksize=4))
    else:
        results = [_worker(w) for w in work]
    rows = []
    for t in tests:
        sl = metrics.second_level([r[t] for r in results])
        rows.append({"config": a.config, "subset": a.subset, "test": t, **sl})
        print(f"{t}: m={sl['m']} proportion={sl['proportion']:.4f} P-value_T={sl['p_value_T']:.3g} "
              f"ok={sl['proportion_ok'] and sl['uniform_ok']}")
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
