#!/usr/bin/env python3
"""Run experiments E0-E5 of the paper on a corpus directory and write CSV, JSON,
LaTeX tables and figures.

Usage (Windows):
    python run_experiments.py --data "D:\\Projects\\Samsul Arifin\\Dataset\\cantrbry" --out results
Usage (Ubuntu):
    python3 run_experiments.py --data ~/Dataset/cantrbry --out results --pin-core 2

    --quick        small trial counts to check the pipeline (minutes, not hours)
    --only e2,e3   run a subset of experiments
    --jobs 8       parallel workers for the NIST tests only (timing stays single-core)

All randomness comes from a SHAKE-256 DRBG seeded by --seed, so a run can be replayed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import math
import os
import platform
import statistics
import string
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uhc import __version__ as UHC_VERSION  # noqa: E402
from uhc import attacks, etm, metrics, variants as V  # noqa: E402
from uhc.primitives import ExperimentDRBG, LogisticGenerator, numba_available, sha256_hex  # noqa: E402

LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def median(xs):
    return float(statistics.median(xs)) if xs else float("nan")


def iqr(xs):
    if len(xs) < 2:
        return float("nan")
    q = np.percentile(xs, [25, 75])
    return float(q[1] - q[0])


# =========================================================================== #
# Corpus and environment                                                      #
# =========================================================================== #
def load_corpus(path: Path) -> list[tuple[str, bytes]]:
    files = sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"no files found in {path}")
    return [(p.name, p.read_bytes()) for p in files]


def _cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _cpu_flags() -> dict:
    flags = {}
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("flags"):
                    f = set(line.split(":", 1)[1].split())
                    return {"aes_ni": "aes" in f, "avx2": "avx2" in f, "sha_ni": "sha_ni" in f}
        import cpuinfo  # optional: pip install py-cpuinfo

        f = set(cpuinfo.get_cpu_info().get("flags", []))
        flags = {"aes_ni": "aes" in f, "avx2": "avx2" in f, "sha_ni": "sha" in f or "sha_ni" in f}
    except Exception:
        flags = {"aes_ni": "unknown (pip install py-cpuinfo)", "avx2": "unknown", "sha_ni": "unknown"}
    return flags


def _gpu() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                       "--format=csv,noheader"], text=True, timeout=10)
        return out.strip().replace("\n", "; ")
    except Exception:
        return "not detected (nvidia-smi unavailable)"


def _version(mod: str) -> str:
    try:
        from importlib.metadata import version

        return version(mod)
    except Exception:
        return "not installed"


def environment() -> dict:
    import cryptography
    from cryptography.hazmat.backends.openssl.backend import backend as ossl

    try:
        import psutil

        mem = round(psutil.virtual_memory().total / 2 ** 30, 1)
        phys = psutil.cpu_count(logical=False)
    except Exception:
        mem, phys = "unknown", "unknown"
    os_name = platform.platform()
    try:
        if sys.platform.startswith("linux") and os.path.exists("/etc/os-release"):
            for line in open("/etc/os-release", encoding="utf-8"):
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"') + f" (kernel {platform.release()})"
    except Exception:
        pass
    return {
        "cpu": _cpu_name(), "physical_cores": phys, "logical_cores": os.cpu_count(),
        "cpu_flags": _cpu_flags(), "ram_gib": mem, "gpu": _gpu(), "os": os_name,
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": _version("scipy"), "cryptography": cryptography.__version__,
        "openssl": ossl.openssl_version_text(), "argon2-cffi": _version("argon2-cffi"),
        "matplotlib": _version("matplotlib"), "numba": _version("numba") if numba_available() else "not used",
        "ml_kem_backend": etm.envelope_backend_name(), "uhc_package": UHC_VERSION,
    }


def pin_core(core: int | None) -> str:
    if core is None:
        return "not pinned"
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})
            return f"pinned to core {core} (sched_setaffinity)"
        import psutil

        psutil.Process().cpu_affinity([core])
        return f"pinned to core {core} (psutil)"
    except Exception as exc:
        return f"pinning failed: {exc}"


# =========================================================================== #
# Scheme registry                                                             #
# =========================================================================== #
@dataclass
class Scheme:
    name: str
    enc: Callable[[bytes], bytes]
    dec: Callable[[bytes], bytes]
    body: Callable[[bytes], bytes]          # ciphertext bytes an observer sees, minus framing
    enc_with: Callable[[bytes, bytes, Callable[[int], bytes]], bytes]  # (key_material, data, rand)
    key: bytes                              # key material, for key-sensitivity flips


def _printable_password(drbg: ExperimentDRBG, length: int = 12) -> bytes:
    alphabet = (string.ascii_letters + string.digits + string.punctuation).encode()
    return bytes(alphabet[drbg.uint(len(alphabet))] for _ in range(length))


def build_schemes(drbg: ExperimentDRBG, rsa_priv) -> list[Scheme]:
    rand = drbg.child("scheme-randomness").bytes
    k_gcm, k_cc, k_etm = drbg.bytes(32), drbg.bytes(32), drbg.bytes(32)
    key_a = V.KeyA(drbg.uniform(0.01, 0.99), drbg.uniform(3.90, 4.00))
    pw_b = _printable_password(drbg)
    key_c = V.KeyC(16, drbg.uint(10 ** 10))
    pub = rsa_priv.public_key()
    p_aes = etm.EtMParams(stream="aes-ctr")
    p_cha = etm.EtMParams(stream="chacha20")

    def a_key(k: bytes) -> V.KeyA:
        x0, r = np.frombuffer(k, dtype=np.float64)
        return V.KeyA(float(x0), float(r))

    def c_key(k: bytes) -> V.KeyC:
        return V.KeyC(16, int.from_bytes(k, "big"))

    return [
        Scheme("AES-256-GCM", lambda d: etm.aesgcm_encrypt(k_gcm, d, rand), lambda c: etm.aesgcm_decrypt(k_gcm, c),
               lambda c: c[12:-16], lambda k, d, r: etm.aesgcm_encrypt(k, d, r), k_gcm),
        Scheme("ChaCha20-Poly1305", lambda d: etm.chacha_encrypt(k_cc, d, rand),
               lambda c: etm.chacha_decrypt(k_cc, c), lambda c: c[12:-16],
               lambda k, d, r: etm.chacha_encrypt(k, d, r), k_cc),
        Scheme("UHC-EtM (AES-CTR)", lambda d: etm.encrypt(k_etm, d, p_aes, rand), lambda c: etm.decrypt(k_etm, c),
               lambda c: c[etm.body_slice(c)], lambda k, d, r: etm.encrypt(k, d, p_aes, r), k_etm),
        Scheme("UHC-EtM (ChaCha20)", lambda d: etm.encrypt(k_etm, d, p_cha, rand), lambda c: etm.decrypt(k_etm, c),
               lambda c: c[etm.body_slice(c)], lambda k, d, r: etm.encrypt(k, d, p_cha, r), k_etm),
        Scheme("Variant A", lambda d: V.a_encrypt(key_a, d), lambda c: V.a_decrypt(key_a, c), lambda c: c,
               lambda k, d, r: V.a_encrypt(a_key(k), d), np.array([key_a.x0, key_a.r]).tobytes()),
        Scheme("Variant B", lambda d: V.b_encrypt(pw_b, d, rand), lambda c: V.b_decrypt(pw_b, c),
               lambda c: c[V.B_HEADER.size:], lambda k, d, r: V.b_encrypt(k, d, r), pw_b),
        Scheme("Variant C", lambda d: V.c_encrypt(key_c, d, pub, rand), lambda c: V.c_decrypt(key_c, c, rsa_priv),
               lambda c: c[V.C_HEADER.size:len(c) - V.c_parse(c)["wrapped"].__len__()],
               lambda k, d, r: V.c_encrypt(c_key(k), d, pub, r), key_c.p2.to_bytes(5, "big")),
    ]


def scheme_block(name: str, data: bytes) -> int:
    if name == "Variant B":
        return V.b_order(len(data), metrics.entropy(data))
    return 16


# =========================================================================== #
# E0: dataset characterisation                                                #
# =========================================================================== #
def stratum(h: float) -> str:
    return "low (<6)" if h < 6 else ("mid (6-7.9)" if h < 7.9 else "high (>=7.9)")


def e0(corpus, out: Path) -> list[dict]:
    rows = []
    for name, d in corpus:
        h = metrics.entropy(d)
        rows.append({"file": name, "bytes": len(d), "entropy": round(h, 4), "stratum": stratum(h),
                     "zlib_ratio": round(metrics.compression_ratio(d), 4), "variant_B_n": V.b_order(len(d), h),
                     "variant_C_alpha": V.c_alpha(d), "sha256": sha256_hex(d)})
    write_csv(out / "e0_dataset.csv", rows)
    return rows


# =========================================================================== #
# E1: round trip and ciphertext expansion                                      #
# =========================================================================== #
def e1(corpus, schemes, out: Path, drbg: ExperimentDRBG) -> list[dict]:
    rows = []
    k = drbg.bytes(32)
    for s in schemes:
        for name, d in corpus:
            blob = s.enc(d)
            rows.append({"scheme": s.name, "file": name, "roundtrip_ok": sha256_hex(s.dec(blob)) == sha256_hex(d),
                         "overhead_bytes": len(blob) - len(d)})
    # tamper rejection for UHC-EtM: flip one random byte per file
    rejected = 0
    for name, d in corpus:
        blob = bytearray(etm.encrypt(k, d, rand=drbg.bytes))
        blob[drbg.uint(len(blob))] ^= 1 + drbg.uint(255)
        try:
            etm.decrypt(k, bytes(blob))
        except etm.InvalidTag:
            rejected += 1
    rows.append({"scheme": "UHC-EtM tamper test", "file": "all", "roundtrip_ok": f"{rejected}/{len(corpus)} rejected",
                 "overhead_bytes": ""})
    write_csv(out / "e1_roundtrip.csv", rows)
    return rows


# =========================================================================== #
# E2: key recovery                                                             #
# =========================================================================== #
def e2(corpus, out: Path, drbg: ExperimentDRBG, trials: int) -> list[dict]:
    rows = []
    nfiles = len(corpus)
    rsa_cache = V.c_recipient_keypair()  # recipient key pair; not used by any attack

    for t in range(trials):
        name, d = corpus[t % nfiles]
        other_name, other = corpus[(t + 1) % nfiles]

        # --- Variant B, known plaintext ---
        pw = _printable_password(drbg)
        blob = V.b_encrypt(pw, d, drbg.bytes)
        t0 = time.perf_counter()
        r = attacks.attack_b_kpa(blob, d)
        dt = time.perf_counter() - t0
        n, lh = V.b_parse(blob)[:2]
        ok = r.success and r.recovered_plaintext == d[:lh] and np.array_equal(r.recovered_key_matrix, V.b_matrix(pw, n))
        rows.append({"attack": "B-KPA", "trial": t, "file": name, "n": n, "success": ok, "queries": 0,
                     "chosen_bytes": 0, "known_bytes": r.known_bytes, "recovered_fraction": lh / max(1, len(d)),
                     "time_ms": dt * 1e3})

        # --- Variant A, chosen plaintext ---
        ka = V.KeyA(drbg.uniform(0.01, 0.99), drbg.uniform(3.90, 4.00))
        target = V.a_encrypt(ka, d)
        t0 = time.perf_counter()
        r = attacks.attack_a_cpa(lambda q: V.a_encrypt(ka, q), target)
        dt = time.perf_counter() - t0
        ok = r.success and r.recovered_plaintext == d
        rows.append({"attack": "A-CPA", "trial": t, "file": name, "n": 16, "success": ok, "queries": r.queries,
                     "chosen_bytes": r.chosen_bytes, "known_bytes": 0, "recovered_fraction": 1.0 if ok else 0.0,
                     "time_ms": dt * 1e3})

        # --- Variant C, chosen plaintext ---
        kc = V.KeyC(8 + drbg.uint(57), drbg.uint(10 ** 10))
        pub = rsa_cache.public_key()
        target = V.c_encrypt(kc, d, pub, drbg.bytes)
        t0 = time.perf_counter()
        r = attacks.attack_c_cpa(lambda q: V.c_encrypt(kc, q, pub, drbg.bytes), target)
        dt = time.perf_counter() - t0
        n1 = V.c_split(d)
        ok = (r.success and r.recovered_plaintext == d[:n1]
              and np.array_equal(r.recovered_key_matrix, V.c_material(kc, 0)[0]))
        rows.append({"attack": "C-CPA", "trial": t, "file": name, "n": kc.p1, "success": ok, "queries": r.queries,
                     "chosen_bytes": r.chosen_bytes, "known_bytes": 0, "recovered_fraction": n1 / max(1, len(d)),
                     "time_ms": dt * 1e3})

        # --- Variant C, differential known plaintext (two files, same key) ---
        ba, bb = V.c_encrypt(kc, d, pub, drbg.bytes), V.c_encrypt(kc, other, pub, drbg.bytes)
        t0 = time.perf_counter()
        r = attacks.attack_c_diff_kpa(ba, bb, d, other)
        dt = time.perf_counter() - t0
        ok = r.success and np.array_equal(r.recovered_key_matrix, V.c_material(kc, 0)[0])
        rows.append({"attack": "C-diff-KPA", "trial": t, "file": f"{name}+{other_name}", "n": kc.p1, "success": ok,
                     "queries": 0, "chosen_bytes": 0, "known_bytes": r.known_bytes if r.success else "",
                     "recovered_fraction": "", "time_ms": dt * 1e3,
                     "reason": r.notes.get("reason", "")})

        # --- Variant C, tail confirmation without the RSA private key ---
        tail = d[n1:]
        t0 = time.perf_counter()
        right = V.c_tail_confirm(target, tail)
        dt = time.perf_counter() - t0
        wrong = V.c_tail_confirm(target, tail[:-1] + bytes([(tail[-1] ^ 1) if tail else 0]))
        rows.append({"attack": "C-tail-confirm", "trial": t, "file": name, "n": kc.p1, "success": right and not wrong,
                     "queries": 0, "chosen_bytes": 0, "known_bytes": len(tail), "recovered_fraction": "",
                     "time_ms": dt * 1e3})
        if (t + 1) % max(1, trials // 10) == 0:
            log(f"  E2 trial {t + 1}/{trials}")

    write_csv(out / "e2_attacks_trials.csv", rows)
    summary = []
    for atk in ["B-KPA", "A-CPA", "C-CPA", "C-diff-KPA", "C-tail-confirm"]:
        rs = [r for r in rows if r["attack"] == atk]
        k = sum(1 for r in rs if r["success"])
        lo, hi = metrics.clopper_pearson(k, len(rs))
        kb = [r["known_bytes"] for r in rs if r["success"] and r["known_bytes"] not in ("", None)]
        summary.append({
            "attack": atk, "trials": len(rs), "successes": k, "success_rate": k / len(rs), "ci95_lo": lo, "ci95_hi": hi,
            "median_queries": median([r["queries"] for r in rs]),
            "median_chosen_bytes": median([r["chosen_bytes"] for r in rs]),
            "median_known_bytes": median(kb) if kb else 0, "max_known_bytes": max(kb) if kb else 0,
            "median_time_ms": median([r["time_ms"] for r in rs]),
            "median_recovered_fraction": median([r["recovered_fraction"] for r in rs
                                                 if r["recovered_fraction"] not in ("", None)]),
        })
    write_csv(out / "e2_attacks_summary.csv", summary)
    # per-file known-plaintext requirement for B
    per_file = []
    for name, _ in corpus:
        rs = [r for r in rows if r["attack"] == "B-KPA" and r["file"] == name]
        per_file.append({"file": name, "n": rs[0]["n"] if rs else "",
                         "median_known_bytes": median([r["known_bytes"] for r in rs if r["success"]]),
                         "success": f"{sum(r['success'] for r in rs)}/{len(rs)}"})
    write_csv(out / "e2_b_kpa_per_file.csv", per_file)
    return summary


# =========================================================================== #
# E3: statistics and NIST SP 800-22                                            #
# =========================================================================== #
def e3_file_stats(corpus, schemes, out: Path) -> list[dict]:
    rows = []
    for name, d in corpus:
        rows.append(_stat_row("Plaintext", name, d, d, 16))
    for s in schemes:
        for name, d in corpus:
            body = s.body(s.enc(d))
            rows.append(_stat_row(s.name, name, d, body, scheme_block(s.name, d)))
    write_csv(out / "e3_file_stats.csv", rows)
    return rows


def _stat_row(scheme: str, name: str, plain: bytes, body: bytes, block: int) -> dict:
    chi2, p = metrics.chi_square_uniform(body)
    return {"scheme": scheme, "file": name, "bytes": len(body), "entropy": metrics.entropy(body),
            "chi2": chi2, "chi2_p": p, "adj_corr": metrics.adjacent_correlation(body),
            "zlib_ratio": metrics.compression_ratio(body), "block": block,
            "dup_block_ratio": metrics.duplicate_block_ratio(body, block),
            "zero_block_fraction": metrics.zero_block_fraction(body, block),
            "self_bit_distance": metrics.bit_diff_fraction(plain, body)}


SEQ_BITS = 1_000_000
SEQ_BYTES = SEQ_BITS // 8


def _battery_worker(chunk: bytes) -> dict:
    return metrics.nist_battery(metrics.to_bits(chunk))


def _nist_sources(corpus, schemes, drbg: ExperimentDRBG, n_keys: int, rsa_priv):
    """Yield (configuration, iterator of byte segments) without holding whole streams in memory.

    The stream of a scheme is the concatenation, over n_keys independent keys, of the
    ciphertext bodies of all corpus files. The generator streams have the same length.
    Segments are produced one file (or one 64 MiB block) at a time, so memory stays
    bounded for large corpora such as Silesia.
    """
    total = sum(len(d) for _, d in corpus)
    need = total * n_keys

    def scheme_segments(s):
        for i in range(n_keys):
            sub = build_schemes(drbg.child(f"nist-{s.name}-{i}"), rsa_priv)
            s_i = next(x for x in sub if x.name == s.name)
            for _, d in corpus:
                yield s_i.body(s_i.enc(d))

    for s in schemes:
        yield s.name, scheme_segments(s)

    block = 64 << 20
    gen = LogisticGenerator(drbg.uniform(0.01, 0.99), 3.923, warmup=1000)

    def logistic_segments():
        left = need
        while left:
            k = min(block, left)
            yield gen.take(k).tobytes()
            left -= k

    yield "Logistic q(x) generator", logistic_segments()
    import hashlib

    shake_seed = drbg.bytes(32)

    def shake_segments():
        if need <= (256 << 20):  # one call, as in the original protocol
            yield hashlib.shake_256(shake_seed).digest(need)
            return
        left, ctr = need, 0
        while left:
            k = min(block, left)
            yield hashlib.shake_256(shake_seed + ctr.to_bytes(8, "big")).digest(k)
            left -= k
            ctr += 1

    yield "SHAKE-256 generator", shake_segments()

    def urandom_segments():
        left = need
        while left:
            k = min(block, left)
            yield os.urandom(k)
            left -= k

    yield "os.urandom (control)", urandom_segments()


def nist_selection(m: int, max_seq: int) -> np.ndarray:
    """Indices of the sequences tested: all of them, or max_seq evenly spaced ones."""
    if not max_seq or m <= max_seq:
        return np.arange(m)
    return np.unique(np.linspace(0, m - 1, max_seq).round().astype(np.int64))


def e3_nist(corpus, schemes, out: Path, drbg: ExperimentDRBG, n_keys: int, jobs: int, rsa_priv,
            max_seq: int = 0) -> list[dict]:
    pv_rows, summary = [], []
    total = sum(len(d) for _, d in corpus) * n_keys
    m_all = total // SEQ_BYTES
    chosen = nist_selection(m_all, max_seq)
    chosen_set = set(int(i) for i in chosen)
    for cfg, segments in _nist_sources(corpus, schemes, drbg, n_keys, rsa_priv):
        t0 = time.perf_counter()
        counts = np.zeros(256, dtype=np.int64)
        chunks, idx, buf, nbytes = [], 0, b"", 0
        for seg in segments:
            counts += np.bincount(np.frombuffer(seg, dtype=np.uint8), minlength=256)
            nbytes += len(seg)
            buf += seg
            full = len(buf) // SEQ_BYTES
            for j in range(full):
                if idx + j in chosen_set:
                    chunks.append(buf[j * SEQ_BYTES:(j + 1) * SEQ_BYTES])
            idx += full
            buf = buf[full * SEQ_BYTES:]
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                results = list(ex.map(_battery_worker, chunks, chunksize=4))
        else:
            results = [_battery_worker(c) for c in chunks]
        m = len(results)
        log(f"  NIST {cfg}: {m} of {m_all} sequences in {time.perf_counter() - t0:.1f} s")
        for i, res in zip(chosen, results):
            pv_rows.append({"config": cfg, "sequence": int(i), **res})
        for test in metrics.NIST_TESTS:
            sl = metrics.second_level([r[test] for r in results])
            summary.append({"config": cfg, "test": test, **sl, "m_total": int(m_all)})
        # Relative byte frequency (1.0 = uniform) over the whole stream, on the three ranges
        # that the orbit support of the Logistic map at r = 3.923 creates for floor(1000 x) mod 256.
        byte_bias = counts / max(1, nbytes)
        for lo_b, hi_b in ((0, 74), (74, 213), (213, 256)):
            summary.append({"config": cfg, "test": f"byte frequency {lo_b}..{hi_b - 1} (rel. to uniform)",
                            "m": m, "proportion": float(byte_bias[lo_b:hi_b].mean() * 256)})
    write_csv(out / "e3_nist_pvalues.csv", pv_rows)
    write_csv(out / "e3_nist_summary.csv", summary)
    return summary


# =========================================================================== #
# E4: diffusion and key sensitivity                                            #
# =========================================================================== #
def _fixed_rand(seed: bytes) -> Callable[[int], bytes]:
    import hashlib

    def r(k: int) -> bytes:
        return hashlib.shake_256(seed + k.to_bytes(4, "big")).digest(k)

    return r


def _flip_key(s: Scheme, key: bytes, bit: int) -> bytes:
    if s.name == "Variant A":
        x0, r = np.frombuffer(key, dtype=np.float64)
        return np.array([np.nextafter(x0, 1.0), r]).tobytes()  # one-ulp change of x0
    if s.name == "Variant C":
        return (int.from_bytes(key, "big") ^ 1).to_bytes(len(key), "big")  # p2 + /- 1
    b = bytearray(key)
    b[(bit // 8) % len(b)] ^= 1 << (bit % 8)
    if s.name == "Variant B":  # keep the password printable
        b[(bit // 8) % len(b)] = 33 + (b[(bit // 8) % len(b)] - 33) % 94
    return bytes(b)


def lemma_prediction(pos: int, length: int, n: int = 16, lanes: int = 1) -> float:
    """Expected changed-bit fraction of a chained UHC-EtM body after flipping plaintext bit ``pos``.

    Flipping bit t of a byte (t = 0 is the least significant bit) changes, in every
    later diffused byte of the same lane, only bits t..7, each with probability about
    1/2 (lemma lem:bitplane of the paper), i.e. (8 - t)/2 bits per byte. Bytes before the flipped block, other
    lanes and the undiffused tail of fewer than n bytes are unchanged.
    """
    t = 7 - pos % 8
    byte = pos // 8
    diffused = (length // n) * n
    if byte >= diffused:
        return 1.0 / (8 * length)
    first = (byte // n) * n
    blocks_after = (diffused - first) // n
    return (8 - t) / 2.0 * n * math.ceil(blocks_after / lanes) / (8 * length)


def e4(corpus, schemes, out: Path, drbg: ExperimentDRBG, trials: int) -> list[dict]:
    rows = []
    for s in schemes:
        for name, d in corpus:
            fixed = _fixed_rand(drbg.bytes(16))
            base = s.body(s.enc_with(s.key, d, fixed))
            for t in range(trials):
                pos = drbg.uint(8 * len(d))
                m = bytearray(d)
                m[pos // 8] ^= 0x80 >> (pos % 8)
                ct = s.body(s.enc_with(s.key, bytes(m), fixed))
                rows.append({"scheme": s.name, "file": name, "kind": "plaintext-bit", "trial": t,
                             "position_fraction": pos / (8 * len(d)), "bit_index": 7 - pos % 8,
                             "lemma_prediction": (lemma_prediction(pos, len(d)) if s.name.startswith("UHC-EtM")
                                                  else float("nan")),
                             "bit_diff": metrics.bit_diff_fraction(base, ct),
                             "byte_diff": metrics.byte_diff_fraction(base, ct)})
            for t in range(max(3, trials // 10)):
                k2 = _flip_key(s, s.key, drbg.uint(8 * len(s.key)))
                ct = s.body(s.enc_with(k2, d, fixed))
                rows.append({"scheme": s.name, "file": name, "kind": "key-bit", "trial": t, "position_fraction": "",
                             "bit_index": "", "lemma_prediction": float("nan"),
                             "bit_diff": metrics.bit_diff_fraction(base, ct),
                             "byte_diff": metrics.byte_diff_fraction(base, ct)})
        log(f"  E4 {s.name} done")
    write_csv(out / "e4_diffusion.csv", rows)
    summary = []
    for s in schemes:
        for kind in ("plaintext-bit", "key-bit"):
            rs = [r for r in rows if r["scheme"] == s.name and r["kind"] == kind]
            bd = [r["bit_diff"] for r in rs]
            yd = [r["byte_diff"] for r in rs]
            summary.append({"scheme": s.name, "kind": kind, "n": len(rs), "bit_diff_mean": float(np.mean(bd)),
                            "bit_diff_median": median(bd), "bit_diff_min": float(np.min(bd)),
                            "bit_diff_max": float(np.max(bd)), "byte_diff_mean": float(np.mean(yd)),
                            "lemma_prediction_mean": float(np.mean([r["lemma_prediction"] for r in rs]))})
    write_csv(out / "e4_summary.csv", summary)
    return summary


# =========================================================================== #
# E5: performance, key handling and ablation                                   #
# =========================================================================== #
def _time(fn: Callable[[], object], runs: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e9)
    return ts


MIB = float(1 << 20)


def e5(corpus, schemes, out: Path, runs: int, warmup: int) -> list[dict]:
    rows, summary = [], []
    small = min(corpus, key=lambda x: len(x[1]))
    large = max(corpus, key=lambda x: len(x[1]))
    total = sum(len(d) for _, d in corpus)
    for s in schemes:
        enc_t = dec_t = 0.0
        for name, d in corpus:
            blob = s.enc(d)
            te = _time(lambda: s.enc(d), runs, warmup)
            td = _time(lambda: s.dec(blob), runs, warmup)
            rows.append({"scheme": s.name, "file": name, "bytes": len(d), "enc_median_s": median(te),
                         "enc_iqr_s": iqr(te), "dec_median_s": median(td), "dec_iqr_s": iqr(td)})
            enc_t += median(te)
            dec_t += median(td)
        tracemalloc.start()
        s.enc(large[1])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sm = next(r for r in rows if r["scheme"] == s.name and r["file"] == small[0])
        lg = next(r for r in rows if r["scheme"] == s.name and r["file"] == large[0])
        summary.append({"scheme": s.name, "corpus_enc_MiBps": total / MIB / enc_t,
                        "corpus_dec_MiBps": total / MIB / dec_t,
                        f"large_enc_MiBps ({large[0]})": len(large[1]) / MIB / lg["enc_median_s"],
                        f"small_enc_us ({small[0]})": sm["enc_median_s"] * 1e6,
                        f"small_dec_us ({small[0]})": sm["dec_median_s"] * 1e6,
                        "peak_mem_MiB": peak / MIB})
        log(f"  E5 {s.name}: {summary[-1]['corpus_enc_MiBps']:.2f} MiB/s enc")
    write_csv(out / "e5_perf.csv", rows)
    write_csv(out / "e5_summary.csv", summary)
    return summary


def e5_keys(out: Path, runs: int, argon2_first: bool) -> list[dict]:
    rows = []
    t = _time(V.c_recipient_keypair, max(5, runs // 3), 1)
    rows.append({"operation": "RSA-2048 key generation (Variant C)", "median_ms": median(t) * 1e3, "runs": len(t)})
    priv = V.c_recipient_keypair()
    pub = priv.public_key()
    ct = pub.encrypt(os.urandom(16), V._OAEP)
    rows.append({"operation": "RSA-2048-OAEP wrap", "median_ms": median(_time(lambda: pub.encrypt(os.urandom(16), V._OAEP), runs, 2)) * 1e3, "runs": runs})
    rows.append({"operation": "RSA-2048-OAEP unwrap", "median_ms": median(_time(lambda: priv.decrypt(ct, V._OAEP), runs, 2)) * 1e3, "runs": runs})
    salt = os.urandom(16)
    configs = [("Argon2id RFC 9106 second option (t=3, m=64 MiB, p=4)", etm.ARGON2_RFC9106_SECOND)]
    if argon2_first:
        configs.append(("Argon2id RFC 9106 first option (t=1, m=2 GiB, p=4)", etm.ARGON2_RFC9106_FIRST))
    for label, a in configs:
        t = _time(lambda: etm.argon2id(b"correct horse battery", salt, a["t"], a["m_kib"], a["p"]), 5, 1)
        rows.append({"operation": label, "median_ms": median(t) * 1e3, "runs": len(t)})
    import hashlib

    rows.append({"operation": "SHA-256(password), Variant B key derivation",
                 "median_ms": median(_time(lambda: hashlib.sha256(b"correct horse battery").digest(), runs * 100, 10)) * 1e3,
                 "runs": runs * 100})
    if etm.envelope_backend_name() != "unavailable":
        t = _time(etm.envelope_keygen, runs, 2)
        rows.append({"operation": f"Envelope key generation (X25519 + {etm.envelope_backend_name()})", "median_ms": median(t) * 1e3, "runs": runs})
        rk = etm.envelope_keygen()
        fk = os.urandom(32)
        blob = etm.envelope_wrap(rk, fk)
        rows.append({"operation": "Envelope wrap (hybrid X25519 + ML-KEM-768)", "median_ms": median(_time(lambda: etm.envelope_wrap(rk, fk), runs, 2)) * 1e3, "runs": runs})
        rows.append({"operation": "Envelope unwrap (hybrid X25519 + ML-KEM-768)", "median_ms": median(_time(lambda: etm.envelope_unwrap(rk, blob), runs, 2)) * 1e3, "runs": runs})
        rows.append({"operation": "Envelope size (bytes)", "median_ms": len(blob), "runs": 1})
    else:
        rows.append({"operation": "Envelope (ML-KEM-768)", "median_ms": "skipped: no ML-KEM backend", "runs": 0})
    write_csv(out / "e5_keys_envelope.csv", rows)
    return rows


def e5_ablation(corpus, out: Path, drbg: ExperimentDRBG, runs: int, warmup: int, flips: int = 50) -> list[dict]:
    k = drbg.bytes(32)
    probe_name, probe = max(corpus, key=lambda x: len(x[1]) if x[1].count(0) < len(x[1]) // 2 else 0)
    configs = [
        ("Reference: n=16, lanes=1, chaining, UHC on", etm.EtMParams()),
        ("n = 8", etm.EtMParams(n=8)), ("n = 32", etm.EtMParams(n=32)), ("n = 64", etm.EtMParams(n=64)),
        ("lanes = 64", etm.EtMParams(lanes=64)), ("lanes = 1024", etm.EtMParams(lanes=1024)),
        ("chaining off", etm.EtMParams(chain=False)),
        ("UHC layer off (AES-CTR + HMAC only)", etm.EtMParams(uhc=False)),
        ("ChaCha20 instead of AES-CTR", etm.EtMParams(stream="chacha20")),
    ]
    total = sum(len(d) for _, d in corpus)
    rows = []
    for label, p in configs:
        enc_t = dec_t = 0.0
        for _, d in corpus:
            r = drbg.child(label).bytes
            blob = etm.encrypt(k, d, p, r)
            enc_t += median(_time(lambda: etm.encrypt(k, d, p, r), runs, warmup))
            dec_t += median(_time(lambda: etm.decrypt(k, blob), runs, warmup))
        fixed = _fixed_rand(drbg.bytes(16))
        base = etm.encrypt(k, probe, p, fixed)
        diffs, spans, preds = [], [], []
        for _ in range(flips):
            pos = drbg.uint(8 * len(probe))
            m = bytearray(probe)
            m[pos // 8] ^= 0x80 >> (pos % 8)
            ct = etm.encrypt(k, bytes(m), p, fixed)
            sl = etm.body_slice(base)
            diffs.append(metrics.bit_diff_fraction(base[sl], ct[sl]))
            spans.append(1 - pos / (8 * len(probe)))
            preds.append(lemma_prediction(pos, len(probe), p.n, p.lanes) if p.uhc and p.chain else float("nan"))
        rows.append({"configuration": label, "corpus_enc_MiBps": total / MIB / enc_t, "corpus_dec_MiBps": total / MIB / dec_t,
                     f"avalanche_bit_diff_mean ({probe_name})": float(np.mean(diffs)),
                     "expected_if_forward_only (0.5 x remaining fraction)": 0.5 * float(np.mean(spans)),
                     "lemma2_prediction": float(np.mean(preds))})
        log(f"  ablation {label}: {rows[-1]['corpus_enc_MiBps']:.2f} MiB/s")
    write_csv(out / "e5_ablation.csv", rows)
    return rows


# =========================================================================== #
# Figures                                                                      #
# =========================================================================== #
def figures(corpus, schemes, out: Path, drbg: ExperimentDRBG) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log("  matplotlib not available, figures skipped")
        return []
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    made = []
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})

    # Figure 1: ptt5 (1728 x 2376 one-bit fax page) under three schemes
    ptt5 = dict(corpus).get("ptt5")
    if ptt5 is not None and len(ptt5) == 216 * 2376:
        by = {s.name: s for s in schemes}
        panels = [("Plaintext (ptt5)", ptt5),
                  ("Variant B ciphertext", by["Variant B"].enc(ptt5)[V.B_HEADER.size:][:len(ptt5)]),
                  ("Variant A ciphertext", by["Variant A"].enc(ptt5)[:len(ptt5)]),
                  ("UHC-EtM ciphertext", by["UHC-EtM (AES-CTR)"].body(by["UHC-EtM (AES-CTR)"].enc(ptt5))[:len(ptt5)])]
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9))
        for ax, (title, buf) in zip(axes, panels):
            img = np.unpackbits(np.frombuffer(buf, dtype=np.uint8)).reshape(2376, 1728)
            ax.imshow(1 - img[::4, ::4], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"fig_ptt5_ciphertexts.{ext}", dpi=300)
        plt.close(fig)
        made.append("fig_ptt5_ciphertexts")

    # Figure 2: byte frequencies of the Logistic generator vs the uniform level
    gen = LogisticGenerator(drbg.uniform(0.01, 0.99), 3.923, warmup=1000)
    sample = gen.take(4_000_000)
    freq = np.bincount(sample, minlength=256) / sample.size
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar(np.arange(256), freq * 256, width=1.0, color="#2a78d6", linewidth=0)
    ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
    ax.text(250, 1.04, "uniform", ha="right", va="bottom", fontsize=7, color="#52514e")
    ax.set_xlim(-1, 256)
    ax.set_xlabel("Byte value q(x)")
    ax.set_ylabel("Frequency / (1/256)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_logistic_byte_bias.{ext}", dpi=300)
    plt.close(fig)
    made.append("fig_logistic_byte_bias")
    return made


# =========================================================================== #
# LaTeX tables                                                                 #
# =========================================================================== #
class Raw(str):
    """A table cell that is already LaTeX and must not be escaped."""


def tex_escape(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}")]:
        s = s.replace(a, b)
    return s


def tex_table(label: str, caption: str, header: list[str], rows: list[list], spec: str, wide: bool = False,
              size: str = "footnotesize", sep: str = "4pt") -> str:
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\{size}", f"\\setlength{{\\tabcolsep}}{{{sep}}}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(_head(h) for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) if isinstance(c, Raw) or str(c).startswith("$") else tex_escape(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def _head(h: str) -> str:
    """A header cell; a newline in the text gives a two-line, bottom-aligned cell."""
    if "\n" not in h:
        return h
    return "\\begin{tabular}[b]{@{}c@{}}" + " \\\\ ".join(h.split("\n")) + "\\end{tabular}"


def f(x, nd=3):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, float):
        if math.isnan(x):
            return "--"
        if nd == 0:
            return f"{x:,.0f}"
        if x != 0 and abs(x) < 0.5 * 10 ** -nd:
            mant, exp = f"{x:.1e}".split("e")
            return f"${mant}\\times 10^{{{int(exp)}}}$"
        return f"{x:.{nd}f}"
    return str(x)


def write_tables(out: Path, res: dict, env: dict, corpus_name: str, label_suffix: str = "") -> None:
    parts = [f"% Generated by run_experiments.py for {corpus_name}. Labels use the prefix tab:res-"
             + (f" and the suffix -{label_suffix}." if label_suffix else "."), ""]
    n_files = len({r["file"] for r in res.get("e3_stats", []) if r.get("scheme") == "Plaintext"}) or len(res.get("e0", []))
    if "env" in res:
        e = env
        parts.append(tex_table("tab:res-env", "Measured hardware and software environment.", ["Component", "Value"], [
            ["Processor", e["cpu"]], ["Cores (physical / logical)", f"{e['physical_cores']} / {e['logical_cores']}"],
            ["AES-NI / AVX2", f"{e['cpu_flags'].get('aes_ni')} / {e['cpu_flags'].get('avx2')}"],
            ["Memory", f"{e['ram_gib']} GiB"], ["Graphics card", e["gpu"]], ["Operating system", e["os"]],
            ["Python", e["python"]], ["NumPy / SciPy", f"{e['numpy']} / {e['scipy']}"],
            ["cryptography / OpenSSL", f"{e['cryptography']} / {e['openssl']}"],
            ["argon2-cffi", e["argon2-cffi"]], ["ML-KEM-768 backend", e["ml_kem_backend"]]], "@{}ll@{}"))
    if "e0" in res:
        rows = [[r["file"], f(r["bytes"]), f(r["entropy"], 3), r["stratum"], f(r["zlib_ratio"], 3), r["variant_B_n"],
                 r["sha256"][:12]] for r in res["e0"]]
        parts.append(tex_table("tab:res-data", f"{corpus_name}: size, empirical byte entropy, entropy stratum, "
                               "zlib ratio, order chosen by Variant B and SHA-256 prefix.",
                               ["File", "Bytes", "$H$", "Stratum", "zlib", "$n_B$", "SHA-256"], rows, "@{}lrrlrrl@{}"))
    if "e2" in res:
        names = {"B-KPA": ("B", "KPA"), "A-CPA": ("A", "CPA"), "C-CPA": ("C", "CPA"),
                 "C-diff-KPA": ("C", "KPA (2 files)"), "C-tail-confirm": ("C (tail)", "KPA")}
        rows = []
        for r in res["e2"]:
            v, mdl = names[r["attack"]]
            data = (f"{f(r['median_known_bytes'], 0)} B (max {f(r['max_known_bytes'], 0)})" if r["median_known_bytes"]
                    else f"{f(r['median_queries'], 0)} queries, {f(r['median_chosen_bytes'], 0)} B")
            rows.append([v, mdl, data, f"{r['successes']}/{r['trials']}",
                         f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]", f(r["median_time_ms"], 2),
                         f(r["median_recovered_fraction"], 3)])
        parts.append(tex_table("tab:res-attacks", "Key recovery against the three variants (experiment E2). Data is the "
                               "median known plaintext in bytes, scanned from the start of the file (maximum over trials "
                               "in brackets), or the number and total size of chosen-plaintext queries; success is exact recovery; CI is the Clopper--Pearson 95\\% interval; recovered is "
                               "the fraction of the file whose plaintext the attack returns.",
                               ["Variant", "Model", "Data", "Success", "95\\% CI", "Time\n(ms)", "Recov-\nered"], rows,
                               "@{}lllrlrr@{}", wide=True))
    if "e3_stats" in res:
        rows = []
        order = ["Plaintext"] + [s for s in res["scheme_names"]]
        for sc in order:
            rs = [r for r in res["e3_stats"] if r["scheme"] == sc]
            ent = [r["entropy"] for r in rs]
            corr = [abs(r["adj_corr"]) for r in rs if not math.isnan(r["adj_corr"])]
            chi_pass = sum(1 for r in rs if r["chi2_p"] >= 0.01)
            rows.append([sc, f(min(ent), 4), f(float(np.mean(ent)), 4), f"{chi_pass}/{len(rs)}",
                         f(max(corr) if corr else float('nan'), 4), f(max(r["zlib_ratio"] for r in rs), 3),
                         f(min(r["zlib_ratio"] for r in rs), 3), f(max(r["dup_block_ratio"] for r in rs), 4),
                         f(float(np.mean([r["self_bit_distance"] for r in rs])), 3)])
        parts.append(tex_table("tab:res-stats", f"Byte statistics of the ciphertext bodies over the {n_files} files of {corpus_name} "
                               "in experiment E3. $\\chi^2$ pass counts files with $p \\ge 0.01$; dup.\\ is the largest "
                               "fraction of repeated blocks; self-dist.\\ is the plaintext-to-ciphertext bit distance that "
                               "the drafts reported as avalanche.",
                               ["Scheme", "min\n$H$", "mean\n$H$", "$\\chi^2$\npass", "max\n$|r|$", "max\nzlib", "min\nzlib",
                                "max\ndup.", "self-\ndist."], rows, "@{}lrrrrrrrr@{}", wide=True))
    if "e3_nist" in res:
        cfgs = list(dict.fromkeys(r["config"] for r in res["e3_nist"]))
        abbrev = {"AES-256-GCM": "GCM", "ChaCha20-Poly1305": "C20P", "UHC-EtM (AES-CTR)": "EtM-A",
                  "UHC-EtM (ChaCha20)": "EtM-C", "Variant A": "Var.A", "Variant B": "Var.B", "Variant C": "Var.C",
                  "Logistic q(x) generator": "Logistic", "SHAKE-256 generator": "SHAKE", "os.urandom (control)": "urandom"}
        short = {c: abbrev.get(c, c) for c in cfgs}
        rows = []
        for test in metrics.NIST_TESTS:
            row = [test]
            for c in cfgs:
                r = next((x for x in res["e3_nist"] if x["config"] == c and x["test"] == test), None)
                if r and r.get("m"):
                    mark = "" if r["proportion_ok"] and r.get("uniform_ok") in (True, None) else "*"
                    row.append(f"{r['proportion']:.3f}{mark}")
                else:
                    row.append("--")
            rows.append(row)
        m = next((x["m"] for x in res["e3_nist"] if x.get("m")), 0)
        rng = next((x for x in res["e3_nist"] if x.get("range_lo")), None)
        m_tot = next((x.get("m_total") for x in res["e3_nist"] if x.get("m")), None)
        try:
            m_tot = int(m_tot)
        except (TypeError, ValueError):
            m_tot = None
        sampled = bool(m_tot and m_tot > m)
        seqs = f"{m:,} of the {m_tot:,} sequences" if sampled else f"{m:,} sequences"
        cap = (f"NIST SP 800-22 pass proportions over {seqs} of $10^6$ bits (experiment E3)"
               + ("; the tested sequences are evenly spaced over the stream. " if sampled else ". ") +
               f"Acceptable range [{rng['range_lo']:.4f}, 1]; * marks a proportion outside the range or a "
               f"$P$-value$_T < 10^{{-4}}$." if rng else "NIST SP 800-22 pass proportions.")
        cap += (" GCM: AES-256-GCM; C20P: ChaCha20-Poly1305; EtM-A, EtM-C: UHC-EtM with AES-CTR or ChaCha20; "
                "Logistic, SHAKE, urandom: raw generator output.")
        parts.append(tex_table("tab:res-nist", cap, ["Test"] + [tex_escape(short[c]) for c in cfgs], rows,
                               "@{}l" + "r" * len(cfgs) + "@{}", wide=True, size="scriptsize", sep="3pt"))
    if "e4" in res:
        rows = []
        for sc in res["scheme_names"]:
            p = next(r for r in res["e4"] if r["scheme"] == sc and r["kind"] == "plaintext-bit")
            k = next(r for r in res["e4"] if r["scheme"] == sc and r["kind"] == "key-bit")
            rows.append([sc, f(p["bit_diff_mean"], 5), f(p["lemma_prediction_mean"], 5), f(p["bit_diff_median"], 5),
                         f(p["bit_diff_max"], 4), f(k["bit_diff_mean"], 4), f(k["byte_diff_mean"], 4)])
        parts.append(tex_table("tab:res-diffusion", "Single-bit plaintext avalanche and single-bit key sensitivity "
                               "(experiment E4), as the fraction of ciphertext-body bits (or bytes) that change, with "
                               "the nonce or IV held fixed. Pred.\\ is the mean predicted by Lemma~\\ref{lem:bitplane} for the same "
                               "flip positions "
                               "(UHC-EtM only).",
                               ["Scheme", "PT\nmean", "Pred.", "PT\nmedian", "PT\nmax", "Key\nbits", "Key\nbytes"], rows,
                               "@{}lrrrrrr@{}"))
    if "e5" in res:
        rows = []
        for r in res["e5"]:
            small_e = next(v for k, v in r.items() if k.startswith("small_enc_us"))
            small_d = next(v for k, v in r.items() if k.startswith("small_dec_us"))
            rows.append([r["scheme"], f(r["corpus_enc_MiBps"], 2), f(r["corpus_dec_MiBps"], 2), f(small_e, 1),
                         f(small_d, 1), f(r["peak_mem_MiB"], 1)])
        parts.append(tex_table("tab:res-perf", "Throughput over the whole corpus and latency on the smallest file "
                               "(experiment E5), single core, median of the timed runs.",
                               ["Scheme", "Enc.\n(MiB/s)", "Dec.\n(MiB/s)", "Small enc.\n($\\mu$s)",
                                "Small dec.\n($\\mu$s)", "Peak mem.\n(MiB)"], rows, "@{}lrrrrr@{}"))
    if "e5_keys" in res:
        rows = [[r["operation"], (f"{int(r['median_ms'])}" if str(r["operation"]).startswith("Envelope size")
                                  else f(r["median_ms"], 3) if isinstance(r["median_ms"], float) else r["median_ms"])]
                for r in res["e5_keys"]]
        parts.append(tex_table("tab:res-keys", "Cost of key derivation and key transport (median, ms).",
                               ["Operation", "Median (ms)"], rows, "@{}lr@{}"))
    if "e5_ablation" in res:
        rows = []
        for r in res["e5_ablation"]:
            av = next(v for k, v in r.items() if k.startswith("avalanche"))
            ex = next(v for k, v in r.items() if k.startswith("expected"))
            label = {"Reference: n=16, lanes=1, chaining, UHC on": "Reference ($n=16$, $L=1$, chained)",
                     "UHC layer off (AES-CTR + HMAC only)": "UHC layer off (CTR + HMAC)",
                     "ChaCha20 instead of AES-CTR": "ChaCha20 for AES-CTR"}.get(r["configuration"], r["configuration"])
            if label.startswith("n = ") or label.startswith("lanes = "):
                sym, val = label.split(" = ")
                label = f"${'n' if sym == 'n' else 'L'} = {val}$"
            rows.append([Raw(label), f(r["corpus_enc_MiBps"], 2), f(r["corpus_dec_MiBps"], 2), f(av, 4),
                         f(r["lemma2_prediction"], 4), f(ex, 4)])
        parts.append(tex_table("tab:res-ablation", "Ablation of UHC-EtM (experiment E5): throughput and single-bit "
                               "avalanche on the largest corpus file. Lemma is the avalanche predicted by Lemma~\\ref{lem:bitplane} for the same flip "
                               "positions; the last column is the value expected if every later bit flipped with "
                               "probability 1/2.",
                               ["Configuration", "Enc.\n(MiB/s)", "Dec.\n(MiB/s)", "Avalanche", "Lemma", "Forward,\nall bits"],
                               rows, "@{}lrrrrr@{}"))
    text = "\n".join(parts)
    if label_suffix:
        text = re.sub(r"\\label\{(tab:res-[a-z0-9-]+)\}", lambda m: "\\label{" + m.group(1) + "-" + label_suffix + "}", text)
    (out / "tables.tex").write_text(text, encoding="utf-8")


def load_results(out: Path) -> tuple[dict, dict]:
    """Rebuild the result dictionary from the CSV files of an earlier run (for --tables-only)."""
    import pandas as pd

    def recs(name):
        path = out / name
        return pd.read_csv(path).to_dict("records") if path.exists() else None

    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    res: dict = {"env": True}
    for key, name in (("e0", "e0_dataset.csv"), ("e2", "e2_attacks_summary.csv"), ("e3_stats", "e3_file_stats.csv"),
                      ("e3_nist", "e3_nist_summary.csv"), ("e4", "e4_summary.csv"), ("e5", "e5_summary.csv"),
                      ("e5_keys", "e5_keys_envelope.csv"), ("e5_ablation", "e5_ablation.csv")):
        r = recs(name)
        if r is not None:
            res[key] = r
    if "e4" in res:
        res["scheme_names"] = list(dict.fromkeys(r["scheme"] for r in res["e4"]))
    elif "e3_stats" in res:
        res["scheme_names"] = [x for x in dict.fromkeys(r["scheme"] for r in res["e3_stats"]) if x != "Plaintext"]
    return res, env


# =========================================================================== #
# Main                                                                         #
# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="corpus directory (e.g. the Canterbury files)")
    ap.add_argument("--corpus-name", default=None, help="name used in table captions (default: data folder name)")
    ap.add_argument("--label-suffix", default="", help="suffix for table labels, e.g. silesia -> tab:res-data-silesia")
    ap.add_argument("--tables-only", action="store_true",
                    help="only rebuild tables.tex from the CSV files already in --out")
    ap.add_argument("--out", default=Path("results"), type=Path)
    ap.add_argument("--seed", default=20261102, type=int)
    ap.add_argument("--trials", default=110, type=int, help="E2 trials (keys) per attack")
    ap.add_argument("--runs", default=30, type=int, help="E5 timed runs")
    ap.add_argument("--warmup", default=5, type=int, help="E5 warm-up runs")
    ap.add_argument("--nist-keys", default=5, type=int, help="E3 keys; corpus bytes x keys / 125000 sequences")
    ap.add_argument("--nist-max-seq", default=0, type=int,
                    help="E3: test at most this many evenly spaced sequences per configuration (0 = all)")
    ap.add_argument("--avalanche-trials", default=50, type=int, help="E4 bit flips per file and scheme")
    ap.add_argument("--jobs", default=1, type=int, help="parallel workers for NIST tests only")
    ap.add_argument("--pin-core", default=None, type=int, help="pin the process to one CPU core")
    ap.add_argument("--argon2-first", action="store_true", help="also time Argon2id with 2 GiB (RFC 9106 option 1)")
    ap.add_argument("--quick", action="store_true", help="small counts for a pipeline check")
    ap.add_argument("--only", default="e0,e1,e2,e3,e4,e5,fig", help="comma list of e0,e1,e2,e3,e4,e5,fig")
    a = ap.parse_args()
    name = a.corpus_name or (a.data.resolve().name if a.data else a.out.resolve().name)
    if a.tables_only:
        res, env = load_results(a.out)
        write_tables(a.out, res, env, name, a.label_suffix)
        print(f"tables.tex rebuilt in {a.out.resolve()}")
        return
    if a.data is None:
        ap.error("--data is required unless --tables-only is given")
    if a.quick:
        a.trials, a.runs, a.warmup, a.nist_keys, a.avalanche_trials = 22, 3, 1, 1, 5
    only = set(a.only.split(","))
    a.out.mkdir(parents=True, exist_ok=True)

    pin = pin_core(a.pin_core)
    env = environment()
    env["pinning"] = pin
    (a.out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (a.out / "assumptions.json").write_text(json.dumps(V.ASSUMPTIONS, indent=2), encoding="utf-8")
    (a.out / "run_config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2), encoding="utf-8")
    log(f"CPU: {env['cpu']} | {pin} | ML-KEM: {env['ml_kem_backend']}")

    corpus = load_corpus(a.data)
    log(f"Corpus: {len(corpus)} files, {sum(len(d) for _, d in corpus):,} bytes from {a.data}")
    root = ExperimentDRBG(a.seed, "root")
    rsa_priv = V.c_recipient_keypair()
    schemes = build_schemes(root.child("schemes"), rsa_priv)
    res: dict = {"env": True, "scheme_names": [s.name for s in schemes]}
    t_all = time.perf_counter()

    if "e0" in only:
        log("E0 dataset characterisation")
        res["e0"] = e0(corpus, a.out)
    if "e1" in only:
        log("E1 round trips and expansion")
        rows = e1(corpus, schemes, a.out, root.child("e1"))
        bad = [r for r in rows if r["roundtrip_ok"] is False]
        log(f"  {len(rows) - 1 - len(bad)}/{len(rows) - 1} round trips OK; {rows[-1]['roundtrip_ok']}")
    if "e2" in only:
        log(f"E2 key recovery ({a.trials} trials per attack)")
        res["e2"] = e2(corpus, a.out, root.child("e2"), a.trials)
        for r in res["e2"]:
            log(f"  {r['attack']}: {r['successes']}/{r['trials']} success")
    if "e3" in only:
        log("E3 byte statistics")
        res["e3_stats"] = e3_file_stats(corpus, schemes, a.out)
        log(f"E3 NIST SP 800-22 ({a.nist_keys} keys, jobs={a.jobs})")
        res["e3_nist"] = e3_nist(corpus, schemes, a.out, root.child("e3"), a.nist_keys, a.jobs, rsa_priv,
                                a.nist_max_seq)
    if "e4" in only:
        log(f"E4 diffusion ({a.avalanche_trials} flips per file)")
        res["e4"] = e4(corpus, schemes, a.out, root.child("e4"), a.avalanche_trials)
    if "e5" in only:
        log(f"E5 performance ({a.runs} runs, {a.warmup} warm-up)")
        res["e5"] = e5(corpus, schemes, a.out, a.runs, a.warmup)
        res["e5_keys"] = e5_keys(a.out, a.runs, a.argon2_first)
        res["e5_ablation"] = e5_ablation(corpus, a.out, root.child("ablation"), a.runs, a.warmup,
                                       a.avalanche_trials)
    if "fig" in only:
        log("Figures")
        res["figures"] = figures(corpus, schemes, a.out, root.child("fig"))

    write_tables(a.out, res, env, name, a.label_suffix)
    (a.out / "summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"Done in {time.perf_counter() - t_all:.1f} s. Results in {a.out.resolve()}")
    (a.out / "log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
