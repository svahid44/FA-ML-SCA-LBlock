
# ============================================================
# Stage 01 — LBlock-64/80 reference verification and tracing
#
# این مرحله فقط هسته مرجع الگوریتم را تثبیت می‌کند:
#   - بردارهای آزمون
#   - encrypt/decrypt تصادفی
#   - تریس تمام 32 دور
#   - ورودی و خروجی هشت S-box دور آخر
#   - قرارداد ثابت endian و شماره‌گذاری nibble
#
# هیچ fault، trace توان یا ML در این مرحله وجود ندارد.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import hashlib
import json
import platform
import random
import sys


BLOCK_SIZE_BITS = 64
KEY_SIZE_BITS = 80
NUM_ROUNDS = 32

MASK32 = 0xFFFFFFFF
MASK80 = (1 << 80) - 1


SBOX = [
    [0xE, 0x9, 0xF, 0x0, 0xD, 0x4, 0xA, 0xB,
     0x1, 0x2, 0x8, 0x3, 0x7, 0x6, 0xC, 0x5],
    [0x4, 0xB, 0xE, 0x9, 0xF, 0xD, 0x0, 0xA,
     0x7, 0xC, 0x5, 0x6, 0x2, 0x8, 0x1, 0x3],
    [0x1, 0xE, 0x7, 0xC, 0xF, 0xD, 0x0, 0x6,
     0xB, 0x5, 0x9, 0x3, 0x2, 0x4, 0x8, 0xA],
    [0x7, 0x6, 0x8, 0xB, 0x0, 0xF, 0x3, 0xE,
     0x9, 0xA, 0xC, 0xD, 0x5, 0x2, 0x4, 0x1],
    [0xE, 0x5, 0xF, 0x0, 0x7, 0x2, 0xC, 0xD,
     0x1, 0x8, 0x4, 0x9, 0xB, 0xA, 0x6, 0x3],
    [0x2, 0xD, 0xB, 0xC, 0xF, 0xE, 0x0, 0x9,
     0x7, 0xA, 0x6, 0x3, 0x1, 0x8, 0x4, 0x5],
    [0xB, 0x9, 0x4, 0xE, 0x0, 0xF, 0xA, 0xD,
     0x6, 0xC, 0x5, 0x7, 0x3, 0x8, 0x1, 0x2],
    [0xD, 0xA, 0xF, 0x0, 0xE, 0x4, 0x9, 0xB,
     0x2, 0x1, 0x8, 0x3, 0x7, 0x5, 0xC, 0x6],
]

S8 = [0x8, 0x7, 0xE, 0x5, 0xF, 0xD, 0x0, 0x6,
      0xB, 0xC, 0x9, 0xA, 0x2, 0x4, 0x1, 0x3]

S9 = [0xB, 0x5, 0xF, 0x0, 0x7, 0x2, 0x9, 0xD,
      0x4, 0x8, 0x1, 0xC, 0xE, 0xA, 0x3, 0x6]

# u[i] = z[P_SOURCE_FOR_OUTPUT[i]]
P_SOURCE_FOR_OUTPUT = [1, 3, 0, 2, 5, 7, 4, 6]

OFFICIAL_TEST_VECTORS = [
    {
        "name": "all_zero_vector",
        "plaintext_hex": "0000000000000000",
        "key_hex": "00000000000000000000",
        "ciphertext_hex": "c218185308e75bcd",
    },
    {
        "name": "incremental_vector",
        "plaintext_hex": "0123456789abcdef",
        "key_hex": "0123456789abcdeffedc",
        "ciphertext_hex": "4b7179d8ebee0c26",
    },
]


@dataclass(frozen=True)
class Stage01Config:
    random_roundtrip_tests: int = 2000
    random_seed: int = 20260718
    output_root: str = "runs/stage_01"
    save_full_trace_json: bool = True


@dataclass(frozen=True)
class FFunctionTrace:
    x: int
    round_key: int
    xor_input: int
    sbox_inputs: Tuple[int, ...]
    sbox_outputs: Tuple[int, ...]
    permuted_nibbles: Tuple[int, ...]
    output: int


@dataclass(frozen=True)
class RoundTrace:
    round_number: int
    x_prev2_index: int
    x_prev1_index: int
    x_new_index: int
    x_prev2: int
    x_prev1: int
    rotated_x_prev2: int
    round_key: int
    f_trace: FFunctionTrace
    x_new: int


@dataclass(frozen=True)
class EncryptionTrace:
    plaintext: int
    master_key: int
    x0: int
    x1: int
    ciphertext: int
    rounds: Tuple[RoundTrace, ...]


def rol(x: int, r: int, n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    r %= n
    mask = (1 << n) - 1
    x &= mask
    return x if r == 0 else ((x << r) | (x >> (n - r))) & mask


def ror(x: int, r: int, n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    r %= n
    mask = (1 << n) - 1
    x &= mask
    return x if r == 0 else ((x >> r) | (x << (n - r))) & mask


def int_to_bits(x: int, n: int) -> str:
    if not (0 <= x < (1 << n)):
        raise ValueError(f"x must fit in {n} bits")
    return format(x, f"0{n}b")


def bits_to_int(bits: str) -> int:
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("bits must contain only 0 and 1")
    return int(bits, 2)


def get_nibble(word: int, nibble_index: int) -> int:
    """nibble 0 = bits 3..0 and nibble 7 = bits 31..28."""
    if not 0 <= nibble_index < 8:
        raise ValueError("nibble_index must be in 0..7")
    return (word >> (4 * nibble_index)) & 0xF


def pack_nibbles(nibbles: Sequence[int]) -> int:
    if len(nibbles) != 8:
        raise ValueError("exactly 8 nibbles are required")
    out = 0
    for index, value in enumerate(nibbles):
        if not 0 <= int(value) <= 0xF:
            raise ValueError("nibble must be in 0..15")
        out |= int(value) << (4 * index)
    return out & MASK32


def hex_fixed(value: int, bits: int) -> str:
    return format(value & ((1 << bits) - 1), f"0{bits // 4}x")


def lblock_F(x: int, k: int) -> int:
    """F(X,K) = P(S(X XOR K))."""
    if not (0 <= x <= MASK32 and 0 <= k <= MASK32):
        raise ValueError("x and k must be 32-bit integers")

    xor_input = x ^ k
    z = [SBOX[i][get_nibble(xor_input, i)] for i in range(8)]
    u = [z[source] for source in P_SOURCE_FOR_OUTPUT]
    return pack_nibbles(u)


def lblock_F_traced(x: int, k: int) -> Tuple[int, FFunctionTrace]:
    """نسخه ابزارگذاری‌شده تابع F."""
    xor_input = x ^ k
    sbox_inputs = tuple(get_nibble(xor_input, i) for i in range(8))
    sbox_outputs = tuple(SBOX[i][sbox_inputs[i]] for i in range(8))
    permuted = tuple(sbox_outputs[source] for source in P_SOURCE_FOR_OUTPUT)
    output = pack_nibbles(permuted)

    return output, FFunctionTrace(
        x=x,
        round_key=k,
        xor_input=xor_input,
        sbox_inputs=sbox_inputs,
        sbox_outputs=sbox_outputs,
        permuted_nibbles=permuted,
        output=output,
    )


def key_schedule_lblock(master_key: int) -> List[int]:
    """تولید K1 تا K32 از کلید اصلی 80 بیتی."""
    if not 0 <= master_key < (1 << 80):
        raise ValueError("master_key must be an 80-bit integer")

    k = master_key
    round_keys = [(k >> 48) & MASK32]

    for i in range(1, 32):
        kb = int_to_bits(k, 80)
        kb = kb[29:] + kb[:29]

        top = S9[int(kb[0:4], 2)]
        nxt = S8[int(kb[4:8], 2)]
        kb = f"{top:04b}{nxt:04b}" + kb[8:]

        ib = f"{i:05b}"
        kb_list = list(kb)

        for offset, bit in enumerate(ib):
            index = 29 + offset
            kb_list[index] = "1" if kb_list[index] != bit else "0"

        k = bits_to_int("".join(kb_list)) & MASK80
        round_keys.append((k >> 48) & MASK32)

    if len(round_keys) != NUM_ROUNDS:
        raise RuntimeError("key schedule did not generate 32 keys")

    return round_keys


def encrypt_block_lblock(plaintext: int, master_key: int) -> int:
    if not 0 <= plaintext < (1 << 64):
        raise ValueError("plaintext must be a 64-bit integer")

    round_keys = key_schedule_lblock(master_key)
    x1 = (plaintext >> 32) & MASK32
    x0 = plaintext & MASK32
    x_prev2, x_prev1 = x0, x1

    for round_index in range(NUM_ROUNDS):
        x_new = (
            lblock_F(x_prev1, round_keys[round_index])
            ^ rol(x_prev2, 8, 32)
        ) & MASK32
        x_prev2, x_prev1 = x_prev1, x_new

    return (x_prev2 << 32) | x_prev1


def encrypt_block_lblock_traced(
    plaintext: int,
    master_key: int,
) -> Tuple[int, EncryptionTrace]:
    """رمزنگاری با ثبت state و S-boxهای تمام دورها."""
    if not 0 <= plaintext < (1 << 64):
        raise ValueError("plaintext must be a 64-bit integer")

    round_keys = key_schedule_lblock(master_key)
    x1 = (plaintext >> 32) & MASK32
    x0 = plaintext & MASK32
    x_prev2, x_prev1 = x0, x1
    rounds: List[RoundTrace] = []

    for round_index in range(NUM_ROUNDS):
        f_output, f_trace = lblock_F_traced(
            x_prev1,
            round_keys[round_index],
        )
        rotated = rol(x_prev2, 8, 32)
        x_new = (f_output ^ rotated) & MASK32

        rounds.append(RoundTrace(
            round_number=round_index + 1,
            x_prev2_index=round_index,
            x_prev1_index=round_index + 1,
            x_new_index=round_index + 2,
            x_prev2=x_prev2,
            x_prev1=x_prev1,
            rotated_x_prev2=rotated,
            round_key=round_keys[round_index],
            f_trace=f_trace,
            x_new=x_new,
        ))

        x_prev2, x_prev1 = x_prev1, x_new

    ciphertext = (x_prev2 << 32) | x_prev1

    return ciphertext, EncryptionTrace(
        plaintext=plaintext,
        master_key=master_key,
        x0=x0,
        x1=x1,
        ciphertext=ciphertext,
        rounds=tuple(rounds),
    )


def decrypt_block_lblock(ciphertext: int, master_key: int) -> int:
    if not 0 <= ciphertext < (1 << 64):
        raise ValueError("ciphertext must be a 64-bit integer")

    round_keys = key_schedule_lblock(master_key)
    x_jp1 = (ciphertext >> 32) & MASK32
    x_jp2 = ciphertext & MASK32

    for round_key in reversed(round_keys):
        x_j = ror(
            lblock_F(x_jp1, round_key) ^ x_jp2,
            8,
            32,
        ) & MASK32
        x_jp2, x_jp1 = x_jp1, x_j

    return (x_jp2 << 32) | x_jp1


def encrypt_lblock_block_bytes(pt8: bytes, key10: bytes) -> bytes:
    if len(pt8) != 8:
        raise ValueError("plaintext block must be exactly 8 bytes")
    if len(key10) != 10:
        raise ValueError("key must be exactly 10 bytes")

    ciphertext = encrypt_block_lblock(
        int.from_bytes(pt8, "big"),
        int.from_bytes(key10, "big"),
    )
    return ciphertext.to_bytes(8, "big")


def decrypt_lblock_block_bytes(ct8: bytes, key10: bytes) -> bytes:
    if len(ct8) != 8:
        raise ValueError("ciphertext block must be exactly 8 bytes")
    if len(key10) != 10:
        raise ValueError("key must be exactly 10 bytes")

    plaintext = decrypt_block_lblock(
        int.from_bytes(ct8, "big"),
        int.from_bytes(key10, "big"),
    )
    return plaintext.to_bytes(8, "big")


def validate_substitution_tables() -> Dict[str, Any]:
    expected = list(range(16))
    failures = []

    for index, table in enumerate(SBOX):
        if sorted(table) != expected:
            failures.append(f"SBOX[{index}] is not a permutation")

    if sorted(S8) != expected:
        failures.append("S8 is not a permutation")
    if sorted(S9) != expected:
        failures.append("S9 is not a permutation")

    return {"passed": not failures, "failures": failures}


def validate_p_layer() -> Dict[str, Any]:
    passed = sorted(P_SOURCE_FOR_OUTPUT) == list(range(8))
    return {
        "passed": passed,
        "mapping": P_SOURCE_FOR_OUTPUT,
        "failure": None if passed else "P mapping is not bijective",
    }


def validate_official_vectors() -> Dict[str, Any]:
    results = []

    for vector in OFFICIAL_TEST_VECTORS:
        pt = int(vector["plaintext_hex"], 16)
        key = int(vector["key_hex"], 16)
        expected = int(vector["ciphertext_hex"], 16)

        ct = encrypt_block_lblock(pt, key)
        dec = decrypt_block_lblock(ct, key)

        ct_bytes = encrypt_lblock_block_bytes(
            pt.to_bytes(8, "big"),
            key.to_bytes(10, "big"),
        )
        pt_bytes = decrypt_lblock_block_bytes(
            ct_bytes,
            key.to_bytes(10, "big"),
        )

        results.append({
            "name": vector["name"],
            "plaintext_hex": vector["plaintext_hex"],
            "key_hex": vector["key_hex"],
            "expected_ciphertext_hex": vector["ciphertext_hex"],
            "observed_ciphertext_hex": hex_fixed(ct, 64),
            "encryption_passed": ct == expected,
            "decryption_passed": dec == pt,
            "byte_encryption_passed": ct_bytes.hex() == vector["ciphertext_hex"],
            "byte_decryption_passed": pt_bytes == pt.to_bytes(8, "big"),
        })

    passed = all(
        row["encryption_passed"]
        and row["decryption_passed"]
        and row["byte_encryption_passed"]
        and row["byte_decryption_passed"]
        for row in results
    )

    return {"passed": passed, "vectors": results}


def validate_random_roundtrips(test_count: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    failures = []

    for test_index in range(test_count):
        pt = rng.getrandbits(64)
        key = rng.getrandbits(80)
        ct = encrypt_block_lblock(pt, key)
        dec = decrypt_block_lblock(ct, key)

        if dec != pt:
            failures.append({
                "test_index": test_index,
                "plaintext_hex": hex_fixed(pt, 64),
                "key_hex": hex_fixed(key, 80),
                "ciphertext_hex": hex_fixed(ct, 64),
                "recovered_hex": hex_fixed(dec, 64),
            })
            if len(failures) == 10:
                break

    return {
        "passed": not failures,
        "test_count": test_count,
        "seed": seed,
        "failures": failures,
    }


def validate_key_schedule() -> Dict[str, Any]:
    test_keys = [
        0,
        MASK80,
        int("0123456789abcdeffedc", 16),
        int("fedcba98765432100123", 16),
    ]
    failures = []
    summaries = []

    for key in test_keys:
        round_keys = key_schedule_lblock(key)

        if len(round_keys) != 32:
            failures.append(
                f"{hex_fixed(key, 80)} generated {len(round_keys)} keys"
            )
        if any(not 0 <= rk <= MASK32 for rk in round_keys):
            failures.append(
                f"{hex_fixed(key, 80)} generated an invalid round key"
            )

        summaries.append({
            "master_key_hex": hex_fixed(key, 80),
            "round_key_count": len(round_keys),
            "k1_hex": hex_fixed(round_keys[0], 32),
            "k32_hex": hex_fixed(round_keys[-1], 32),
            "distinct_round_keys": len(set(round_keys)),
        })

    return {
        "passed": not failures,
        "failures": failures,
        "summaries": summaries,
    }


def validate_trace_consistency() -> Dict[str, Any]:
    failures = []
    summaries = []

    for vector in OFFICIAL_TEST_VECTORS:
        pt = int(vector["plaintext_hex"], 16)
        key = int(vector["key_hex"], 16)

        normal_ct = encrypt_block_lblock(pt, key)
        traced_ct, trace = encrypt_block_lblock_traced(pt, key)

        if normal_ct != traced_ct:
            failures.append(f"{vector['name']}: traced ciphertext mismatch")

        if len(trace.rounds) != 32:
            failures.append(f"{vector['name']}: wrong trace length")

        for round_trace in trace.rounds:
            if lblock_F(
                round_trace.x_prev1,
                round_trace.round_key,
            ) != round_trace.f_trace.output:
                failures.append(
                    f"{vector['name']}: F mismatch in round "
                    f"{round_trace.round_number}"
                )

            reconstructed = (
                round_trace.f_trace.output
                ^ rol(round_trace.x_prev2, 8, 32)
            ) & MASK32

            if reconstructed != round_trace.x_new:
                failures.append(
                    f"{vector['name']}: recurrence mismatch in round "
                    f"{round_trace.round_number}"
                )

        last = trace.rounds[-1]
        x31 = last.x_prev2
        x32 = last.x_prev1
        x33 = last.x_new
        k32 = last.round_key

        if x33 != (
            lblock_F(x32, k32) ^ rol(x31, 8, 32)
        ) & MASK32:
            failures.append(
                f"{vector['name']}: final-round equation failed"
            )

        if ((traced_ct >> 32) & MASK32) != x32:
            failures.append(
                f"{vector['name']}: ciphertext left half is not X32"
            )

        if (traced_ct & MASK32) != x33:
            failures.append(
                f"{vector['name']}: ciphertext right half is not X33"
            )

        for sbox_index in range(8):
            expected_input = (
                get_nibble(x32, sbox_index)
                ^ get_nibble(k32, sbox_index)
            )
            expected_output = SBOX[sbox_index][expected_input]

            if last.f_trace.sbox_inputs[sbox_index] != expected_input:
                failures.append(
                    f"{vector['name']}: wrong S{sbox_index} input"
                )
            if last.f_trace.sbox_outputs[sbox_index] != expected_output:
                failures.append(
                    f"{vector['name']}: wrong S{sbox_index} output"
                )

        summaries.append({
            "name": vector["name"],
            "x31_hex": hex_fixed(x31, 32),
            "x32_hex": hex_fixed(x32, 32),
            "k32_hex": hex_fixed(k32, 32),
            "x32_xor_k32_hex": hex_fixed(x32 ^ k32, 32),
            "x33_hex": hex_fixed(x33, 32),
            "ciphertext_hex": hex_fixed(traced_ct, 64),
        })

    return {
        "passed": not failures,
        "failures": failures,
        "trace_summaries": summaries,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def f_trace_to_dict(trace: FFunctionTrace) -> Dict[str, Any]:
    return {
        "x_hex": hex_fixed(trace.x, 32),
        "round_key_hex": hex_fixed(trace.round_key, 32),
        "xor_input_hex": hex_fixed(trace.xor_input, 32),
        "sbox_inputs": list(trace.sbox_inputs),
        "sbox_inputs_hex": [f"{v:x}" for v in trace.sbox_inputs],
        "sbox_outputs": list(trace.sbox_outputs),
        "sbox_outputs_hex": [f"{v:x}" for v in trace.sbox_outputs],
        "permuted_nibbles": list(trace.permuted_nibbles),
        "output_hex": hex_fixed(trace.output, 32),
    }


def trace_to_dict(trace: EncryptionTrace) -> Dict[str, Any]:
    return {
        "plaintext_hex": hex_fixed(trace.plaintext, 64),
        "master_key_hex": hex_fixed(trace.master_key, 80),
        "x0_hex": hex_fixed(trace.x0, 32),
        "x1_hex": hex_fixed(trace.x1, 32),
        "ciphertext_hex": hex_fixed(trace.ciphertext, 64),
        "rounds": [
            {
                "round_number": row.round_number,
                "x_prev2_index": row.x_prev2_index,
                "x_prev1_index": row.x_prev1_index,
                "x_new_index": row.x_new_index,
                "x_prev2_hex": hex_fixed(row.x_prev2, 32),
                "x_prev1_hex": hex_fixed(row.x_prev1, 32),
                "rotated_x_prev2_hex": hex_fixed(
                    row.rotated_x_prev2,
                    32,
                ),
                "round_key_hex": hex_fixed(row.round_key, 32),
                "f_trace": f_trace_to_dict(row.f_trace),
                "x_new_hex": hex_fixed(row.x_new, 32),
            }
            for row in trace.rounds
        ],
    }


def write_round_trace_csv(path: Path, trace: EncryptionTrace) -> None:
    fields = [
        "round_number",
        "x_prev2_index",
        "x_prev1_index",
        "x_new_index",
        "x_prev2_hex",
        "x_prev1_hex",
        "round_key_hex",
        "xor_input_hex",
        "sbox_inputs_hex_high_to_low",
        "sbox_outputs_hex_high_to_low",
        "p_output_hex",
        "rotated_x_prev2_hex",
        "x_new_hex",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in trace.rounds:
            writer.writerow({
                "round_number": row.round_number,
                "x_prev2_index": row.x_prev2_index,
                "x_prev1_index": row.x_prev1_index,
                "x_new_index": row.x_new_index,
                "x_prev2_hex": hex_fixed(row.x_prev2, 32),
                "x_prev1_hex": hex_fixed(row.x_prev1, 32),
                "round_key_hex": hex_fixed(row.round_key, 32),
                "xor_input_hex": hex_fixed(row.f_trace.xor_input, 32),
                "sbox_inputs_hex_high_to_low": "".join(
                    f"{v:x}" for v in reversed(row.f_trace.sbox_inputs)
                ),
                "sbox_outputs_hex_high_to_low": "".join(
                    f"{v:x}" for v in reversed(row.f_trace.sbox_outputs)
                ),
                "p_output_hex": hex_fixed(row.f_trace.output, 32),
                "rotated_x_prev2_hex": hex_fixed(
                    row.rotated_x_prev2,
                    32,
                ),
                "x_new_hex": hex_fixed(row.x_new, 32),
            })


def write_last_round_sbox_csv(path: Path, trace: EncryptionTrace) -> None:
    last = trace.rounds[-1]
    inverse_p_position = {
        source: output
        for output, source in enumerate(P_SOURCE_FOR_OUTPUT)
    }

    fields = [
        "round_number",
        "target_sbox",
        "x32_nibble_hex",
        "k32_nibble_hex",
        "sbox_input_hex",
        "sbox_output_hex",
        "p_output_nibble_position",
        "x32_hex",
        "x33_hex",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for sbox_index in range(8):
            writer.writerow({
                "round_number": 32,
                "target_sbox": sbox_index,
                "x32_nibble_hex": f"{get_nibble(last.x_prev1, sbox_index):x}",
                "k32_nibble_hex": f"{get_nibble(last.round_key, sbox_index):x}",
                "sbox_input_hex": f"{last.f_trace.sbox_inputs[sbox_index]:x}",
                "sbox_output_hex": f"{last.f_trace.sbox_outputs[sbox_index]:x}",
                "p_output_nibble_position": inverse_p_position[sbox_index],
                "x32_hex": hex_fixed(last.x_prev1, 32),
                "x33_hex": hex_fixed(last.x_new, 32),
            })


def build_schema() -> Dict[str, Any]:
    return {
        "algorithm": "LBlock-64/80",
        "block_size_bits": 64,
        "master_key_size_bits": 80,
        "round_count": 32,
        "external_byte_order": "big-endian",
        "internal_nibble_numbering": {
            "nibble_0": "bits 3..0",
            "nibble_7": "bits 31..28",
        },
        "plaintext_layout": "M = X1 || X0",
        "ciphertext_layout": "C = X32 || X33",
        "round_equation": (
            "X_i = F(X_{i-1},K_{i-1}) XOR ROL8(X_{i-2}), i=2..33"
        ),
        "f_equation": "F(X,K)=P(S(X XOR K))",
        "p_source_for_output": P_SOURCE_FOR_OUTPUT,
        "last_round": {
            "round_number": 32,
            "inputs": ["X31", "X32", "K32"],
            "sbox_input_equation": (
                "input_i = nibble_i(X32) XOR nibble_i(K32)"
            ),
            "sbox_output_equation": (
                "output_i = SBOX[i][input_i]"
            ),
            "x33_equation": (
                "X33 = P(S(X32 XOR K32)) XOR ROL8(X31)"
            ),
            "observable_relation": (
                "X32 is the left 32-bit half of the correct ciphertext"
            ),
            "key_guess_relation": (
                "input_i(k)=nibble_i(X32) XOR k"
            ),
        },
    }


def run_stage_01(
    config: Optional[Stage01Config] = None,
) -> Dict[str, Any]:
    if config is None:
        config = Stage01Config()

    if config.random_roundtrip_tests <= 0:
        raise ValueError("random_roundtrip_tests must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"stage01_{timestamp}_seed{config.random_seed}"
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    tests = {
        "substitution_tables": validate_substitution_tables(),
        "p_layer": validate_p_layer(),
        "official_vectors": validate_official_vectors(),
        "key_schedule": validate_key_schedule(),
        "trace_consistency": validate_trace_consistency(),
        "random_roundtrips": validate_random_roundtrips(
            config.random_roundtrip_tests,
            config.random_seed,
        ),
    }
    all_passed = all(result["passed"] for result in tests.values())
    tests["all_passed"] = all_passed

    example = OFFICIAL_TEST_VECTORS[0]
    example_ct, example_trace = encrypt_block_lblock_traced(
        int(example["plaintext_hex"], 16),
        int(example["key_hex"], 16),
    )

    write_json(
        run_dir / "reference_vectors.json",
        tests["official_vectors"],
    )
    write_json(
        run_dir / "unit_test_results.json",
        tests,
    )
    write_json(
        run_dir / "lblock_schema.json",
        build_schema(),
    )

    manifest_config = json.dumps(
        asdict(config),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    write_json(
        run_dir / "run_manifest.json",
        {
            "stage": 1,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config": asdict(config),
            "config_sha256": hashlib.sha256(
                manifest_config
            ).hexdigest(),
        },
    )

    write_round_trace_csv(
        run_dir / "round_trace_example.csv",
        example_trace,
    )
    write_last_round_sbox_csv(
        run_dir / "last_round_sbox_trace.csv",
        example_trace,
    )

    if config.save_full_trace_json:
        write_json(
            run_dir / "round_trace_example.json",
            trace_to_dict(example_trace),
        )

    summary = {
        "stage": 1,
        "run_id": run_id,
        "run_directory": str(run_dir.resolve()),
        "all_tests_passed": all_passed,
        "official_vectors_passed": tests["official_vectors"]["passed"],
        "trace_consistency_passed": tests["trace_consistency"]["passed"],
        "random_roundtrip_tests": config.random_roundtrip_tests,
        "random_roundtrip_passed": tests["random_roundtrips"]["passed"],
        "example_ciphertext_hex": hex_fixed(example_ct, 64),
    }

    write_json(
        run_dir / "stage_01_summary.json",
        summary,
    )

    print("\n" + "=" * 72)
    print("Stage 01 complete: LBlock reference verification and tracing")
    print("=" * 72)
    print("Run directory           :", summary["run_directory"])
    print("All tests passed        :", summary["all_tests_passed"])
    print("Official vectors passed :", summary["official_vectors_passed"])
    print("Trace consistency       :", summary["trace_consistency_passed"])
    print("Random roundtrips       :", summary["random_roundtrip_tests"])
    print("Random tests passed     :", summary["random_roundtrip_passed"])
    print("Example ciphertext      :", summary["example_ciphertext_hex"])
    print("=" * 72)

    print("\nGenerated files:")
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            print("  -", path.name)

    if not all_passed:
        raise AssertionError(
            "Stage 01 failed; inspect unit_test_results.json"
        )

    return summary


def load_stage_01_config(config_path: str | Path) -> Stage01Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return Stage01Config(**raw)


if __name__ == "__main__":
    run_stage_01()
