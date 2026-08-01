# ============================================================
# Stage 02 — Full power-trace simulator for LBlock-64/80
#
# هدف این مرحله:
#   1) تولید تریس کامل اجرای هر 32 دور LBlock
#   2) استفاده از یک trigger عمومی در ابتدای رمزنگاری
#   3) عدم تولید trigger جداگانه برای S-boxها
#   4) ایجاد نشتی داده‌محور HW/HD همراه با نویز، jitter و drift
#   5) ذخیره زمان واقعی رویدادها فقط در بخش private_ground_truth
#
# فایل‌های public ورودی مراحل Timing Map خواهند بود.
# فایل‌های private_ground_truth فقط برای ارزیابی صحت استفاده می‌شوند و
# الگوریتم Timing Map مجاز نیست زمان‌های خود را از آن‌ها بخواند.
# ============================================================

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import csv
import hashlib
import importlib
import json
import math
import platform
import sys
import time

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency error is explicit
    raise ImportError(
        "Stage 02 requires matplotlib. Install it with: pip install matplotlib"
    ) from exc


# Stage 02 عمداً هسته رمز را دوباره تعریف نمی‌کند.
# تمام عملیات رمزنگاری از فایل تأییدشده Stage 01 وارد می‌شوند.
#
# این loader به مسیر جاری Jupyter وابسته نیست. ابتدا مسیر صریح Stage 01
# را از environment variable می‌گیرد و سپس مسیرهای شناخته‌شده پروژه را
# بررسی می‌کند.

def load_verified_stage_01_module(
    explicit_source: str | Path | None = None,
):
    import importlib.util
    import os

    candidates: List[Path] = []

    # 1) مسیر صریحی که از نوت‌بوک یا terminal تنظیم می‌شود.
    if explicit_source is not None:
        candidates.append(Path(explicit_source).expanduser())

    env_source = os.environ.get("LBLOCK_STAGE1_SOURCE")
    if env_source:
        candidates.append(Path(env_source).expanduser())

    env_run_dir = os.environ.get("LBLOCK_STAGE1_RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir).expanduser()
        candidates.append(run_dir / "stage_01_lblock_reference.py")

        # برای حالتی که فایل در زیرپوشه‌ای از run directory قرار گرفته باشد.
        if run_dir.is_dir():
            candidates.extend(
                sorted(
                    run_dir.rglob("stage_01_lblock_reference.py"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )

    # 2) مسیر دقیق پروژه فعلی کاربر.
    default_project_root = Path(
        '.'
    )
    default_stage1_run = (
        default_project_root
        / "runs"
        / "stage_01"
        / "stage01_20260718_160320_087778_seed20260718"
    )

    candidates.append(
        default_stage1_run / "stage_01_lblock_reference.py"
    )
    candidates.append(
        default_project_root / "stage_01_lblock_reference.py"
    )

    # 3) تمام runهای Stage 01، از جدیدترین به قدیمی‌ترین.
    stage1_runs_root = default_project_root / "runs" / "stage_01"

    if stage1_runs_root.is_dir():
        for run_directory in sorted(
            stage1_runs_root.iterdir(),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            if run_directory.is_dir():
                candidates.append(
                    run_directory / "stage_01_lblock_reference.py"
                )

    # 4) پوشه فایل Stage 02 و پوشه جاری Jupyter.
    if "__file__" in globals():
        candidates.append(
            Path(__file__).resolve().parent
            / "stage_01_lblock_reference.py"
        )

    current_directory = Path.cwd().resolve()
    candidates.append(
        current_directory / "stage_01_lblock_reference.py"
    )

    # 5) برای تست بسته در سیستم‌های دیگر.
    candidates.append(
        Path(__file__).resolve().parent
        / "stage_01_lblock_reference.py"
        if "__file__" in globals()
        else current_directory / "stage_01_lblock_reference.py"
    )

    # حذف تکراری‌ها با حفظ ترتیب.
    unique_candidates: List[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        candidate = candidate.resolve()
        candidate_string = str(candidate)

        if candidate_string not in seen:
            seen.add(candidate_string)
            unique_candidates.append(candidate)

    searched: List[str] = []

    for candidate in unique_candidates:
        searched.append(str(candidate))

        if not candidate.is_file():
            continue

        spec = importlib.util.spec_from_file_location(
            "stage_01_lblock_reference",
            candidate,
        )

        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        verification = module.validate_official_vectors()

        if not verification.get("passed", False):
            raise RuntimeError(
                "Stage 01 was found, but its official-vector "
                "verification failed."
            )

        print("Loaded verified Stage 01 core from:")
        print(" ", candidate)

        return module

    raise ImportError(
        "The verified Stage 01 Python source was not found.\n\n"
        "Expected file name:\n"
        "  stage_01_lblock_reference.py\n\n"
        "Configured Stage 01 run directory:\n"
        '  ./runs/stage_01/'
        "stage01_20260718_160320_087778_seed20260718\n\n"
        "Searched paths:\n  - "
        + "\n  - ".join(searched)
    )


lblock = load_verified_stage_01_module()


MASK64 = (1 << 64) - 1


# ============================================================
# 1. تنظیمات شبیه‌ساز
# ============================================================


def _default_base_amplitudes() -> Dict[str, float]:
    return {
        "xor": 0.30,
        "sbox": 0.55,
        "permutation": 0.35,
        "rotate": 0.28,
        "mix": 0.42,
        "write": 0.32,
    }


def _default_hw_gains() -> Dict[str, float]:
    return {
        "xor": 0.42,
        "sbox": 0.95,
        "permutation": 0.30,
        "rotate": 0.18,
        "mix": 0.38,
        "write": 0.35,
    }


def _default_hd_gains() -> Dict[str, float]:
    return {
        "xor": 0.25,
        "sbox": 0.42,
        "permutation": 0.34,
        "rotate": 0.26,
        "mix": 0.42,
        "write": 0.30,
    }


@dataclass(frozen=True)
class Stage02Config:
    """تنظیمات قابل بازتولید شبیه‌ساز تریس توان."""

    # حجم مجموعه profiling مرحله Timing Map
    number_of_traces: int = 1024
    number_of_keys: int = 4
    number_of_sessions: int = 4
    random_seed: int = 20260718
    output_root: str = "runs/stage_02"

    # نرخ نمونه‌برداری فقط برای تبدیل sample به زمان فیزیکی استفاده می‌شود.
    sampling_rate_hz: float = 100_000_000.0

    # تنها trigger عمومی در sample زیر ثبت می‌شود.
    global_trigger_sample: int = 96
    pretrigger_samples: int = 96
    posttrigger_samples: int = 96

    # زمان‌بندی اسمی هر دور. زمان واقعی با jitter و session drift تغییر می‌کند.
    round_duration_samples: int = 128
    xor_offset_samples: int = 8
    sbox_start_offset_samples: int = 20
    sbox_spacing_samples: int = 9
    permutation_offset_samples: int = 96
    rotate_offset_samples: int = 106
    mix_offset_samples: int = 115
    write_offset_samples: int = 122

    # عدم قطعیت زمانی. این مقادیر زمان عملیات را نسبت به trigger عمومی جابه‌جا می‌کنند.
    trace_jitter_std_samples: float = 1.50
    round_jitter_std_samples: float = 0.30
    event_jitter_std_samples: float = 0.45
    session_timing_shift_std_samples: float = 1.50
    clock_scale_std: float = 0.0008

    # نویز و تغییرات acquisition/session
    gaussian_noise_std: float = 0.50
    impulsive_noise_probability: float = 0.0008
    impulsive_noise_scale: float = 2.20
    baseline_offset_std: float = 0.18
    baseline_slope_std: float = 0.10
    low_frequency_drift_std: float = 0.11
    session_gain_std: float = 0.10
    trace_gain_std: float = 0.035
    session_noise_scale_std: float = 0.10
    clock_artifact_amplitude: float = 0.035
    clock_artifact_period_samples: float = 8.0

    # ضرایب مدل نشتی. HW و HD به بازه تقریباً [-1, +1] نرمال می‌شوند.
    operation_base_amplitudes: Dict[str, float] = field(
        default_factory=_default_base_amplitudes
    )
    operation_hw_gains: Dict[str, float] = field(
        default_factory=_default_hw_gains
    )
    operation_hd_gains: Dict[str, float] = field(
        default_factory=_default_hd_gains
    )

    # شعاع template هر رویداد. طول template برابر 2*radius+1 است.
    pulse_radius_samples: int = 7

    # ترکیب plaintextها: random، walking-bit و الگوهای ساختاری
    plaintext_strategy: str = "mixed"

    # ذخیره اطلاعات داخلی تمام دورها فقط در فایل private.
    save_full_crypto_ground_truth: bool = True

    # رسم شکل‌های sanity-check
    save_plots: bool = True


# ============================================================
# 2. ساختارهای کمکی
# ============================================================


@dataclass(frozen=True)
class SessionProfile:
    """ویژگی‌های ثابت یک جلسه acquisition شبیه‌سازی‌شده."""

    session_id: int
    gain: float
    noise_scale: float
    baseline_offset: float
    baseline_slope: float
    timing_shift_samples: int
    clock_scale: float
    clock_phase: float
    drift_phase: float


@dataclass(frozen=True)
class TraceAssignment:
    """اختصاص متوازن trace به key و session."""

    trace_id: int
    key_id: int
    session_id: int


# ============================================================
# 3. اعتبارسنجی config
# ============================================================


def validate_config(config: Stage02Config) -> None:
    """کنترل سازگاری پارامترهای زمانی و عددی پیش از شروع شبیه‌سازی."""

    if config.number_of_traces <= 0:
        raise ValueError("number_of_traces must be positive")

    if config.number_of_keys <= 0:
        raise ValueError("number_of_keys must be positive")

    if config.number_of_sessions <= 0:
        raise ValueError("number_of_sessions must be positive")

    if config.sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    if config.pretrigger_samples < 0 or config.posttrigger_samples < 0:
        raise ValueError("pretrigger_samples and posttrigger_samples cannot be negative")

    if config.global_trigger_sample != config.pretrigger_samples:
        raise ValueError(
            "For Stage 02, global_trigger_sample must equal pretrigger_samples. "
            "This keeps a single, explicit acquisition reference."
        )

    if config.round_duration_samples <= 0:
        raise ValueError("round_duration_samples must be positive")

    sbox_last_offset = (
        config.sbox_start_offset_samples
        + 7 * config.sbox_spacing_samples
    )

    ordered_offsets = [
        config.xor_offset_samples,
        config.sbox_start_offset_samples,
        sbox_last_offset,
        config.permutation_offset_samples,
        config.rotate_offset_samples,
        config.mix_offset_samples,
        config.write_offset_samples,
    ]

    if ordered_offsets != sorted(ordered_offsets):
        raise ValueError(
            "Nominal event offsets must be ordered as XOR, S-boxes, P, rotate, mix, write"
        )

    if config.write_offset_samples >= config.round_duration_samples:
        raise ValueError("write_offset_samples must be inside the round")

    if config.pulse_radius_samples < 2:
        raise ValueError("pulse_radius_samples must be at least 2")

    if config.gaussian_noise_std < 0:
        raise ValueError("gaussian_noise_std cannot be negative")

    if not (0 <= config.impulsive_noise_probability <= 1):
        raise ValueError("impulsive_noise_probability must be in [0, 1]")

    required_operations = {
        "xor", "sbox", "permutation", "rotate", "mix", "write"
    }

    for mapping_name, mapping in [
        ("operation_base_amplitudes", config.operation_base_amplitudes),
        ("operation_hw_gains", config.operation_hw_gains),
        ("operation_hd_gains", config.operation_hd_gains),
    ]:
        missing = required_operations.difference(mapping)
        if missing:
            raise ValueError(f"{mapping_name} misses operations: {sorted(missing)}")

    if config.plaintext_strategy not in {"mixed", "random"}:
        raise ValueError("plaintext_strategy must be 'mixed' or 'random'")


# ============================================================
# 4. ابزارهای عددی و مدل نشتی
# ============================================================


def hamming_weight(value: int) -> int:
    """تعداد بیت‌های یک در نمایش دودویی value."""

    return int(value).bit_count()


def normalized_hw(value: int, bit_width: int) -> float:
    """نگاشت HW از [0, bit_width] به تقریباً [-1, +1]."""

    if bit_width <= 0:
        raise ValueError("bit_width must be positive")

    return 2.0 * hamming_weight(value) / bit_width - 1.0


def normalized_hd(value_a: int, value_b: int, bit_width: int) -> float:
    """نگاشت HD دو مقدار به تقریباً [-1, +1]."""

    return normalized_hw(value_a ^ value_b, bit_width)


def safe_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """محاسبه correlation بدون تولید NaN برای بردارهای ثابت."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size != y.size or x.size < 3:
        return 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered))
        * float(np.dot(y_centered, y_centered))
    )

    if denominator <= 1e-15:
        return 0.0

    return float(np.dot(x_centered, y_centered) / denominator)


def operation_template(operation: str, radius: int) -> np.ndarray:
    """
    ساخت waveform کوتاه برای یک رویداد داخلی.

    templateها عمداً شبیه هم ولی کاملاً یکسان نیستند. این کار باعث می‌شود
    تریس فقط مجموعه‌ای از spikeهای ایده‌آل و غیرواقعی نباشد.
    """

    x = np.arange(-radius, radius + 1, dtype=np.float64)
    scale = max(radius / 2.4, 1.0)
    gaussian = np.exp(-0.5 * (x / scale) ** 2)

    if operation == "sbox":
        # قله مرکزی مثبت و دو side-lobe ضعیف
        template = gaussian - 0.24 * np.exp(-0.5 * ((x - 3.0) / 1.5) ** 2)
        template -= 0.16 * np.exp(-0.5 * ((x + 3.0) / 1.7) ** 2)
    elif operation == "xor":
        template = gaussian * (1.0 - 0.12 * x)
    elif operation == "permutation":
        template = (
            0.72 * np.exp(-0.5 * ((x + 1.8) / 2.0) ** 2)
            + 0.52 * np.exp(-0.5 * ((x - 2.0) / 2.2) ** 2)
        )
    elif operation == "rotate":
        template = gaussian * np.cos(0.38 * x)
    elif operation == "mix":
        template = gaussian - 0.28 * np.exp(-0.5 * ((x - 3.2) / 1.8) ** 2)
    elif operation == "write":
        template = gaussian * (1.0 + 0.10 * x)
    else:
        raise ValueError(f"unknown operation: {operation}")

    peak = float(np.max(np.abs(template)))
    if peak <= 0:
        raise RuntimeError("generated an empty operation template")

    return (template / peak).astype(np.float64)


def add_template(
    signal: np.ndarray,
    center: int,
    template: np.ndarray,
    amplitude: float,
) -> None:
    """افزودن template به signal با مدیریت امن مرزهای آرایه."""

    radius = len(template) // 2
    start = center - radius
    end = center + radius + 1

    signal_start = max(start, 0)
    signal_end = min(end, signal.size)

    if signal_start >= signal_end:
        return

    template_start = signal_start - start
    template_end = template_start + (signal_end - signal_start)

    signal[signal_start:signal_end] += (
        amplitude * template[template_start:template_end]
    )


def event_amplitude(
    operation: str,
    hw_term: float,
    hd_term: float,
    config: Stage02Config,
) -> float:
    """محاسبه دامنه رویداد از مدل base + alpha*HW + beta*HD."""

    return (
        config.operation_base_amplitudes[operation]
        + config.operation_hw_gains[operation] * hw_term
        + config.operation_hd_gains[operation] * hd_term
    )


# ============================================================
# 5. تولید کلید، session و plaintext
# ============================================================


def make_key_set(config: Stage02Config) -> List[int]:
    """
    تولید مجموعه کلیدها.

    دو کلید اول، در صورت امکان، کلیدهای بردارهای مرجع هستند. بقیه کلیدها
    به‌صورت تصادفی ولی بازتولیدپذیر تولید می‌شوند.
    """

    rng = np.random.default_rng(config.random_seed ^ 0x4B455953)

    keys: List[int] = []

    reference_keys = [
        int("00000000000000000000", 16),
        int("0123456789abcdeffedc", 16),
    ]

    for key in reference_keys:
        if len(keys) < config.number_of_keys:
            keys.append(key)

    while len(keys) < config.number_of_keys:
        # NumPy مستقیماً uint80 ندارد؛ کلید از 64 بیت پایین و 16 بیت بالا ساخته می‌شود.
        low64 = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        high16 = int(rng.integers(0, 1 << 16, dtype=np.uint32))
        candidate = ((high16 << 64) | low64) & lblock.MASK80

        if candidate not in keys:
            keys.append(candidate)

    return keys


def make_session_profiles(config: Stage02Config) -> List[SessionProfile]:
    """تولید پروفایل ثابت و بازتولیدپذیر برای هر session."""

    rng = np.random.default_rng(config.random_seed ^ 0x53455353)
    profiles: List[SessionProfile] = []

    for session_id in range(config.number_of_sessions):
        gain = max(0.60, 1.0 + rng.normal(0.0, config.session_gain_std))
        noise_scale = max(
            0.50,
            1.0 + rng.normal(0.0, config.session_noise_scale_std),
        )
        baseline_offset = float(
            rng.normal(0.0, config.baseline_offset_std)
        )
        baseline_slope = float(
            rng.normal(0.0, config.baseline_slope_std)
        )
        timing_shift = int(round(
            rng.normal(0.0, config.session_timing_shift_std_samples)
        ))
        clock_scale = max(
            0.995,
            1.0 + rng.normal(0.0, config.clock_scale_std),
        )

        profiles.append(
            SessionProfile(
                session_id=session_id,
                gain=float(gain),
                noise_scale=float(noise_scale),
                baseline_offset=baseline_offset,
                baseline_slope=baseline_slope,
                timing_shift_samples=timing_shift,
                clock_scale=float(clock_scale),
                clock_phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                drift_phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            )
        )

    return profiles


def make_balanced_assignments(config: Stage02Config) -> List[TraceAssignment]:
    """اختصاص تقریباً متوازن traceها به تمام زوج‌های key/session."""

    pairs = [
        (key_id, session_id)
        for key_id in range(config.number_of_keys)
        for session_id in range(config.number_of_sessions)
    ]

    repeated_pairs = [
        pairs[index % len(pairs)]
        for index in range(config.number_of_traces)
    ]

    rng = np.random.default_rng(config.random_seed ^ 0x41535349)
    permutation = rng.permutation(config.number_of_traces)

    shuffled_pairs = [repeated_pairs[int(index)] for index in permutation]

    return [
        TraceAssignment(
            trace_id=trace_id,
            key_id=key_id,
            session_id=session_id,
        )
        for trace_id, (key_id, session_id) in enumerate(shuffled_pairs)
    ]


def generate_plaintext(
    trace_id: int,
    rng: np.random.Generator,
    strategy: str,
) -> Tuple[int, str]:
    """
    تولید plaintext متنوع.

    در حالت mixed، 12 نمونه از هر 16 نمونه random هستند و چهار نمونه دیگر
    الگوهای walking-one، walking-zero، repeated-nibble و counter هستند.
    """

    if strategy == "random":
        plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        return plaintext, "random"

    mode = trace_id % 16

    if mode < 12:
        plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        return plaintext, "random"

    if mode == 12:
        bit_index = (trace_id // 16) % 64
        return (1 << bit_index) & MASK64, "walking_one"

    if mode == 13:
        bit_index = (trace_id // 16) % 64
        return (MASK64 ^ (1 << bit_index)), "walking_zero"

    if mode == 14:
        nibble = (trace_id // 16) & 0xF
        plaintext = 0
        for nibble_index in range(16):
            plaintext |= nibble << (4 * nibble_index)
        return plaintext & MASK64, "repeated_nibble"

    # یک counter ضرب‌شده در ثابت فرد برای پراکندگی بهتر بیت‌ها
    plaintext = (trace_id * 0x9E3779B97F4A7C15) & MASK64
    return plaintext, "counter"


# ============================================================
# 6. زمان‌بندی و شبیه‌سازی یک trace
# ============================================================


def trace_length_samples(config: Stage02Config) -> int:
    """طول کل trace شامل pre-trigger، 32 دور و post-trigger."""

    return (
        config.pretrigger_samples
        + lblock.NUM_ROUNDS * config.round_duration_samples
        + config.posttrigger_samples
    )


def make_trace_rng(seed: int, trace_id: int) -> np.random.Generator:
    """RNG مستقل هر trace برای بازتولیدپذیری و امکان parallelization آینده."""

    seed_sequence = np.random.SeedSequence([seed, trace_id, 0x54524143])
    return np.random.default_rng(seed_sequence)


def simulate_one_trace(
    *,
    trace_id: int,
    key_id: int,
    session: SessionProfile,
    master_key: int,
    config: Stage02Config,
    templates: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray | int | float | str]]:
    """
    تولید یک trace سالم همراه با metadata عمومی و ground truth خصوصی.

    هیچ زمان S-box یا state داخلی وارد metadata عمومی نمی‌شود.
    """

    rng = make_trace_rng(config.random_seed, trace_id)

    plaintext, plaintext_class = generate_plaintext(
        trace_id,
        rng,
        config.plaintext_strategy,
    )

    ciphertext, crypto_trace = lblock.encrypt_block_lblock_traced(
        plaintext,
        master_key,
    )

    number_of_samples = trace_length_samples(config)
    sample_index = np.arange(number_of_samples, dtype=np.float64)
    normalized_axis = sample_index / max(number_of_samples - 1, 1)

    # --------------------------
    # پس‌زمینه acquisition
    # --------------------------

    trace_gain = max(0.75, 1.0 + rng.normal(0.0, config.trace_gain_std))
    effective_gain = session.gain * trace_gain

    baseline = (
        session.baseline_offset
        + session.baseline_slope * (normalized_axis - 0.5)
    )

    low_frequency_drift = (
        config.low_frequency_drift_std
        * np.sin(
            2.0 * math.pi * (0.65 * normalized_axis)
            + session.drift_phase
            + rng.normal(0.0, 0.08)
        )
    )

    clock_artifact = (
        config.clock_artifact_amplitude
        * np.sin(
            2.0 * math.pi * sample_index / config.clock_artifact_period_samples
            + session.clock_phase
        )
    )

    clean_signal = baseline + low_frequency_drift + clock_artifact

    # --------------------------
    # زمان‌بندی واقعی ولی مخفی
    # --------------------------

    trace_shift = int(round(
        rng.normal(0.0, config.trace_jitter_std_samples)
    ))

    encryption_start = (
        config.global_trigger_sample
        + session.timing_shift_samples
        + trace_shift
    )

    trace_clock_scale = max(
        0.992,
        session.clock_scale
        * (1.0 + rng.normal(0.0, config.clock_scale_std / 2.0)),
    )

    xor_centers = np.empty(lblock.NUM_ROUNDS, dtype=np.int16)
    sbox_centers = np.empty((lblock.NUM_ROUNDS, 8), dtype=np.int16)
    permutation_centers = np.empty(lblock.NUM_ROUNDS, dtype=np.int16)
    rotate_centers = np.empty(lblock.NUM_ROUNDS, dtype=np.int16)
    mix_centers = np.empty(lblock.NUM_ROUNDS, dtype=np.int16)
    write_centers = np.empty(lblock.NUM_ROUNDS, dtype=np.int16)

    sbox_inputs = np.empty((lblock.NUM_ROUNDS, 8), dtype=np.uint8)
    sbox_outputs = np.empty((lblock.NUM_ROUNDS, 8), dtype=np.uint8)

    x_prev2_values = np.empty(lblock.NUM_ROUNDS, dtype=np.uint32)
    x_prev1_values = np.empty(lblock.NUM_ROUNDS, dtype=np.uint32)
    xor_words = np.empty(lblock.NUM_ROUNDS, dtype=np.uint32)
    f_outputs = np.empty(lblock.NUM_ROUNDS, dtype=np.uint32)
    x_new_values = np.empty(lblock.NUM_ROUNDS, dtype=np.uint32)
    round_keys = np.empty(lblock.NUM_ROUNDS, dtype=np.uint32)

    round_random_walk = 0.0

    for round_index, round_trace in enumerate(crypto_trace.rounds):
        round_random_walk += rng.normal(
            0.0,
            config.round_jitter_std_samples,
        )

        nominal_round_start = (
            encryption_start
            + round_index * config.round_duration_samples * trace_clock_scale
            + round_random_walk
        )

        def actual_center(offset: int) -> int:
            local_jitter = rng.normal(
                0.0,
                config.event_jitter_std_samples,
            )
            return int(round(nominal_round_start + offset + local_jitter))

        # مقادیر رمزنگاری این دور
        x_prev2 = int(round_trace.x_prev2)
        x_prev1 = int(round_trace.x_prev1)
        round_key = int(round_trace.round_key)
        xor_word = int(round_trace.f_trace.xor_input)
        f_output = int(round_trace.f_trace.output)
        rotated_x_prev2 = int(round_trace.rotated_x_prev2)
        x_new = int(round_trace.x_new)

        x_prev2_values[round_index] = x_prev2
        x_prev1_values[round_index] = x_prev1
        xor_words[round_index] = xor_word
        f_outputs[round_index] = f_output
        x_new_values[round_index] = x_new
        round_keys[round_index] = round_key

        sbox_inputs[round_index, :] = np.asarray(
            round_trace.f_trace.sbox_inputs,
            dtype=np.uint8,
        )
        sbox_outputs[round_index, :] = np.asarray(
            round_trace.f_trace.sbox_outputs,
            dtype=np.uint8,
        )

        # XOR X32-like state with round key
        xor_center = actual_center(config.xor_offset_samples)
        xor_centers[round_index] = xor_center
        xor_amplitude = event_amplitude(
            "xor",
            normalized_hw(xor_word, 32),
            normalized_hd(x_prev1, xor_word, 32),
            config,
        )
        add_template(
            clean_signal,
            xor_center,
            templates["xor"],
            effective_gain * xor_amplitude,
        )

        # هشت S-box به‌صورت ترتیبی و بدون trigger اختصاصی
        for sbox_index in range(8):
            sbox_center = actual_center(
                config.sbox_start_offset_samples
                + sbox_index * config.sbox_spacing_samples
            )
            sbox_centers[round_index, sbox_index] = sbox_center

            input_nibble = int(round_trace.f_trace.sbox_inputs[sbox_index])
            output_nibble = int(round_trace.f_trace.sbox_outputs[sbox_index])

            sbox_amplitude = event_amplitude(
                "sbox",
                normalized_hw(output_nibble, 4),
                normalized_hd(input_nibble, output_nibble, 4),
                config,
            )

            # تغییر کوچک و ثابت بین S-boxها برای مدل‌کردن تفاوت مسیرهای منطقی
            structural_scale = 1.0 + 0.035 * (sbox_index - 3.5) / 3.5

            add_template(
                clean_signal,
                sbox_center,
                templates["sbox"],
                effective_gain * structural_scale * sbox_amplitude,
            )

        # P-layer
        permutation_center = actual_center(
            config.permutation_offset_samples
        )
        permutation_centers[round_index] = permutation_center

        packed_sbox_output = lblock.pack_nibbles(
            round_trace.f_trace.sbox_outputs
        )
        permutation_amplitude = event_amplitude(
            "permutation",
            normalized_hw(f_output, 32),
            normalized_hd(packed_sbox_output, f_output, 32),
            config,
        )
        add_template(
            clean_signal,
            permutation_center,
            templates["permutation"],
            effective_gain * permutation_amplitude,
        )

        # Rotate
        rotate_center = actual_center(config.rotate_offset_samples)
        rotate_centers[round_index] = rotate_center
        rotate_amplitude = event_amplitude(
            "rotate",
            normalized_hw(rotated_x_prev2, 32),
            normalized_hd(x_prev2, rotated_x_prev2, 32),
            config,
        )
        add_template(
            clean_signal,
            rotate_center,
            templates["rotate"],
            effective_gain * rotate_amplitude,
        )

        # XOR نهایی تابع دور
        mix_center = actual_center(config.mix_offset_samples)
        mix_centers[round_index] = mix_center
        mix_amplitude = event_amplitude(
            "mix",
            normalized_hw(x_new, 32),
            normalized_hd(f_output, x_new, 32),
            config,
        )
        add_template(
            clean_signal,
            mix_center,
            templates["mix"],
            effective_gain * mix_amplitude,
        )

        # ثبت state جدید
        write_center = actual_center(config.write_offset_samples)
        write_centers[round_index] = write_center
        write_amplitude = event_amplitude(
            "write",
            normalized_hw(x_new, 32),
            normalized_hd(x_prev1, x_new, 32),
            config,
        )
        add_template(
            clean_signal,
            write_center,
            templates["write"],
            effective_gain * write_amplitude,
        )

    # --------------------------
    # نویز تصادفی
    # --------------------------

    noise_std = config.gaussian_noise_std * session.noise_scale
    gaussian_noise = rng.normal(0.0, noise_std, size=number_of_samples)

    impulsive_mask = (
        rng.random(number_of_samples)
        < config.impulsive_noise_probability
    )
    impulsive_noise = np.zeros(number_of_samples, dtype=np.float64)

    if np.any(impulsive_mask):
        impulsive_noise[impulsive_mask] = rng.normal(
            0.0,
            config.impulsive_noise_scale,
            size=int(np.sum(impulsive_mask)),
        )

    observed_trace = (
        clean_signal + gaussian_noise + impulsive_noise
    ).astype(np.float32)

    trace_digest = hashlib.sha256(observed_trace.tobytes()).hexdigest()

    public_metadata = {
        "trace_id": trace_id,
        "key_id": key_id,
        "session_id": session.session_id,
        "plaintext_class": plaintext_class,
        "plaintext_hex": lblock.hex_fixed(plaintext, 64),
        "ciphertext_hex": lblock.hex_fixed(ciphertext, 64),
        "global_trigger_sample": config.global_trigger_sample,
        "sampling_rate_hz": config.sampling_rate_hz,
        "number_of_samples": number_of_samples,
        "trace_sha256": trace_digest,
    }

    private_ground_truth: Dict[str, np.ndarray | int | float | str] = {
        "trace_id": trace_id,
        "key_id": key_id,
        "session_id": session.session_id,
        "master_key_hex": lblock.hex_fixed(master_key, 80),
        "plaintext_hex": lblock.hex_fixed(plaintext, 64),
        "ciphertext_hex": lblock.hex_fixed(ciphertext, 64),
        "encryption_start_sample": encryption_start,
        "trace_shift_samples": trace_shift,
        "trace_clock_scale": trace_clock_scale,
        "xor_centers": xor_centers,
        "sbox_centers": sbox_centers,
        "permutation_centers": permutation_centers,
        "rotate_centers": rotate_centers,
        "mix_centers": mix_centers,
        "write_centers": write_centers,
        "sbox_inputs": sbox_inputs,
        "sbox_outputs": sbox_outputs,
        "x_prev2": x_prev2_values,
        "x_prev1": x_prev1_values,
        "xor_words": xor_words,
        "f_outputs": f_outputs,
        "x_new": x_new_values,
        "round_keys": round_keys,
        "noise_std": noise_std,
        "effective_gain": effective_gain,
    }

    return observed_trace, public_metadata, private_ground_truth


# ============================================================
# 7. ذخیره فایل‌ها
# ============================================================


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_metadata_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("metadata rows cannot be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stack_private_ground_truth(
    private_rows: Sequence[Mapping[str, Any]],
    save_full_crypto_ground_truth: bool,
) -> Dict[str, np.ndarray]:
    """تبدیل ground truth هر trace به آرایه‌های فشرده NPZ."""

    scalar_integer_fields = [
        "trace_id",
        "key_id",
        "session_id",
        "encryption_start_sample",
        "trace_shift_samples",
    ]
    scalar_float_fields = [
        "trace_clock_scale",
        "noise_std",
        "effective_gain",
    ]
    timing_array_fields = [
        "xor_centers",
        "sbox_centers",
        "permutation_centers",
        "rotate_centers",
        "mix_centers",
        "write_centers",
    ]

    result: Dict[str, np.ndarray] = {}

    for field_name in scalar_integer_fields:
        result[field_name] = np.asarray(
            [row[field_name] for row in private_rows],
            dtype=np.int32,
        )

    for field_name in scalar_float_fields:
        result[field_name] = np.asarray(
            [row[field_name] for row in private_rows],
            dtype=np.float64,
        )

    for field_name in timing_array_fields:
        result[field_name] = np.stack(
            [np.asarray(row[field_name]) for row in private_rows],
            axis=0,
        )

    if save_full_crypto_ground_truth:
        for field_name in ["sbox_inputs", "sbox_outputs"]:
            result[field_name] = np.stack(
                [np.asarray(row[field_name]) for row in private_rows],
                axis=0,
            ).astype(np.uint8)

        for field_name in [
            "x_prev2",
            "x_prev1",
            "xor_words",
            "f_outputs",
            "x_new",
            "round_keys",
        ]:
            result[field_name] = np.stack(
                [np.asarray(row[field_name]) for row in private_rows],
                axis=0,
            ).astype(np.uint32)

    return result


# ============================================================
# 8. sanity checks و leakage audit
# ============================================================


def public_metadata_leakage_audit(
    metadata_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """کنترل اینکه زمان S-box، state و کلید واقعی در metadata عمومی نشت نکرده باشند."""

    if not metadata_rows:
        return {
            "passed": False,
            "reason": "metadata is empty",
            "columns": [],
        }

    columns = list(metadata_rows[0].keys())

    forbidden_exact = {
        "master_key",
        "master_key_hex",
        "round_key",
        "round_keys",
        "sbox_centers",
        "sbox_center",
        "round_centers",
        "encryption_start_sample",
        "trace_shift_samples",
        "internal_state",
        "x_prev1",
        "x_prev2",
        "x_new",
    }

    forbidden_substrings = [
        "actual_",
        "hidden_",
        "sbox_input",
        "sbox_output",
        "round_key",
    ]

    violations = []

    for column in columns:
        lowered = column.lower()

        if lowered in forbidden_exact:
            violations.append(column)
            continue

        if any(token in lowered for token in forbidden_substrings):
            violations.append(column)

    return {
        "passed": len(violations) == 0,
        "columns": columns,
        "violations": violations,
        "allowed_global_reference": "global_trigger_sample",
    }


def validate_event_bounds(
    private_arrays: Mapping[str, np.ndarray],
    number_of_samples: int,
    pulse_radius: int,
) -> Dict[str, Any]:
    """کنترل قرارگیری همه رویدادها داخل محدوده trace."""

    timing_fields = [
        "xor_centers",
        "sbox_centers",
        "permutation_centers",
        "rotate_centers",
        "mix_centers",
        "write_centers",
    ]

    minimum_center = min(
        int(np.min(private_arrays[field_name]))
        for field_name in timing_fields
    )
    maximum_center = max(
        int(np.max(private_arrays[field_name]))
        for field_name in timing_fields
    )

    passed = (
        minimum_center - pulse_radius >= 0
        and maximum_center + pulse_radius < number_of_samples
    )

    return {
        "passed": passed,
        "minimum_event_center": minimum_center,
        "maximum_event_center": maximum_center,
        "number_of_samples": number_of_samples,
        "pulse_radius": pulse_radius,
    }


def validate_event_order(
    private_arrays: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """کنترل ترتیب منطقی رویدادها در تک‌تک دورها و traceها."""

    xor_centers = private_arrays["xor_centers"]
    sbox_centers = private_arrays["sbox_centers"]
    permutation_centers = private_arrays["permutation_centers"]
    rotate_centers = private_arrays["rotate_centers"]
    mix_centers = private_arrays["mix_centers"]
    write_centers = private_arrays["write_centers"]

    violations = 0

    for trace_index in range(xor_centers.shape[0]):
        for round_index in range(xor_centers.shape[1]):
            sequence = [
                int(xor_centers[trace_index, round_index]),
                *[
                    int(value)
                    for value in sbox_centers[trace_index, round_index, :]
                ],
                int(permutation_centers[trace_index, round_index]),
                int(rotate_centers[trace_index, round_index]),
                int(mix_centers[trace_index, round_index]),
                int(write_centers[trace_index, round_index]),
            ]

            if any(
                current >= following
                for current, following in zip(sequence, sequence[1:])
            ):
                violations += 1

    return {
        "passed": violations == 0,
        "violating_trace_round_pairs": violations,
        "checked_trace_round_pairs": int(
            xor_centers.shape[0] * xor_centers.shape[1]
        ),
    }


def validate_ciphertexts(
    metadata_rows: Sequence[Mapping[str, Any]],
    key_set: Sequence[int],
) -> Dict[str, Any]:
    """بازمحاسبه ciphertextهای عمومی با استفاده از key manifest خصوصی."""

    failures: List[Dict[str, Any]] = []

    for row in metadata_rows:
        plaintext = int(str(row["plaintext_hex"]), 16)
        key_id = int(row["key_id"])
        expected_ciphertext = int(str(row["ciphertext_hex"]), 16)

        observed_ciphertext = lblock.encrypt_block_lblock(
            plaintext,
            key_set[key_id],
        )

        if observed_ciphertext != expected_ciphertext:
            failures.append({
                "trace_id": int(row["trace_id"]),
                "expected_ciphertext_hex": lblock.hex_fixed(
                    expected_ciphertext,
                    64,
                ),
                "observed_ciphertext_hex": lblock.hex_fixed(
                    observed_ciphertext,
                    64,
                ),
            })

            if len(failures) >= 10:
                break

    return {
        "passed": len(failures) == 0,
        "checked_traces": len(metadata_rows),
        "failures": failures,
    }


def evaluate_final_round_sbox_signal(
    traces: np.ndarray,
    private_arrays: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """
    oracle sanity-check برای وجود نشتی S-box دور آخر.

    این تابع فقط برای ارزیابی Stage 02 از زمان و state مخفی استفاده می‌کند.
    Stage 03 و Stage 04 مجاز نیستند این اطلاعات را برای localization بخوانند.
    """

    final_round_index = lblock.NUM_ROUNDS - 1
    centers = private_arrays["sbox_centers"][:, final_round_index, :]
    inputs = private_arrays["sbox_inputs"][:, final_round_index, :]
    outputs = private_arrays["sbox_outputs"][:, final_round_index, :]

    per_sbox: List[Dict[str, Any]] = []

    for sbox_index in range(8):
        sample_values = np.asarray([
            traces[trace_index, int(centers[trace_index, sbox_index])]
            for trace_index in range(traces.shape[0])
        ], dtype=np.float64)

        output_hw = np.asarray([
            hamming_weight(int(value))
            for value in outputs[:, sbox_index]
        ], dtype=np.float64)

        input_output_hd = np.asarray([
            hamming_weight(int(input_value) ^ int(output_value))
            for input_value, output_value
            in zip(inputs[:, sbox_index], outputs[:, sbox_index])
        ], dtype=np.float64)

        hw_correlation = safe_pearson_correlation(
            sample_values,
            output_hw,
        )
        hd_correlation = safe_pearson_correlation(
            sample_values,
            input_output_hd,
        )

        per_sbox.append({
            "target_sbox": sbox_index,
            "oracle_hw_correlation_at_true_center": hw_correlation,
            "oracle_hd_correlation_at_true_center": hd_correlation,
            "maximum_absolute_oracle_correlation": max(
                abs(hw_correlation),
                abs(hd_correlation),
            ),
            "median_true_center_sample": float(
                np.median(centers[:, sbox_index])
            ),
            "true_center_std_samples": float(
                np.std(centers[:, sbox_index])
            ),
        })

    minimum_signal = min(
        row["maximum_absolute_oracle_correlation"]
        for row in per_sbox
    )

    # آستانه پایین است چون این فقط sanity check است، نه نتیجه Timing Mapper.
    passed = minimum_signal >= 0.08

    return {
        "passed": passed,
        "minimum_absolute_correlation_across_sboxes": minimum_signal,
        "threshold": 0.08,
        "per_sbox": per_sbox,
        "validation_only": True,
    }


def trace_array_checks(traces: np.ndarray, config: Stage02Config) -> Dict[str, Any]:
    """کنترل شکل، نوع، finite بودن و تغییرپذیری مجموعه traceها."""

    expected_shape = (
        config.number_of_traces,
        trace_length_samples(config),
    )

    all_finite = bool(np.all(np.isfinite(traces)))
    non_constant = bool(float(np.std(traces)) > 1e-6)
    trace_to_trace_variation = float(
        np.mean(np.std(traces, axis=0))
    )

    return {
        "passed": (
            traces.shape == expected_shape
            and traces.dtype == np.float32
            and all_finite
            and non_constant
            and trace_to_trace_variation > 1e-4
        ),
        "expected_shape": list(expected_shape),
        "observed_shape": list(traces.shape),
        "dtype": str(traces.dtype),
        "all_finite": all_finite,
        "global_mean": float(np.mean(traces)),
        "global_std": float(np.std(traces)),
        "mean_pointwise_trace_std": trace_to_trace_variation,
    }


def deterministic_first_trace_check(
    assignment: TraceAssignment,
    session: SessionProfile,
    master_key: int,
    config: Stage02Config,
    templates: Mapping[str, np.ndarray],
    first_trace: np.ndarray,
) -> Dict[str, Any]:
    """تولید دوباره اولین trace و کنترل byte-for-byte آن."""

    regenerated_trace, regenerated_metadata, _ = simulate_one_trace(
        trace_id=assignment.trace_id,
        key_id=assignment.key_id,
        session=session,
        master_key=master_key,
        config=config,
        templates=templates,
    )

    byte_equal = bool(np.array_equal(first_trace, regenerated_trace))

    return {
        "passed": byte_equal,
        "trace_id": assignment.trace_id,
        "regenerated_sha256": regenerated_metadata["trace_sha256"],
    }


# ============================================================
# 9. شکل‌های مرحله دوم
# ============================================================


def save_public_trace_plots(
    public_directory: Path,
    traces: np.ndarray,
    config: Stage02Config,
) -> List[str]:
    """رسم trace نمونه و میانگین تریس؛ بدون استفاده از timing ground truth."""

    saved_files: List[str] = []
    sample_axis_us = (
        np.arange(traces.shape[1], dtype=np.float64)
        / config.sampling_rate_hz
        * 1e6
    )

    plt.figure(figsize=(15, 4.5))
    plt.plot(sample_axis_us, traces[0], linewidth=0.65)
    plt.axvline(
        config.global_trigger_sample / config.sampling_rate_hz * 1e6,
        linestyle="--",
        linewidth=1.0,
        label="Global trigger",
    )
    plt.xlabel("Time (µs)")
    plt.ylabel("Simulated power")
    plt.title("Stage 02 — Example full LBlock trace")
    plt.legend()
    plt.tight_layout()

    path = public_directory / "example_healthy_trace.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved_files.append(path.name)

    plt.figure(figsize=(15, 4.5))
    plt.plot(sample_axis_us, np.mean(traces, axis=0), linewidth=0.75)
    plt.axvline(
        config.global_trigger_sample / config.sampling_rate_hz * 1e6,
        linestyle="--",
        linewidth=1.0,
        label="Global trigger",
    )
    plt.xlabel("Time (µs)")
    plt.ylabel("Mean simulated power")
    plt.title("Stage 02 — Mean of all healthy traces")
    plt.legend()
    plt.tight_layout()

    path = public_directory / "mean_healthy_trace.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved_files.append(path.name)

    return saved_files


def save_private_validation_plot(
    private_directory: Path,
    traces: np.ndarray,
    private_arrays: Mapping[str, np.ndarray],
    config: Stage02Config,
) -> str:
    """
    رسم ROI دور آخر با زمان median واقعی S-boxها.

    فایل در private_ground_truth ذخیره می‌شود و فقط sanity-check است.
    Timing Mapper نباید این فایل یا زمان‌های آن را ورودی بگیرد.
    """

    final_centers = private_arrays["sbox_centers"][:, -1, :]
    median_centers = np.median(final_centers, axis=0)

    roi_start = int(max(0, np.min(median_centers) - 35))
    roi_end = int(min(traces.shape[1], np.max(median_centers) + 35))

    sample_axis = np.arange(roi_start, roi_end)
    mean_trace = np.mean(traces[:, roi_start:roi_end], axis=0)

    plt.figure(figsize=(14, 5))
    plt.plot(sample_axis, mean_trace, linewidth=0.85, label="Mean trace")

    for sbox_index, center in enumerate(median_centers):
        plt.axvline(
            float(center),
            linestyle="--",
            linewidth=0.9,
            label=f"Hidden S{sbox_index} median",
        )

    plt.xlabel("Sample index")
    plt.ylabel("Mean simulated power")
    plt.title(
        "Validation only — hidden final-round S-box timing ground truth"
    )
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()

    path = private_directory / "final_round_hidden_timing_validation.png"
    plt.savefig(path, dpi=160)
    plt.close()

    return path.name


# ============================================================
# 10. اجرای کامل Stage 02
# ============================================================


def run_stage_02(
    config: Optional[Stage02Config] = None,
) -> Dict[str, Any]:
    """اجرای کامل شبیه‌ساز تریس و ذخیره خروجی‌های public/private."""

    if config is None:
        config = Stage02Config()

    validate_config(config)

    start_time = time.perf_counter()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"stage02_{timestamp}_seed{config.random_seed}"

    run_directory = Path(config.output_root) / run_id
    public_directory = run_directory / "public"
    private_directory = run_directory / "private_ground_truth"

    public_directory.mkdir(parents=True, exist_ok=False)
    private_directory.mkdir(parents=True, exist_ok=False)

    number_of_samples = trace_length_samples(config)
    key_set = make_key_set(config)
    sessions = make_session_profiles(config)
    assignments = make_balanced_assignments(config)

    templates = {
        operation: operation_template(
            operation,
            config.pulse_radius_samples,
        )
        for operation in [
            "xor",
            "sbox",
            "permutation",
            "rotate",
            "mix",
            "write",
        ]
    }

    traces = np.empty(
        (config.number_of_traces, number_of_samples),
        dtype=np.float32,
    )
    public_metadata_rows: List[Dict[str, Any]] = []
    private_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 76)
    print("Stage 02 — Full LBlock power-trace simulation")
    print("=" * 76)
    print("Number of traces :", config.number_of_traces)
    print("Trace length     :", number_of_samples, "samples")
    print("Keys             :", config.number_of_keys)
    print("Sessions         :", config.number_of_sessions)
    print("Only public trigger: global_trigger_sample =", config.global_trigger_sample)
    print("=" * 76)

    progress_step = max(config.number_of_traces // 10, 1)

    for trace_index, assignment in enumerate(assignments):
        trace, public_metadata, private_ground_truth = simulate_one_trace(
            trace_id=assignment.trace_id,
            key_id=assignment.key_id,
            session=sessions[assignment.session_id],
            master_key=key_set[assignment.key_id],
            config=config,
            templates=templates,
        )

        traces[trace_index, :] = trace
        public_metadata_rows.append(public_metadata)
        private_rows.append(private_ground_truth)

        if (
            (trace_index + 1) % progress_step == 0
            or trace_index + 1 == config.number_of_traces
        ):
            print(
                f"Generated {trace_index + 1:>5} / "
                f"{config.number_of_traces} traces"
            )

    private_arrays = stack_private_ground_truth(
        private_rows,
        config.save_full_crypto_ground_truth,
    )

    # --------------------------
    # ذخیره public data
    # --------------------------

    trace_ids = np.asarray(
        [row["trace_id"] for row in public_metadata_rows],
        dtype=np.int32,
    )
    sample_axis_seconds = (
        np.arange(number_of_samples, dtype=np.float64)
        / config.sampling_rate_hz
    )

    np.savez_compressed(
        public_directory / "healthy_traces.npz",
        traces=traces,
        trace_ids=trace_ids,
        sample_axis_seconds=sample_axis_seconds,
        global_trigger_sample=np.asarray(
            config.global_trigger_sample,
            dtype=np.int32,
        ),
        sampling_rate_hz=np.asarray(
            config.sampling_rate_hz,
            dtype=np.float64,
        ),
    )

    write_metadata_csv(
        public_directory / "healthy_trace_metadata.csv",
        public_metadata_rows,
    )

    write_json(
        public_directory / "trace_simulation_config.json",
        asdict(config),
    )

    # --------------------------
    # ذخیره private ground truth
    # --------------------------

    np.savez_compressed(
        private_directory / "hidden_timing_and_crypto_ground_truth.npz",
        **private_arrays,
    )

    write_json(
        private_directory / "private_key_manifest.json",
        {
            "warning": (
                "Validation/profiling only. Do not use actual key values "
                "as attack features."
            ),
            "keys": [
                {
                    "key_id": key_id,
                    "master_key_hex": lblock.hex_fixed(master_key, 80),
                    "round_key_32_hex": lblock.hex_fixed(
                        lblock.key_schedule_lblock(master_key)[-1],
                        32,
                    ),
                }
                for key_id, master_key in enumerate(key_set)
            ],
        },
    )

    write_json(
        private_directory / "session_profiles.json",
        [asdict(session) for session in sessions],
    )

    nominal_final_round_start = (
        config.global_trigger_sample
        + (lblock.NUM_ROUNDS - 1) * config.round_duration_samples
    )

    write_json(
        private_directory / "hidden_nominal_timing_map.json",
        {
            "warning": (
                "Validation only. Stage 03/04 must estimate these times "
                "from public traces and must not load this file."
            ),
            "global_trigger_sample": config.global_trigger_sample,
            "nominal_final_round_start": nominal_final_round_start,
            "nominal_final_round_sbox_centers": {
                f"S{sbox_index}": (
                    nominal_final_round_start
                    + config.sbox_start_offset_samples
                    + sbox_index * config.sbox_spacing_samples
                )
                for sbox_index in range(8)
            },
        },
    )

    # --------------------------
    # sanity checks
    # --------------------------

    checks = {
        "stage_01_reference_recheck": lblock.validate_official_vectors(),
        "trace_array": trace_array_checks(traces, config),
        "public_metadata_leakage_audit": public_metadata_leakage_audit(
            public_metadata_rows
        ),
        "event_bounds": validate_event_bounds(
            private_arrays,
            number_of_samples,
            config.pulse_radius_samples,
        ),
        "event_order": validate_event_order(private_arrays),
        "ciphertexts": validate_ciphertexts(
            public_metadata_rows,
            key_set,
        ),
        "final_round_sbox_signal": evaluate_final_round_sbox_signal(
            traces,
            private_arrays,
        ),
        "deterministic_first_trace": deterministic_first_trace_check(
            assignments[0],
            sessions[assignments[0].session_id],
            key_set[assignments[0].key_id],
            config,
            templates,
            traces[0],
        ),
    }

    all_checks_passed = all(
        bool(check_result.get("passed", False))
        for check_result in checks.values()
    )

    write_json(
        run_directory / "stage_02_validation_checks.json",
        {
            "all_checks_passed": all_checks_passed,
            "checks": checks,
        },
    )

    # --------------------------
    # آمار و شکل‌ها
    # --------------------------

    unique_plaintexts = len({
        row["plaintext_hex"]
        for row in public_metadata_rows
    })
    unique_ciphertexts = len({
        row["ciphertext_hex"]
        for row in public_metadata_rows
    })

    final_round_centers = private_arrays["sbox_centers"][:, -1, :]

    trace_statistics = {
        "trace_shape": list(traces.shape),
        "trace_dtype": str(traces.dtype),
        "sampling_rate_hz": config.sampling_rate_hz,
        "sample_period_ns": 1e9 / config.sampling_rate_hz,
        "trace_duration_us": number_of_samples / config.sampling_rate_hz * 1e6,
        "global_trigger_sample": config.global_trigger_sample,
        "global_mean": float(np.mean(traces)),
        "global_std": float(np.std(traces)),
        "minimum_value": float(np.min(traces)),
        "maximum_value": float(np.max(traces)),
        "unique_plaintexts": unique_plaintexts,
        "unique_ciphertexts": unique_ciphertexts,
        "final_round_hidden_center_summary_validation_only": [
            {
                "target_sbox": sbox_index,
                "median_sample": float(
                    np.median(final_round_centers[:, sbox_index])
                ),
                "std_samples": float(
                    np.std(final_round_centers[:, sbox_index])
                ),
                "minimum_sample": int(
                    np.min(final_round_centers[:, sbox_index])
                ),
                "maximum_sample": int(
                    np.max(final_round_centers[:, sbox_index])
                ),
            }
            for sbox_index in range(8)
        ],
    }

    write_json(
        public_directory / "trace_statistics.json",
        {
            key: value
            for key, value in trace_statistics.items()
            if key != "final_round_hidden_center_summary_validation_only"
        },
    )

    write_json(
        private_directory / "hidden_timing_statistics.json",
        {
            "warning": "Validation only",
            "final_round_hidden_center_summary": (
                trace_statistics[
                    "final_round_hidden_center_summary_validation_only"
                ]
            ),
        },
    )

    plot_files: List[str] = []

    if config.save_plots:
        plot_files.extend(
            save_public_trace_plots(
                public_directory,
                traces,
                config,
            )
        )
        plot_files.append(
            save_private_validation_plot(
                private_directory,
                traces,
                private_arrays,
                config,
            )
        )

    elapsed_seconds = time.perf_counter() - start_time

    config_digest = hashlib.sha256(
        json.dumps(
            asdict(config),
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    run_manifest = {
        "stage": 2,
        "run_id": run_id,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "matplotlib_version": importlib.import_module("matplotlib").__version__,
        "platform": platform.platform(),
        "config_sha256": config_digest,
        "random_seed": config.random_seed,
        "stage_01_module_path": str(Path(lblock.__file__).resolve()),
    }

    write_json(run_directory / "run_manifest.json", run_manifest)

    summary = {
        "stage": 2,
        "run_id": run_id,
        "run_directory": str(run_directory.resolve()),
        "public_directory": str(public_directory.resolve()),
        "private_ground_truth_directory": str(private_directory.resolve()),
        "all_checks_passed": all_checks_passed,
        "number_of_traces": config.number_of_traces,
        "number_of_samples_per_trace": number_of_samples,
        "number_of_keys": config.number_of_keys,
        "number_of_sessions": config.number_of_sessions,
        "sampling_rate_hz": config.sampling_rate_hz,
        "global_trigger_sample": config.global_trigger_sample,
        "public_metadata_leakage_audit_passed": checks[
            "public_metadata_leakage_audit"
        ]["passed"],
        "event_order_passed": checks["event_order"]["passed"],
        "event_bounds_passed": checks["event_bounds"]["passed"],
        "ciphertext_validation_passed": checks["ciphertexts"]["passed"],
        "deterministic_generation_passed": checks[
            "deterministic_first_trace"
        ]["passed"],
        "final_round_sbox_signal_check_passed": checks[
            "final_round_sbox_signal"
        ]["passed"],
        "minimum_final_round_oracle_correlation": checks[
            "final_round_sbox_signal"
        ]["minimum_absolute_correlation_across_sboxes"],
        "elapsed_seconds": elapsed_seconds,
        "public_files": sorted(
            path.name
            for path in public_directory.iterdir()
            if path.is_file()
        ),
        "private_files": sorted(
            path.name
            for path in private_directory.iterdir()
            if path.is_file()
        ),
    }

    write_json(run_directory / "stage_02_summary.json", summary)

    print("\n" + "=" * 76)
    print("Stage 02 complete: full healthy power traces generated")
    print("=" * 76)
    print("Run directory                  :", summary["run_directory"])
    print("All checks passed              :", summary["all_checks_passed"])
    print("Trace shape                    :", traces.shape)
    print("Only public timing reference   : global_trigger_sample =",
          config.global_trigger_sample)
    print("Public metadata leakage audit  :",
          summary["public_metadata_leakage_audit_passed"])
    print("Event ordering                 :", summary["event_order_passed"])
    print("Ciphertext validation          :",
          summary["ciphertext_validation_passed"])
    print("Deterministic generation       :",
          summary["deterministic_generation_passed"])
    print("Final-round S-box signal check :",
          summary["final_round_sbox_signal_check_passed"])
    print("Minimum oracle correlation     :",
          f"{summary['minimum_final_round_oracle_correlation']:.4f}")
    print("Elapsed seconds                :", f"{elapsed_seconds:.2f}")
    print("=" * 76)

    print("\nPublic files for Stage 03/04:")
    for file_name in summary["public_files"]:
        print("  -", file_name)

    print("\nPrivate validation files — do not use for timing estimation:")
    for file_name in summary["private_files"]:
        print("  -", file_name)

    if not all_checks_passed:
        raise AssertionError(
            "Stage 02 validation failed. Inspect stage_02_validation_checks.json."
        )

    return summary


# ============================================================
# 11. بارگذاری config از JSON
# ============================================================


def load_stage_02_config(config_path: str | Path) -> Stage02Config:
    """بارگذاری Stage02Config از فایل JSON."""

    path = Path(config_path)

    with path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    return Stage02Config(**raw_config)


if __name__ == "__main__":
    run_stage_02()
