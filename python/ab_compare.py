#!/usr/bin/env python3
"""Run A/B singing VC comparison against local FastAPI server."""

import argparse
import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


def parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def bool_form(value: bool) -> str:
    return "true" if value else "false"


@dataclass
class CaseParams:
    name: str
    diffusion_steps: int
    length_adjust: float
    inference_cfg_rate: float
    auto_f0_adjust: bool
    pitch_shift: int
    use_fp16: bool


def run_case(port: int, source: Path, target: Path, out_file: Path, params: CaseParams) -> None:
    command = [
        "curl",
        "-sN",
        "-X",
        "POST",
        f"http://127.0.0.1:{port}/convert",
        "-F",
        f"source_file=@{source}",
        "-F",
        f"target_file=@{target}",
        "-F",
        f"diffusion_steps={params.diffusion_steps}",
        "-F",
        f"length_adjust={params.length_adjust}",
        "-F",
        f"inference_cfg_rate={params.inference_cfg_rate}",
        "-F",
        f"auto_f0_adjust={bool_form(params.auto_f0_adjust)}",
        "-F",
        f"pitch_shift={params.pitch_shift}",
        "-F",
        f"use_fp16={bool_form(params.use_fp16)}",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    done_audio_b64 = None
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        event_type = payload.get("type")
        if event_type == "error":
            message = payload.get("message", "Unknown backend error")
            process.kill()
            raise RuntimeError(f"{params.name} failed: {message}")
        if event_type == "done":
            done_audio_b64 = payload.get("audio")

    stderr = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{params.name} curl failed ({return_code}): {stderr.strip()}")
    if not done_audio_b64:
        raise RuntimeError(f"{params.name} did not return final audio")

    out_file.write_bytes(base64.b64decode(done_audio_b64))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B compare old defaults vs singing-optimized parameters."
    )
    parser.add_argument("--source", required=True, help="Path to source audio")
    parser.add_argument("--target", required=True, help="Path to reference audio")
    parser.add_argument("--port", type=int, default=18888, help="FastAPI server port")
    parser.add_argument("--out-dir", default="ab_outputs", help="Output directory")
    parser.add_argument("--quality-steps", type=int, default=40, help="B case diffusion steps")
    parser.add_argument(
        "--quality-fp16",
        type=parse_bool,
        default=False,
        help="B case fp16 switch, e.g. true/false (default: false)",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Source audio not found: {source}")
    if not target.exists():
        raise FileNotFoundError(f"Target audio not found: {target}")

    cases = [
        CaseParams(
            name="A_old_default",
            diffusion_steps=10,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
            auto_f0_adjust=True,
            pitch_shift=0,
            use_fp16=True,
        ),
        CaseParams(
            name="B_singing_optimized",
            diffusion_steps=args.quality_steps,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
            auto_f0_adjust=False,
            pitch_shift=0,
            use_fp16=args.quality_fp16,
        ),
    ]

    print(f"Using server: http://127.0.0.1:{args.port}")
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Output dir: {out_dir}")

    for case in cases:
        output_file = out_dir / f"{case.name}.wav"
        print(
            f"\n[{case.name}] steps={case.diffusion_steps}, "
            f"auto_f0_adjust={case.auto_f0_adjust}, use_fp16={case.use_fp16}"
        )
        run_case(args.port, source, target, output_file, case)
        print(f"Saved: {output_file}")

    print("\nA/B comparison complete. Listen to the two WAV files in output dir.")


if __name__ == "__main__":
    main()
