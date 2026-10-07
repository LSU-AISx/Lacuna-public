"""Build and test independent native libraries without switching global types."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


PROFILES = {
    "float64": ("build", 64, 64),
    "float32-time64": ("build-float32-time64", 32, 64),
    "float32": ("build-float32", 32, 32),
    "float16": ("build-float16", 16, 16),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        default=("float64", "float32"))
    parser.add_argument("--build-type", choices=("Release", "Debug"), default="Release")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    for profile in args.profiles:
        directory, real_bits, time_bits = PROFILES[profile]
        build = root / directory
        print(f"Building and testing {profile}", flush=True)
        subprocess.run([
            "cmake", "-S", str(root), "-B", str(build),
            f"-DCMAKE_BUILD_TYPE={args.build_type}",
            f"-DLACUNA_REAL_BITS={real_bits}", f"-DLACUNA_TIME_BITS={time_bits}",
            f"-DLACUNA_ENABLE_PROFILING={'OFF' if time_bits <= 32 else 'ON'}",
            f"-DLACUNA_BUILD_TARGET_BINDER={'ON' if real_bits == 64 else 'OFF'}",
            "-DBUILD_TESTING=ON",
        ], check=True)
        subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], check=True)
        subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
