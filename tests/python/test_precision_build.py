"""Validate native precision selection and reject contradictory build settings."""

from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SUPPORTED_SETTINGS = (
    {},
    {"LACUNA_REAL_BITS": "64", "LACUNA_TIME_BITS": "64"},
    {"LACUNA_REAL_BITS": "32", "LACUNA_TIME_BITS": "64"},
    {"LACUNA_REAL_BITS": "32", "LACUNA_TIME_BITS": "32"},
    {"LACUNA_REAL_BITS": "16", "LACUNA_TIME_BITS": "16"},
)
_UNSUPPORTED_SETTINGS = (
    {"LACUNA_TIME_BITS": "32"},
    {"LACUNA_REAL_BITS": "16"},
    {"LACUNA_REAL_BITS": "32", "LACUNA_TIME_BITS": "16"},
    {"LACUNA_REAL_BITS": "invalid"},
    {"LACUNA_TIME_BITS": "invalid"},
    {"LACUNA_REAL_BITS": ""},
    {"LACUNA_TIME_BITS": ""},
)


def _run(command):
    return subprocess.run(
        command,
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.fixture(scope="module")
def cmake_command():
    executable = shutil.which("cmake")
    if executable is None:
        pytest.skip("CMake is unavailable for precision configuration checks")
    return executable


@pytest.fixture(scope="module")
def c_compiler():
    command = shlex.split(os.environ.get("CC", "cc"))
    if not command or shutil.which(command[0]) is None:
        pytest.skip("A C compiler is unavailable for public-header checks")
    return command


def _configure(cmake_command, build, settings):
    return _run(
        [
            cmake_command,
            "-S", str(_ROOT),
            "-B", str(build),
            "-DBUILD_TESTING=OFF",
            *[f"-D{name}={value}" for name, value in settings.items()],
        ]
    )


@pytest.mark.parametrize(
    "settings",
    _SUPPORTED_SETTINGS,
    ids=("default", "explicit-binary64", "binary32-time64", "strict-binary32", "strict-binary16"),
)
def test_cmake_accepts_supported_profiles(cmake_command, tmp_path, settings):
    result = _configure(cmake_command, tmp_path / "build", settings)
    assert result.returncode == 0, result.stdout
    cache = (tmp_path / "build" / "CMakeCache.txt").read_text()
    for name in ("LACUNA_REAL_BITS", "LACUNA_TIME_BITS"):
        values = [
            line.split("=", 1)[1]
            for line in cache.splitlines()
            if line.startswith(f"{name}:")
        ]
        assert values == [settings.get(name, "64")], cache


@pytest.mark.parametrize("settings", _UNSUPPORTED_SETTINGS)
def test_cmake_rejects_unsupported_precision(cmake_command, tmp_path, settings):
    result = _configure(cmake_command, tmp_path / "build", settings)
    assert result.returncode != 0, result.stdout
    assert any(name in result.stdout for name in settings), result.stdout
    assert "64" in result.stdout, result.stdout


@pytest.fixture
def header_probe(tmp_path):
    source = tmp_path / "precision_probe.c"
    source.write_text(
        '#include "lacuna.h"\n'
        '#include <limits.h>\n'
        'typedef char real_width[(sizeof(lc_real_t) * CHAR_BIT == '
        'LACUNA_REAL_BITS) ? 1 : -1];\n'
        'typedef char time_width[(sizeof(lc_time_t) * CHAR_BIT == '
        'LACUNA_TIME_BITS) ? 1 : -1];\n'
        'int main(void) { return 0; }\n'
    )
    return source


def _compile_header(c_compiler, header_probe, settings, extra_flags=()):
    return _run(
        [
            *c_compiler,
            "-std=c99",
            "-fsyntax-only",
            "-I", str(_ROOT / "c/include"),
            *[f"-D{name}={value}" for name, value in settings.items()],
            *extra_flags,
            str(header_probe),
        ]
    )


@pytest.mark.parametrize(
    "settings",
    _SUPPORTED_SETTINGS,
    ids=("default", "explicit-binary64", "binary32-time64", "strict-binary32", "strict-binary16"),
)
def test_direct_header_accepts_supported_profiles(c_compiler, header_probe, settings):
    result = _compile_header(c_compiler, header_probe, settings)
    if settings.get("LACUNA_REAL_BITS") == "16" and (
        "Strict binary16 requires native half arithmetic" in result.stdout
    ):
        pytest.skip("The compiler target does not provide native binary16 arithmetic")
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("settings", _UNSUPPORTED_SETTINGS)
def test_direct_header_rejects_caller_overrides(c_compiler, header_probe, settings):
    result = _compile_header(c_compiler, header_probe, settings)
    assert result.returncode != 0, result.stdout
    assert "lacuna_numeric.h" in result.stdout, result.stdout


@pytest.mark.parametrize("settings", (
    {"LACUNA_REAL_BITS": "32"},
    {"LACUNA_TIME_BITS": "32"},
    {"LACUNA_REAL_BITS": "32", "LACUNA_TIME_BITS": "32"},
))
def test_cmake_caller_flags_cannot_hide_float32_request(
    cmake_command, header_probe, tmp_path, settings
):
    build = tmp_path / "build"
    result = _configure(
        cmake_command,
        build,
        {
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "CMAKE_C_FLAGS": " ".join(
                f"-D{name}={value}" for name, value in settings.items()
            ),
        },
    )
    assert result.returncode == 0, result.stdout
    commands_file = build / "compile_commands.json"
    if not commands_file.exists():
        pytest.skip("The selected CMake generator does not export compiler commands")
    command = json.loads(commands_file.read_text())[0]
    arguments = command.get("arguments") or shlex.split(command["command"])
    probe_arguments = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-o":
            index += 2
            continue
        if argument == "-c" or argument == command["file"]:
            index += 1
            continue
        probe_arguments.append(argument)
        index += 1
    result = _run([*probe_arguments, "-fsyntax-only", str(header_probe)])
    assert result.returncode != 0, result.stdout
    assert any(name in result.stdout for name in settings), result.stdout


def test_strict_binary32_profiling_is_rejected(cmake_command, c_compiler,
                                              header_probe, tmp_path):
    settings = {
        "LACUNA_REAL_BITS": "32", "LACUNA_TIME_BITS": "32",
        "LACUNA_ENABLE_PROFILING": "1",
    }
    configured = _configure(cmake_command, tmp_path / "build", settings)
    assert configured.returncode != 0
    assert "LACUNA_ENABLE_PROFILING" in configured.stdout
    compiled = _compile_header(c_compiler, header_probe, settings)
    assert compiled.returncode != 0
    assert "LACUNA_ENABLE_PROFILING" in compiled.stdout


def test_direct_binary32_rejects_fast_math(c_compiler, header_probe):
    result = _compile_header(
        c_compiler, header_probe,
        {"LACUNA_REAL_BITS": "32", "LACUNA_TIME_BITS": "32"},
        ("-ffast-math",),
    )
    assert result.returncode != 0
    assert "strict floating-point" in result.stdout


@pytest.mark.parametrize("arguments,profiles", (
    ((), ("float64", "float32")),
    (("--profiles", "float16"), ("float16",)),
    (("--profiles", "float32-time64"), ("float32-time64",)),
    (("--profiles", "float64", "float32", "float16"), ("float64", "float32", "float16")),
))
def test_build_script_selects_independent_profiles(monkeypatch, arguments, profiles):
    from scripts import build_precision_profiles

    calls = []
    monkeypatch.setattr(build_precision_profiles.subprocess, "run",
                        lambda command, **kwargs: calls.append((command, kwargs)))
    assert build_precision_profiles.main([*arguments, "--jobs", "3"]) == 0
    assert len(calls) == 3 * len(profiles)
    for index, profile in enumerate(profiles):
        directory, real_bits, time_bits = build_precision_profiles.PROFILES[profile]
        build = str(_ROOT / directory)
        configure, compile_command, test_command = calls[index * 3:index * 3 + 3]
        assert configure[0][:5] == ["cmake", "-S", str(_ROOT), "-B", build]
        assert f"-DLACUNA_REAL_BITS={real_bits}" in configure[0]
        assert f"-DLACUNA_TIME_BITS={time_bits}" in configure[0]
        assert f"-DLACUNA_ENABLE_PROFILING={'OFF' if time_bits <= 32 else 'ON'}" in configure[0]
        assert f"-DLACUNA_BUILD_TARGET_BINDER={'ON' if real_bits == 64 else 'OFF'}" in configure[0]
        assert "-DCMAKE_BUILD_TYPE=Release" in configure[0]
        assert "-DBUILD_TESTING=ON" in configure[0]
        assert compile_command[0] == ["cmake", "--build", build, "--parallel", "3"]
        assert test_command[0] == ["ctest", "--test-dir", build, "--output-on-failure"]
        assert all(options == {"check": True} for _, options in
                   (configure, compile_command, test_command))


def test_build_script_rejects_invalid_jobs_before_configuring(monkeypatch):
    from scripts import build_precision_profiles

    monkeypatch.setattr(build_precision_profiles.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail("invalid jobs started a build"))
    with pytest.raises(SystemExit, match="2"):
        build_precision_profiles.main(["--jobs", "0"])
