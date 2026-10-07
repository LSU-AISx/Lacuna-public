"""Bounded, process-local memoization of pure authoring operations.

This cache contains Python model descriptions only. Compiled graphs, C handles,
simulation state, and binary image loading do not participate in it.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import wraps
import os
import struct
import sys
from threading import RLock
from types import MappingProxyType
from typing import Callable, Iterator, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_DISABLED: ContextVar[bool] = ContextVar("lacuna_resolution_cache_disabled", default=False)
_MISSING = object()
_VALUE_MODULES = frozenset(("lacuna.ir", "lacuna.expr", "lacuna.resolver"))


class _Uncacheable(TypeError):
    pass


def _key(value: object, active: set[int]) -> object:
    """Preserve all declared fields, mapping order, types, and float bits.

    Only known value records and ordinary containers participate. Custom
    bindings/objects retain their original behavior through the uncached path.
    """

    kind = type(value)
    if isinstance(value, Enum) and kind.__module__ in _VALUE_MODULES:
        return (kind, value.name)
    if kind is float:
        return (float, struct.pack("!d", value))
    if kind in (int, str, bytes, bool, type(None)):
        return (kind, value)
    record = is_dataclass(value) and kind.__module__ in _VALUE_MODULES
    if not record and kind not in (dict, OrderedDict, MappingProxyType, tuple, list):
        raise _Uncacheable(kind.__name__)
    identity = id(value)
    if identity in active:
        raise _Uncacheable("cyclic argument")
    active.add(identity)
    try:
        if record:
            return (kind, tuple((field.name, _key(getattr(value, field.name), active))
                                for field in fields(value)))
        if kind in (dict, OrderedDict, MappingProxyType):
            return (kind, tuple((_key(name, active), _key(item, active))
                                for name, item in value.items()))
        return (kind, tuple(_key(item, active) for item in value))
    finally:
        active.remove(identity)


def _retained_size(value: object, seen: set[int]) -> int:
    """Estimate retained Python storage, counting shared references once.

    Shared objects across separate cache entries may be counted more than once;
    this is a conservative accounting budget, not a process RSS measurement.
    """

    identity = id(value)
    if identity in seen or isinstance(value, (type, Enum)):
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if is_dataclass(value):
        attributes = getattr(value, "__dict__", None)
        if attributes is not None:
            size += _retained_size(attributes, seen)
        else:
            size += sum(_retained_size(getattr(value, field.name), seen)
                        for field in fields(value))
    elif isinstance(value, (dict, MappingProxyType)):
        size += sum(_retained_size(k, seen) + _retained_size(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(_retained_size(item, seen) for item in value)
    return size


@dataclass(frozen=True)
class ResolutionCacheInfo:
    """Process-wide cache settings and counters (excluding scoped bypasses)."""

    enabled: bool
    max_entries: int
    max_bytes: int
    entries: int
    estimated_bytes: int
    hits: int
    misses: int


class _ResolutionCache:
    def __init__(self) -> None:
        self.lock = RLock()
        self.enabled = True
        self.max_entries = 4096
        self.max_bytes = 64 * 1024 * 1024
        self.generation = 0
        self._reset()

    def _reset(self) -> None:
        self.entries: OrderedDict[object, tuple[object, int]] = OrderedDict()
        self.estimated_bytes = self.hits = self.misses = 0
        self.generation += 1

    def after_fork(self) -> None:
        # A different parent thread might have held the old lock at fork.
        self.lock = RLock()
        self._reset()

    def info(self) -> ResolutionCacheInfo:
        with self.lock:
            return ResolutionCacheInfo(self.enabled, self.max_entries, self.max_bytes,
                                       len(self.entries), self.estimated_bytes,
                                       self.hits, self.misses)

    def call(self, function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if _DISABLED.get():
            return function(*args, **kwargs)
        with self.lock:
            enabled = self.enabled and self.max_entries > 0 and self.max_bytes > 0
        if not enabled:
            return function(*args, **kwargs)
        try:
            key = (function, _key(args, set()), _key(kwargs, set()))
        except (_Uncacheable, RecursionError):
            return function(*args, **kwargs)
        with self.lock:
            enabled = self.enabled and self.max_entries > 0 and self.max_bytes > 0
            generation = self.generation
            entry = self.entries.get(key, _MISSING) if enabled else _MISSING
            if entry is not _MISSING:
                self.entries.move_to_end(key)
                self.hits += 1
            elif enabled:
                self.misses += 1
        if not enabled:
            return function(*args, **kwargs)
        if entry is not _MISSING:
            # No caller receives the private stored value. Copy outside the lock.
            return deepcopy(entry[0])

        # Do not hold the lock during symbolic work. Concurrent misses may compute
        # the same pure result independently. Errors are propagated, never cached.
        result = function(*args, **kwargs)
        stored = deepcopy(result)
        size = _retained_size((key, stored), set())
        with self.lock:
            # Clear/configure must not be undone by an earlier in-flight miss.
            if (generation == self.generation and self.enabled
                    and self.max_entries > 0 and size <= self.max_bytes
                    and key not in self.entries):
                while self.entries and (len(self.entries) >= self.max_entries
                        or self.estimated_bytes + size > self.max_bytes):
                    _, (_, removed_size) = self.entries.popitem(last=False)
                    self.estimated_bytes -= removed_size
                self.entries[key] = (stored, size)
                self.estimated_bytes += size
        return result


_CACHE = _ResolutionCache()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_CACHE.after_fork)


def _cached_resolution(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def cached(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return _CACHE.call(function, *args, **kwargs)
    return cached


def resolution_cache_info() -> ResolutionCacheInfo:
    """Inspect process-wide cache settings, accounting, and hit/miss counts."""

    return _CACHE.info()


def clear_resolution_cache() -> None:
    """Release retained descriptions and reset counters in this process."""

    with _CACHE.lock:
        _CACHE._reset()


def configure_resolution_cache(
    *, enabled: bool | None = None, max_entries: int | None = None,
    max_bytes: int | None = None,
) -> ResolutionCacheInfo:
    """Update supplied settings and clear the cache; zero limits bypass it.

    Defaults are enabled, 4,096 entries and 64 MiB of estimated retained Python
    storage. These settings do not affect compiled graphs or binary loaders.
    """

    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    with _CACHE.lock:
        if enabled is not None:
            _CACHE.enabled = enabled
        if max_entries is not None:
            _CACHE.max_entries = max_entries
        if max_bytes is not None:
            _CACHE.max_bytes = max_bytes
        _CACHE._reset()
        return _CACHE.info()


@contextmanager
def resolution_cache_disabled() -> Iterator[None]:
    """Bypass reads/writes in this thread/task context, including nested calls."""

    token = _DISABLED.set(True)
    try:
        yield
    finally:
        _DISABLED.reset(token)
