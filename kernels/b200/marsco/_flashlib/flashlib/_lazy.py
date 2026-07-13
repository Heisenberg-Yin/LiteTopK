"""Shared lazy-import helper.

Each primitive's public API surfaces a small handful of CuteDSL backend
symbols. Importing CuteDSL eagerly pulls in the full ``cutlass`` /
``cutlass.cute`` stack which is heavy and hardware-gated (Hopper-only).
We therefore expose those names as thin wrappers that resolve their
backing module on first call.

This used to be duplicated as ``_lazy_cutedsl_attr`` inside every
``primitives/*/__init__.py``. Now it lives here so each primitive's
``__init__`` is one line per name::

    from flashlib._lazy import lazy_attr

    cutedsl_foo = lazy_attr("flashlib.primitives.foo.cutedsl", "cutedsl_foo")

The returned callable forwards ``*args, **kwargs`` to the resolved
attribute, so the wrapper is indistinguishable from the real function
at the call site.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable


def lazy_attr(module_path: str, name: str) -> Callable[..., Any]:
    """Return a callable that resolves ``getattr(import(module_path), name)``
    on first invocation and forwards ``*args, **kwargs`` to it.

    Parameters
    ----------
    module_path:
        Fully-qualified import path of the module that hosts ``name``
        (e.g. ``"flashlib.primitives.knn.cutedsl"``).
    name:
        Attribute (typically a function or class) to fetch from the
        module.
    """
    def _f(*args, **kwargs):
        mod = importlib.import_module(module_path)
        return getattr(mod, name)(*args, **kwargs)

    _f.__name__ = name
    _f.__qualname__ = name
    _f.__doc__ = f"Lazy proxy for ``{module_path}.{name}``."
    return _f


__all__ = ["lazy_attr"]
