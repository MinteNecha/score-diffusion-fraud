"""Simple YAML-backed config loader.

Keeps things lightweight (no dependency on hydra/omegaconf) since the
project only needs a handful of flat config values per experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """A dict that also allows attribute-style access, e.g. cfg.batch_size."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config({k: Config._wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, keeping keys `override`
    doesn't touch (e.g. overriding dataset.name shouldn't wipe out
    dataset.raw_dir / test_size / val_size).
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Config:
    """Load a YAML config file into a Config object.

    `overrides` is a (possibly nested) dict of keys to override, e.g.
    {"dataset": {"name": "paysim"}} - nested dicts are merged recursively
    rather than replacing the whole sub-config, so unrelated keys survive.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    return Config._wrap(raw)
