"""Shared lookups over a stage's method registry (see the method registry page).

Each stage owns one plain module-level dict from an exact frozen config type to
its own frozen spec dataclass. These functions only read such a mapping: they
never cache it or copy it into a second list, so a method inserted into the
dict (for example with monkeypatch.setitem) is seen by every lookup. what is
the noun of the stage's messages ("preprocessing step", "registration method").
"""
import importlib
import re
from dataclasses import dataclass
from importlib import metadata

_NAME = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")


@dataclass(frozen=True)
class Dependency:
    """An optional dependency of a method: imported lazily, recorded with its version."""
    module: str               # imported lazily, e.g. "SimpleITK"
    distribution: str         # recorded with its version, e.g. "SimpleITK"
    extra: str | None = None  # install hint, e.g. "local-registration"

    def __post_init__(self):
        for name in ("module", "distribution"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise TypeError(f"dependency {name} must be a nonempty string")
        if self.extra is not None and (not isinstance(self.extra, str) or not self.extra):
            raise TypeError("dependency extra must be a nonempty string or None")


def check_name(name: str, what: str) -> None:
    """Raise ValueError unless name is lowercase snake_case ([a-z][a-z0-9]*(_[a-z0-9]+)*)."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"{what} name must be lowercase snake_case")


def check_shared(spec, what: str) -> None:
    """Validate the shared spec fields: name, callable run, requires and min_shape_zyx."""
    check_name(spec.name, what)
    if not callable(spec.run):
        raise TypeError(f"{what} run must be callable")
    if not isinstance(spec.requires, tuple) or not all(isinstance(d, Dependency) for d in spec.requires):
        raise TypeError(f"{what} {spec.name!r} requires must be a tuple of Dependency")
    shape = spec.min_shape_zyx
    if (not isinstance(shape, tuple) or len(shape) != 3
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape)):
        raise ValueError(f"{what} {spec.name!r} min_shape_zyx must be three positive integers")


def spec_for(registry, config, what: str, error: type[Exception] = TypeError, message: str | None = None):
    """Spec registered for type(config) exactly; subclasses are not matched.

    Raises error with message, or with the default "no <what> is registered
    for <type> (lookup uses the exact config type)".
    """
    spec = registry.get(type(config))
    if spec is None:
        raise error(message or f"no {what} is registered for {type(config).__qualname__} "
                               "(lookup uses the exact config type)")
    return spec


def config_type_for(registry, name: str, what: str, error: type[Exception] = ValueError,
                    message: str | None = None) -> type:
    """Config type whose spec is called name.

    Raises error with message (default "unknown <what> '<name>'") when no
    spec has this name, and error when more than one has it.
    """
    matches = [config_type for config_type, spec in registry.items() if spec.name == name]
    if not matches:
        raise error(message or f"unknown {what} {name!r}")
    if len(matches) > 1:
        raise error(f"{what} name {name!r} is registered more than once")
    return matches[0]


def names(registry) -> tuple[str, ...]:
    """Registered names in registry order."""
    return tuple(spec.name for spec in registry.values())


def require(spec, what: str, error: type[Exception]) -> None:
    """Import each declared dependency; raise error naming the module and extra if one is missing."""
    for dependency in spec.requires:
        try:
            importlib.import_module(dependency.module)
        except ImportError as exc:
            extra = dependency.extra
            hint = f"; install the '{extra}' extra (starfinder[{extra}])" if extra else ""
            raise error(f"{what} {spec.name!r} requires {dependency.module}{hint}") from exc


def _public_path(obj) -> str:
    """Qualified name through the public module: private module components are dropped."""
    parts = obj.__module__.split(".")
    for i, part in enumerate(parts):
        if part.startswith("_"):
            parts = parts[:i]
            break
    return ".".join(parts + [obj.__qualname__])


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def provenance(spec, config, stage: str) -> dict:
    """Uniform provenance entry of one method invocation.

    config is serialized as run.json serializes dataclasses; requires maps
    each declared distribution to its installed version (None if absent);
    artifacts is reserved and empty.
    """
    from starfinder.io._checkpoint import _jsonable
    return {"stage": stage, "method": spec.name, "config_type": _public_path(type(config)),
            "implementation": f"{spec.run.__module__}.{spec.run.__qualname__}", "config": _jsonable(config),
            "requires": {d.distribution: _version(d.distribution) for d in spec.requires}, "artifacts": []}
