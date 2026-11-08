"""Run configuration: TOML file -> validated pydantic models (DESIGN §10-11).

Only sections implemented so far exist; unknown keys are errors, so typos
are caught instead of silently ignored. Paths are resolved relative to the
config file.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    """Bad configuration or input (CLI exit code 2)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanetConfig(_Strict):
    radius_km: float = Field(6371.0, gt=0)


class ResolutionConfig(_Strict):
    output_width: int = Field(8192, ge=8)
    levels: list[int] = Field(default_factory=lambda: [512, 2048])

    @field_validator("output_width")
    @classmethod
    def _width_multiple_of_8(cls, v: int) -> int:
        if v % 8:
            raise ValueError("output_width must be a multiple of 8 (N = width / 4 must be even)")
        return v

    @field_validator("levels")
    @classmethod
    def _levels_increasing(cls, v: list[int]) -> list[int]:
        if not v or any(n < 2 for n in v) or any(b <= a for a, b in pairwise(v)):
            raise ValueError("levels must be a non-empty, strictly increasing list of N >= 2")
        return v

    @model_validator(mode="after")
    def _final_level_matches_output(self) -> ResolutionConfig:
        if self.levels[-1] != self.output_width // 4:
            raise ValueError(
                f"last level N={self.levels[-1]} must equal "
                f"output_width/4 = {self.output_width // 4}"
            )
        return self


class LevelClass(_Strict):
    """One painted level: a pixel value in the levels image."""

    value: int = Field(ge=0, le=65535)
    name: str
    kind: Literal["land", "ocean"]
    base: float | None = Field(None, ge=0.0, le=1.0, description="land: 0..1 before the curve")
    depth_m: float | None = Field(None, lt=0.0, description="ocean: depth in metres (negative)")

    @model_validator(mode="after")
    def _kind_fields(self) -> LevelClass:
        if self.kind == "land" and (self.base is None or self.depth_m is not None):
            raise ValueError(f"land class {self.name!r} needs 'base' (0..1) and no 'depth_m'")
        if self.kind == "ocean" and (self.depth_m is None or self.base is not None):
            raise ValueError(f"ocean class {self.name!r} needs 'depth_m' (< 0) and no 'base'")
        return self


class InputConfig(_Strict):
    levels: Path
    classes: list[LevelClass]

    @model_validator(mode="after")
    def _classes_ok(self) -> InputConfig:
        values = [c.value for c in self.classes]
        names = [c.name for c in self.classes]
        if len(set(values)) != len(values):
            raise ValueError("class pixel values must be unique")
        if len(set(names)) != len(names):
            raise ValueError("class names must be unique")
        kinds = {c.kind for c in self.classes}
        if kinds != {"land", "ocean"}:
            raise ValueError("need at least one land class and one ocean class")
        return self


def _default_curve() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (0.2, 150.0), (0.5, 900.0), (0.8, 3500.0), (1.0, 7500.0)]


class NoiseConfig(_Strict):
    seed: int = 42
    octaves: int = Field(8, ge=1, le=16)
    base_km: float = Field(800.0, gt=0)
    gain: float = Field(0.7, gt=0, lt=1)
    warp: float = Field(0.3, ge=0)
    amp_m: float = Field(400.0, ge=0, description="noise amplitude at sea level")
    amp_per_m: float = Field(0.25, ge=0, description="extra amplitude per metre of height")


class PrepConfig(_Strict):
    sigma_km: float = Field(120.0, ge=0, description="land smoothing width")
    coast_ramp_km: float = Field(40.0, ge=0)
    coast_base: float | None = Field(
        None, ge=0, le=1, description="value at the coast; default: lowest land base"
    )
    curve: list[tuple[float, float]] = Field(default_factory=_default_curve)
    noise: NoiseConfig = NoiseConfig()

    @field_validator("curve")
    @classmethod
    def _curve_ok(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        xs = [p[0] for p in v]
        ys = [p[1] for p in v]
        if len(v) < 2 or any(b <= a for a, b in pairwise(xs)):
            raise ValueError("curve needs >= 2 points with strictly increasing x")
        if any(b < a for a, b in pairwise(ys)):
            raise ValueError("curve heights must not decrease")
        return v


class OceanConfig(_Strict):
    sigma_km: float = Field(150.0, ge=0)
    shelf_depth_m: float = Field(-130.0, lt=0)
    shelf_width_km: float = Field(80.0, gt=0)
    slope_width_km: float = Field(150.0, gt=0)
    floor_noise_m: float = Field(150.0, ge=0)


class VariationConfig(_Strict):
    """A smooth random field that scales a parameter across the globe (D26).

    The factor is ``contrast ** (v / 2)`` with ``v = tanh(sharpness * z)`` and
    ``z`` unit-variance fBm, so the factor stays within 1/sqrt(contrast) ..
    sqrt(contrast) and its geometric mean is about 1. ``contrast = 1`` is off.
    """

    contrast: float = Field(1.0, ge=1, description="max/min ratio of the factor (1 = off)")
    scale_km: float = Field(1000.0, gt=0, description="largest feature size")
    octaves: int = Field(4, ge=1, le=12)
    sharpness: float = Field(1.0, gt=0, description="higher = sharper boundaries")


# One subclass per field so a partial table (e.g. only ``scale_km``) keeps that
# field's own defaults instead of falling back to "off".
class ErodibilityVariation(VariationConfig):
    contrast: float = Field(8.0, ge=1)
    scale_km: float = Field(300.0, gt=0)
    octaves: int = Field(6, ge=1, le=12)
    sharpness: float = Field(1.5, gt=0)


class RainfallVariation(VariationConfig):
    contrast: float = Field(4.0, ge=1)
    scale_km: float = Field(1500.0, gt=0)


class UpliftVariation(VariationConfig):
    contrast: float = Field(3.0, ge=1)
    scale_km: float = Field(800.0, gt=0)


class StrataConfig(_Strict):
    """Layered rock (D33): hardness alternates with rock elevation in bands.

    Band position is the rock's elevation (height minus uplift so far) plus a
    smooth fold offset, so layers tilt and bend across a region. Hard bands
    form cliffs, ledges and mesa caps; soft bands form benches and slopes.
    """

    contrast: float = Field(4.0, ge=1, description="hard/soft erodibility ratio (1 = off)")
    thickness_m: float = Field(800.0, gt=0, description="one hard + one soft band")
    sharpness: float = Field(2.0, gt=0, description="higher = sharper band edges")
    fold_amp_m: float = Field(1500.0, ge=0, description="vertical size of folds")
    fold_scale_km: float = Field(800.0, gt=0, description="horizontal size of folds")


class ErodeConfig(_Strict):
    enabled: bool = True
    k: float = Field(7e-7, gt=0, description="erodibility K (1/yr * m^(1-2m))")
    m: float = Field(0.5, gt=0, le=1)
    dt_yr: float = Field(1e5, gt=0)
    uplift_per_km_m_per_yr: float = Field(
        1.3e-4, ge=0, description="uplift per km of painted height (m/yr)"
    )
    kappa_m2_per_yr: float = Field(1.0, ge=0, description="hillslope diffusivity")
    iterations: list[int] = Field(default_factory=lambda: [200, 100])
    epsilon_m: float = Field(1e-3, gt=0)
    restore_heights: Literal["local", "none"] = "local"
    restore_sigma_km: float = Field(200.0, gt=0, description="region size for the height restore")
    peak_cap_ratio: float = Field(
        1.1, ge=0, description="compress land above this x prepared height (0 = off)"
    )
    peak_cap_soft: float = Field(0.1, gt=0, description="headroom above the cap, as a fraction")
    river_threshold_km2: float = Field(50_000.0, gt=0)
    max_slope_deg: float = Field(
        8.0, ge=0, lt=90, description="landslide slope limit at cell scale (0 = off)"
    )
    deposition_g: float = Field(
        1.0, ge=0, description="sediment deposition coefficient G (0 = off)"
    )
    deposition_passes: int = Field(
        5, ge=2, le=50, description="Gauss-Seidel passes per step when depositing (>= 2)"
    )
    flow_routing: Literal["rho8", "d8"] = Field(
        "rho8", description="rho8 breaks straight grid-aligned channels (D32)"
    )
    strata: StrataConfig = StrataConfig()
    erodibility: ErodibilityVariation = ErodibilityVariation()
    rainfall: RainfallVariation = RainfallVariation()
    uplift: UpliftVariation = UpliftVariation()

    @field_validator("iterations")
    @classmethod
    def _iters_ok(cls, v: list[int]) -> list[int]:
        if not v or any(i < 0 for i in v):
            raise ValueError("iterations must be a non-empty list of counts >= 0")
        return v


class CoastConfig(_Strict):
    mode: Literal["locked"] = "locked"
    min_land_height_m: float = Field(1.0, gt=0)
    max_sea_height_m: float = Field(-1.0, lt=0)


Format = Literal["exr32", "png16", "tiff16", "tiff32"]


def _default_formats() -> list[Format]:
    return ["exr32", "png16"]


class OutputConfig(_Strict):
    formats: list[Format] = Field(default_factory=_default_formats)
    png_range_m: tuple[float, float] = (-11000.0, 9000.0)
    save_state: bool = Field(True, description="save state.npz so `globerode report` can rerun")

    @field_validator("png_range_m")
    @classmethod
    def _range_ok(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[1] <= v[0]:
            raise ValueError("png_range_m must be [low, high] with high > low")
        return v


class ReportConfig(_Strict):
    """Check thresholds. Hard checks fail the run (exit code 1); others warn."""

    seam_ratio: tuple[float, float] = (0.85, 1.15)
    hack_exponent: tuple[float, float] = (0.45, 0.65)
    horton_ratio: tuple[float, float] = (3.0, 5.0)
    max_height_ratio: tuple[float, float] = (0.85, 1.25)
    hard_checks: list[Literal["coast", "drainage", "seams", "hack", "horton", "max_height"]] = (
        Field(default_factory=lambda: ["coast", "drainage", "seams"])  # type: ignore[arg-type]
    )


class BlenderConfig(_Strict):
    exaggeration: float = Field(20.0, gt=0, description="vertical exaggeration in renders")


class Config(_Strict):
    name: str = "world"
    planet: PlanetConfig = PlanetConfig()
    resolution: ResolutionConfig = ResolutionConfig()
    input: InputConfig
    prep: PrepConfig = PrepConfig()
    ocean: OceanConfig = OceanConfig()
    erode: ErodeConfig = ErodeConfig()
    coast: CoastConfig = CoastConfig()
    output: OutputConfig = OutputConfig()
    report: ReportConfig = ReportConfig()
    blender: BlenderConfig = BlenderConfig()

    @model_validator(mode="after")
    def _iterations_match_levels(self) -> Config:
        if len(self.erode.iterations) != len(self.resolution.levels):
            raise ValueError(
                f"erode.iterations has {len(self.erode.iterations)} entries but "
                f"resolution.levels has {len(self.resolution.levels)}; give one count per level"
            )
        return self

    @property
    def radius_m(self) -> float:
        return self.planet.radius_km * 1000.0

    def class_by_value(self) -> dict[int, LevelClass]:
        return {c.value: c for c in self.input.classes}

    def preview(self) -> Config:
        """The same config at preview size: 2048-wide output, single level N=512."""
        data = self.model_dump()
        data["resolution"] = {"output_width": 2048, "levels": [512]}
        data["erode"]["iterations"] = [self.erode.iterations[0]]
        return Config.model_validate(data)


def _parse_value(text: str) -> Any:
    """Parse an override value as TOML; fall back to a bare string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides to a nested dict (returns a new dict)."""
    import copy

    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        key, text = item.split("=", 1)
        parts = key.strip().split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {item!r}: {part!r} is not a section")
        node[parts[-1]] = _parse_value(text.strip())
    return out


def load_config(path: Path, overrides: Sequence[str] = ()) -> Config:
    """Load, override, validate. Raises ConfigError with a readable message."""
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from None
    data = apply_overrides(data, overrides)
    try:
        cfg = Config.model_validate(data)
    except ValidationError as e:
        lines = [f"{path}: invalid config:"]
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from None
    levels = cfg.input.levels if cfg.input.levels.is_absolute() else path.parent / cfg.input.levels
    return cfg.model_copy(
        update={"input": cfg.input.model_copy(update={"levels": levels.resolve()})}
    )


def dump_toml(cfg: Config) -> str:
    """Write a resolved config back out as TOML (for the run directory)."""
    data = cfg.model_dump(mode="json")
    return _to_toml(data)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v)} to TOML")


def _to_toml(data: dict[str, Any], prefix: str = "") -> str:
    lines: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    arrays: list[tuple[str, list[dict[str, Any]]]] = []
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, dict):
            tables.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            arrays.append((k, v))
        else:
            lines.append(f"{k} = {_toml_value(v)}")
    out = "\n".join(lines) + ("\n" if lines else "")
    for k, v in tables:
        name = f"{prefix}{k}"
        out += f"\n[{name}]\n" + _to_toml(v, prefix=f"{name}.")
    for k, items in arrays:
        name = f"{prefix}{k}"
        for item in items:
            out += f"\n[[{name}]]\n" + _to_toml(item, prefix=f"{name}.")
    return out
