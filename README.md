[![CI](https://github.com/ProjectTorreyPines/M3DC1Reader/actions/workflows/CI.yml/badge.svg)](https://github.com/ProjectTorreyPines/M3DC1Reader/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/projecttorreypines/m3dc1reader/graph/badge.svg?token=JDIwgzYKI2)](https://codecov.io/github/projecttorreypines/m3dc1reader)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

# M3DC1Reader.jl

Read M3D-C1 `C1.h5` simulation output and work with its native C¹ reduced
quintic Hermite finite-element fields directly in Julia — reading the mesh,
fields, and scalars; evaluating fields and their derivatives at arbitrary
points or grids; locating critical points; and applying unit normalizations.

A lightweight, single-core, pure-Julia toolkit that plays a role analogous to
parts of `fusion-io` and of the PPPL `visM3D` MATLAB tools (their
non-visualization core), and serves as the M3D-C1 counterpart to the NIMROD
workflow in `disruption_database_tools`.

The first end-to-end use case implemented is reducing a full 3D field to an
IMAS-compatible **2D** (rectilinear R-Z) field and **1D** (radial profile vs
ρ_pol = √ψ_N) — operating directly on the native finite elements, with no
intermediate resampling. More general field/coordinate/unit utilities
(following `visM3D`) may be added over time.

## Pipeline

```
raw 3D mesh + 80-coef Hermite fields per element
  │  average_toroidal_axisymmetric      (n=0 toroidal Fourier mode)
  ▼
axisymmetric 80-coef per element (single R-Z plane)
  │  interpolate_axisym_to_grid         (analytic FEM evaluation)
  ▼
2D field on a rectilinear R-Z grid
  │  reduce_1d_psi_func                  (adjoint scattered→grid projection)
  ▼
1D profile vs ρ_pol = √ψ_N
```

The magnetic-axis reference for ψ_N is found by `find_axis_newton`, a Julia
reimplementation of M3D-C1's internal `magaxis` Newton iteration on the FEM
polynomial (`diagnostics.f90:magaxis`, `imethod=0`).

## Install / activate

```julia
using Pkg
Pkg.activate("/home/myoo/Prog/M3DC1Reader.jl")
Pkg.instantiate()           # resolves HDF5, FastInterpolations
```

`FastInterpolations.jl` is a local/unregistered dependency; if `instantiate`
cannot find it, `Pkg.develop(path=...)` it first.

## Quickstart

```julia
using M3DC1Reader

file  = M3DC1File("/scratch/gpfs/myoo/m3d_smoke/C1.h5")   # caches mesh + norms
slice = read_timeslice(file, 24; fields = (:psi, :te))

# (1) n=0 toroidal average
psi_ax = average_toroidal_axisymmetric(slice.fields[:psi], file.nplanes)
te_ax  = average_toroidal_axisymmetric(slice.fields[:te] .* te_to_eV_factor(file),
                                       file.nplanes)

# self-consistent n=0 magnetic axis
ep = elems_plane(file)
ax = find_axis_newton(psi_ax, ep, slice.xmag, slice.zmag)

# (2) 2D rectilinear field
Rg = collect(range(extrema(ep[5, :])..., length = 100))
Zg = collect(range(extrema(ep[6, :])..., length = 100))
id_map = build_grid_to_element_map(Rg, Zg, ep)
te_2d  = interpolate_axisym_to_grid(te_ax, ep, Rg, Zg; id_map = id_map)

# (3) 1D Te(ρ_pol)
psi_2d = interpolate_axisym_to_grid(psi_ax, ep, Rg, Zg; id_map = id_map)
ψn   = psi_to_psi_norm.(vec(psi_2d), ax.ψ, slice.psi_lcfs)
ρpol = psi_n_to_rho_pol.(ψn)
prof = reduce_1d_psi_func(ρpol, vec(te_2d); n_bins = 60,
                          psi_range = (0.0, 1.2), adj = :linear)
# prof.psi_grid, prof.func_bin
```

See `examples/quickstart.jl`.

## Module layout

| File | Scope |
|------|-------|
| `src/io.jl` | `C1.h5` reader (`M3DC1File`, `read_timeslice`, …) — **M3D-C1-specific** |
| `src/elements.jl` | reduced quintic Hermite FEM math, element location — **M3D-C1-specific** |
| `src/reductions.jl` | toroidal average, 2D evaluation, 1D adjoint reduction |
| `src/magaxis.jl` | Newton O-point finder (`find_axis_newton`) |
| `src/units.jl` | selectable per-quantity unit normalization (M3D / cgs / SI) |
| `src/normalization.jl` | flux coordinates ψ_N, ρ_pol = √ψ_N |

The reduction / magaxis / normalization layers are kept independent of the
M3D-C1 element & IO layers, so they can be lifted into a multi-code package
(e.g. a future `MHDFieldIO`) when a NIMROD backend is added — at which point
`M3DC1Reader` becomes one backend.

## Units

M3D-C1 stores fields in its own dimensionless normalization. Convert any
quantity to **cgs** or **SI** with a data-driven registry (no per-call switch):

```julia
norm = normalization(file)               # b0, n0, l0, ion_mass (+ derived V0, T0)

unit_factor(norm, :temperature)          # eV per M3D unit  (system=:si default)
unit_factor(norm, :magnetic_field; system=:cgs)   # Gauss per M3D unit
to_units(norm, slice.fields[:te], :temperature)   # Te field → eV
unit_label(:magnetic_field; system=:si)  # "T"
available_quantities()                   # :density, :velocity, :temperature, …
```

Systems: `:m3d` (factor 1), `:cgs`, `:si` (alias `:mks`). Factors follow the
visM3D `Units_cgs_mks` table. `te_to_eV_factor(file)` is a shorthand for
`unit_factor(normalization(file), :temperature)`.

## Coordinates

- The 1D radial coordinate is **`ρ_pol = √ψ_N`**, matching the
  `rho_pol_norm` grid used by the NIMROD IMAS path.
- `psi_axis` / `psi_lcfs` from `scalars` are M3D-C1's **plane-1 (φ=0)** values;
  for a toroidally-averaged reference use `find_axis_newton` on the averaged ψ
  (the two differ by ~1% of |Δψ| during disruptions).

## Tests

```julia
Pkg.test("M3DC1Reader")
```

Unit tests use a synthetic single-element mesh (no data file needed). If a
`C1.h5` is available (default `/scratch/gpfs/myoo/m3d_smoke/C1.h5`, or set
`ENV["M3DC1_TEST_FILE"]`), an end-to-end slice test also runs.
