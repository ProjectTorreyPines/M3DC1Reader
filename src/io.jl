# io.jl — M3D-C1 `C1.h5` file reader (format-specific backend).
#
# Abstracts the on-disk layout of an M3D-C1 C1.h5 hub file: a per-time-group
# mesh (`time_NNN/mesh`), per-time-group 80-coef Hermite fields
# (`time_NNN/fields/*`), and a run-level scalar time series (`scalars/*`)
# indexed by each slice's stored `ntimestep`.

_ts_name(ts::Integer) = "time_$(lpad(ts, 3, '0'))"

"""
    M3DC1File(path; ts_ref=nothing)

Open an M3D-C1 `C1.h5` hub file and cache its mesh-level metadata: the
plane-stacked element table, number of toroidal planes, toroidal period and
plane angles, and the `b0_norm` / `n0_norm` normalization constants. The mesh
is identical across time slices, so it is read once from a reference slice
(`ts_ref`, default = first available time group). Field data are read on
demand via [`read_timeslice`](@ref) / [`read_field`](@ref).

# Fields
- `path::String`           — path to C1.h5
- `nplanes::Int`           — toroidal planes
- `npp::Int`               — elements per plane
- `period::Float64`        — toroidal period (rad)
- `phi_of_planes::Vector`  — plane reference angles (rad)
- `b0_norm`, `n0_norm`, `l0_norm`, `ion_mass` — normalization constants
  (Gauss, cm⁻³, cm, mᵢ/mₚ); see [`normalization`](@ref) / [`unit_factor`](@ref)
- `elems::Matrix{Float64}` — `(10, nplanes*npp)` full mesh element table
"""
struct M3DC1File
    path::String
    nplanes::Int
    npp::Int
    period::Float64
    phi_of_planes::Vector{Float64}
    b0_norm::Float64
    n0_norm::Float64
    l0_norm::Float64
    ion_mass::Float64
    elems::Matrix{Float64}
end

# First `time_NNN` group (lowest index) whose external-link target actually opens.
# C1.h5 stores each slice as an external link into a multi-GB `time_NNN.h5`; when
# only a subset of those is present locally (or one is unreadable/mid-write), the
# lowest-index *key* may be a broken link. Blindly opening it (the old behaviour)
# aborted the whole run on, e.g., a missing `time_000.h5`; instead skip to the
# first slice that resolves — the mesh/normalization are slice-independent, so any
# accessible slice serves. Errors only if NONE of the slices open.
function _first_time_group(f::HDF5.File)
    HDF5.API.h5e_set_auto(HDF5.API.H5E_DEFAULT, C_NULL, C_NULL)   # silence HDF5 error stack
    names = sort(
        [k for k in keys(f) if startswith(k, "time_") && tryparse(Int, k[6:end]) !== nothing];
        by = k -> parse(Int, k[6:end]),
    )
    isempty(names) && error("no `time_NNN` group found in $(HDF5.filename(f))")
    for k in names
        try
            close(f[k])          # force external-link resolution, then release the handle
            return k
        catch
            # broken / inaccessible external link — try the next slice
        end
    end
    return error(
        "no accessible `time_NNN` slice in $(HDF5.filename(f)) — every external link " *
            "(e.g. `$(first(names))` → its `time_*.h5`) failed to open; check the run " *
            "folder still has the `time_*.h5` files and that they are readable",
    )
end

function M3DC1File(path::AbstractString; ts_ref::Union{Nothing, Integer} = nothing)
    return h5open(String(path), "r") do f
        ts_name = ts_ref === nothing ? _first_time_group(f) : _ts_name(ts_ref)
        haskey(f, ts_name) || error("time group `$ts_name` not found in $path")
        g = f[ts_name]
        m = g["mesh"]
        elems = Float64.(read(m["elements"]))           # (10, nelms)
        nplanes = Int(read_attribute(m, "nplanes"))
        period = Float64(read_attribute(m, "period"))
        phi = Float64.(read_attribute(m, "phi"))
        b0 = Float64(read_attribute(f, "b0_norm"))   # Gauss
        n0 = Float64(read_attribute(f, "n0_norm"))   # cm^-3
        l0 = Float64(read_attribute(f, "l0_norm"))   # cm
        mi = Float64(read_attribute(f, "ion_mass"))  # mi/mp
        nelms = size(elems, 2)
        nelms % nplanes == 0 ||
            error("nelms ($nelms) not divisible by nplanes ($nplanes)")
        npp = nelms ÷ nplanes
        return M3DC1File(String(path), nplanes, npp, period, phi, b0, n0, l0, mi, elems)
    end
end

"""
    normalization(file) -> M3DNormalization

Bundle the file's normalization constants for use with [`unit_factor`](@ref) /
[`to_units`](@ref).
"""
normalization(file::M3DC1File) =
    M3DNormalization(;
    b0 = file.b0_norm, n0 = file.n0_norm,
    l0 = file.l0_norm, ion_mass = file.ion_mass
)

"""
    elems_plane(file) -> SubArray

Single-plane mesh element table `(10, npp)` (plane 1). The M3D-C1 mesh is
plane-stacked and identical across planes, so this view is the right input
for the element/grid/magaxis routines.
"""
elems_plane(file::M3DC1File) = view(file.elems, :, 1:file.npp)

"""
    list_timeslices(file) -> Vector{Int}

Sorted indices `N` of `time_NNN` slices in the file whose external-link
target file is **actually accessible** on disk. Slices listed in C1.h5 but
whose `time_NNN.h5` is missing (e.g. when only a subset has been copied
locally) are filtered out.
"""
function list_timeslices(file::M3DC1File)
    return h5open(file.path, "r") do f
        ts = Int[]
        HDF5.API.h5e_set_auto(HDF5.API.H5E_DEFAULT, C_NULL, C_NULL)   # silence HDF5 error stack
        for k in keys(f)
            startswith(k, "time_") || continue
            n = tryparse(Int, k[6:end])
            n === nothing && continue
            try
                _ = f[k]                                              # forces external-link resolution
                push!(ts, n)
            catch
                # broken link: target file missing — skip
            end
        end
        return sort!(ts)
    end
end

"""
    read_field(file, ts, field; rows=Colon()) -> Matrix{Float64}

Read the 80-coef Hermite coefficients of `field` (e.g. `:psi`, `:te`) for time
slice `ts` — shape `(nrows, nplanes*npp)`, plane-stacked, where `nrows` = 80 by
default. Pass `rows` (e.g. `1:20`) to read only a leading block of coefficients
via an HDF5 hyperslab: axisymmetric evaluation at a plane (ζ=0 — all the FSA /
q / B-map / modern-δB/B paths) touches only rows `1:20`, and legacy-`f` δB/B
needs `1:40`; skipping the ζ¹–ζ³ layers cuts both the read volume and the
Float32→Float64 conversion (the on-disk data is `Float32`). The full `1:80` is
required only for arbitrary-φ evaluation (the ASCOT5 writer's ζ-collapse).
"""
function read_field(file::M3DC1File, ts::Integer, field::Symbol; rows = Colon())
    return h5open(file.path, "r") do f
        d = f[_ts_name(ts)]["fields/$(field)"]
        Float64.(rows === Colon() ? read(d) : d[rows, :])
    end
end

"""
    read_timeslice(file, ts; fields=(:psi, :te), optional=(), rows=Colon(), rows_of=nothing) -> NamedTuple

Read one time slice. Returns the requested 80-coef fields plus the scalars
(`psi_axis`, `psi_lcfs`, `xmag`, `zmag`, and the X-points `xnull`, `znull`,
`xnull2`, `znull2`) and the slice `time`, sampled at this slice's stored
`ntimestep`. These scalars are M3D-C1's plane-1 (φ=0) values; for a
toroidally-averaged reference, recompute the axis / X-points with
[`find_critical_point`](@ref) on the averaged ψ (seed the O-point from
`(xmag, zmag)`, the X-points from `(xnull, znull)` / `(xnull2, znull2)`), or
use [`find_lcfs`](@ref) for the full boundary-flux determination. The `null`
scalars are `NaN` if the file does not store them; M3D-C1 uses `xnull ≤ 0` to
mean "no X-point tracked".

`fields` are required (error if missing); `optional` fields are read when
present and silently skipped when absent — both share the same single file
open, so reading N fields costs one external-link resolution rather than N
(the C1.h5 slice groups are external links into multi-GB `time_XXX.h5` files,
so re-opening per field dominates the cost). `rows` selects a coefficient-row
hyperslab (see [`read_field`](@ref)); pass `rows_of = fld -> …` to vary the
row range per field (e.g. legacy `f` needs `1:40` for its ∂/∂φ while everything
else needs only `1:20`).

# Returns
`(; nstep, fields::Dict{Symbol,Matrix}, psi_axis, psi_lcfs, xmag, zmag,
   xnull, znull, xnull2, znull2, time)`
"""
function read_timeslice(
        file::M3DC1File, ts::Integer; fields = (:psi, :te),
        optional = (), rows = Colon(), rows_of = nothing
    )
    _rows(fld) = rows_of === nothing ? rows : rows_of(fld)
    _slab(d, r) = Float64.(r === Colon() ? read(d) : d[r, :])
    return h5open(file.path, "r") do f
        g = f[_ts_name(ts)]
        fg = g["fields"]
        scl = f["scalars"]
        nstep = Int(read_attribute(g, "ntimestep"))
        flds = Dict{Symbol, Matrix{Float64}}()
        for fld in fields
            flds[fld] = _slab(fg[String(fld)], _rows(fld))
        end
        for fld in optional
            haskey(fg, String(fld)) || continue
            flds[fld] = _slab(fg[String(fld)], _rows(fld))
        end
        return _slice_result(flds, scl, nstep)
    end
end

# Per-slice scalar NamedTuple, shared by read_timeslice / read_timeslice!. `flds`
# is the already-populated field Dict; `scl` the open run-level scalars group.
# The X-point scalars are NaN on files that predate them.
function _slice_result(flds, scl, nstep::Integer)
    psi0 = read(scl["psi0"]);  psi_lcfs = read(scl["psi_lcfs"])
    xmag = read(scl["xmag"]);  zmag = read(scl["zmag"]);  tval = read(scl["time"])
    _scl(k) = haskey(scl, k) ? Float64(read(scl[k])[nstep + 1]) : NaN
    return (;
        nstep,
        fields = flds,
        psi_axis = Float64(psi0[nstep + 1]),
        psi_lcfs = Float64(psi_lcfs[nstep + 1]),
        xmag = Float64(xmag[nstep + 1]),
        zmag = Float64(zmag[nstep + 1]),
        xnull = _scl("xnull"),
        znull = _scl("znull"),
        xnull2 = _scl("xnull2"),
        znull2 = _scl("znull2"),
        time = Float64(tval[nstep + 1]),
    )
end

"""
    read_timeslice!(fields, staging, file, ts; required=(:psi,), optional=(),
                    rows=1:20, rows_of=nothing) -> NamedTuple

In-place variant of [`read_timeslice`](@ref) for tight per-slice loops (the
20-30-slice export). Each field's coefficient-row hyperslab is read directly
into a persistent `Float32` `staging` buffer via the low-level `h5d_read` (no
allocation) and converted into the persistent `Float64` `fields` buffer. Both
dicts are reused across calls, so the field matrices — the dominant allocation
(~28 MiB each) — are allocated ONCE for the whole loop instead of per slice,
eliminating the field-read GC pressure. Buffers grow/reallocate only when a
field's shape changes; a field present last slice but absent now is dropped
from `fields`. Non-`Float32` datasets fall back to a plain allocating read.

Returns the same NamedTuple as `read_timeslice` (with `.fields === fields`).
Reusing `fields` across slices is safe: the FSA reduction consumes the raw
coefficients synchronously (`average_toroidal_axisymmetric` allocates fresh
n=0 arrays) and never aliases a raw field matrix into its result.
"""
function read_timeslice!(
        fields::AbstractDict{Symbol, Matrix{Float64}},
        staging::AbstractDict{Symbol, Matrix{Float32}},
        file::M3DC1File, ts::Integer;
        required = (:psi,), optional = (),
        rows = 1:20, rows_of = nothing
    )
    _rows(fld) = rows_of === nothing ? rows : rows_of(fld)
    present = Set{Symbol}()
    return h5open(file.path, "r") do f
        g = f[_ts_name(ts)]
        fg = g["fields"]
        scl = f["scalars"]
        nstep = Int(read_attribute(g, "ntimestep"))
        for fld in required
            _read_field_into!(fields, staging, fg, fld, _rows(fld));  push!(present, fld)
        end
        for fld in optional
            haskey(fg, String(fld)) || continue
            _read_field_into!(fields, staging, fg, fld, _rows(fld));  push!(present, fld)
        end
        for k in collect(keys(fields))       # a field gone this slice must not leak a stale buffer
            k in present || delete!(fields, k)
        end
        return _slice_result(fields, scl, nstep)
    end
end

# Read the `rows` coefficient-row hyperslab of dataset `fg[fld]` into the
# persistent Float64 `fields[fld]`, staging through a persistent Float32 buffer.
# On-disk fields are Float32, so the fast path selects the hyperslab with the
# low-level H5D API and reads straight into the reused buffer (zero allocation);
# any non-Float32 dataset falls back to a plain allocating read for correctness.
function _read_field_into!(fields, staging, fg, fld::Symbol, rows)
    d = fg[String(fld)]
    ncols = size(d, 2)
    nrow = rows === Colon() ? size(d, 1) : length(rows)
    if eltype(d) === Float32
        s = get(staging, fld, nothing)
        if s === nothing || size(s) != (nrow, ncols)
            s = Matrix{Float32}(undef, nrow, ncols);  staging[fld] = s
        end
        off0 = rows === Colon() ? 0 : Int(first(rows)) - 1
        # HDF5 C dimensions are the reverse of Julia's: a Julia (nrow_total,
        # ncols) dataset is C (ncols, nrow_total). Select `nrow` rows starting
        # at `off0` along the fast (Julia dim-1) axis.
        fspace = HDF5.dataspace(d)
        mspace = HDF5.dataspace((nrow, ncols))
        try
            HDF5.API.h5s_select_hyperslab(
                fspace, HDF5.API.H5S_SELECT_SET,
                HDF5.API.hsize_t[0, off0], C_NULL, HDF5.API.hsize_t[ncols, nrow], C_NULL
            )
            HDF5.API.h5d_read(
                d, HDF5.datatype(Float32), mspace, fspace,
                HDF5.API.H5P_DEFAULT, s
            )
        finally
            close(mspace);  close(fspace)
        end
        b = get(fields, fld, nothing)
        if b === nothing || size(b) != (nrow, ncols)
            b = Matrix{Float64}(undef, nrow, ncols);  fields[fld] = b
        end
        b .= s
    else
        raw = rows === Colon() ? read(d) : d[rows, :]
        fields[fld] = Float64.(raw)
    end
    return nothing
end

"""
    scalar_names(file) -> Vector{String}

Sorted dataset names under the run-level `scalars/*` group — the 0D time
histories, e.g. `"toroidal_current"`, `"W_P"`, `"loop_voltage"`, `"radiation"`.
Empty if the file has no `scalars` group.
"""
function scalar_names(file::M3DC1File)
    return h5open(file.path, "r") do f
        haskey(f, "scalars") || return String[]
        sort!(collect(String, keys(f["scalars"])))
    end
end

"""
    read_scalar(file, name) -> Vector{Float64}
    read_scalar(file, name, ts) -> Float64

Read the `scalars/<name>` 0D time history, in M3D-C1 normalized units (convert
with [`unit_factor`](@ref): `:current` for `"toroidal_current"`, `:energy` for
`"W_P"`/`"E_P"`, `:voltage` for `"loop_voltage"`, `:power` for `"radiation"`,
…). Traces are indexed by simulation step — entry `n+1` belongs to
`ntimestep = n` — so the 2-arg form covers every step, not just the field
slices. The 3-arg form returns the value at time slice `ts`'s stored
`ntimestep`, the same sampling [`read_timeslice`](@ref) uses for its scalars.
See [`scalar_names`](@ref) for what a file provides.
"""
function read_scalar(file::M3DC1File, name::AbstractString)
    return h5open(file.path, "r") do f
        Float64.(read(_scalar_ds(f, name, file.path)))
    end
end

function read_scalar(file::M3DC1File, name::AbstractString, ts::Integer)
    return h5open(file.path, "r") do f
        nstep = Int(read_attribute(f[_ts_name(ts)], "ntimestep"))
        v = read(_scalar_ds(f, name, file.path))
        nstep + 1 ≤ length(v) ||
            error("scalar `$name` has $(length(v)) entries but slice $ts stores ntimestep=$nstep")
        Float64(v[nstep + 1])
    end
end

function _scalar_ds(f::HDF5.File, name::AbstractString, path)
    haskey(f, "scalars") || error("no `scalars` group in $path")
    scl = f["scalars"]
    haskey(scl, name) ||
        error("scalar `$name` not found in $path; see scalar_names(file)")
    return scl[name]
end

"""
    limiter_points(file) -> (; xlim, zlim, xlim2, zlim2)

The M3D-C1 input limiter points, stored as C1.h5 root attributes (file
version ≥ 18). Semantics follow M3D-C1's `lcfs`: `xlim == 0` means "no
limiter" (skip the limiter candidates in [`find_lcfs`](@ref)). Attributes
missing from older files are returned as `NaN`.
"""
function limiter_points(file::M3DC1File)
    return h5open(file.path, "r") do f
        a = attrs(f)
        g(k) = haskey(a, k) ? Float64(a[k]) : NaN
        return (;
            xlim = g("xlim"), zlim = g("zlim"),
            xlim2 = g("xlim2"), zlim2 = g("zlim2"),
        )
    end
end

"""
    read_pellets(file) -> NamedTuple or nothing

Read the root-level `/pellet` group (the M3D-C1 pellet/SPI source model;
`output.f90`): per-pellet, per-step traces as `(npellets, nsteps)` matrices in
M3D-C1 normalized units — lengths in l0, velocities in v0, and `rate` (and
`rate_D2`) as the TOTAL deposited-particle rate in n0·l0³ particles per t0
(`pellet_distribution` is a unit-integral cloud, so the rate amplitude is the
whole-pellet ablation rate). `mix` is the D2 mole fraction
(moles D2)/(moles D2 + moles impurity) (`pellet.f90`). Datasets a file lacks
come back `nothing`; returns `nothing` when there is no `/pellet` group.
"""
function read_pellets(file::M3DC1File)
    return h5open(file.path, "r") do f
        haskey(f, "pellet") || return nothing
        g = f["pellet"]
        np = haskey(attrs(f), "npellets") ? Int(read_attribute(f, "npellets")) : -1
        function rd(k)
            haskey(g, k) || return nothing
            A = Float64.(read(g[k]))
            A isa Vector && (A = reshape(A, 1, :))                 # single pellet
            (np > 0 && size(A, 1) != np && size(A, 2) == np) && (A = permutedims(A))
            return A
        end
        return (;
            npellets = np,
            rate = rd("pellet_rate"), rate_D2 = rd("pellet_rate_D2"),
            r_p = rd("r_p"), r = rd("pellet_r"), phi = rd("pellet_phi"),
            z = rd("pellet_z"), velr = rd("pellet_velr"),
            velphi = rd("pellet_velphi"), velz = rd("pellet_velz"),
            cloud = rd("cloud_pel"), mix = rd("pellet_mix"),
        )
    end
end

# Whether the file carries f' = ∂f/∂φ as its own `fp` dataset (the modern
# convention; the `f` dataset is then the potential f itself). Rule from
# fusion-io m3dc1_source.cpp:37-40 (m3dc1_field.cpp:553-557 loads "fp" when
# set, "f" otherwise):
#   ifprime = 1  iff  version ≥ 38, or (version ≥ 35 and numvar > 1).
# See docs/deltab_over_b.md §1.
function _fprime_stored(file::M3DC1File)
    return h5open(file.path, "r") do f
        rd(a) = haskey(attrs(f), a) ? Int(read_attribute(f, a)) : 0
        version = rd("version");  numvar = rd("numvar")
        return version >= 38 || (version >= 35 && numvar > 1)
    end
end

"""
    te_to_eV_factor(file) -> Float64

Convenience: the Te→eV conversion factor for this file's normalization,
equivalent to `unit_factor(normalization(file), :temperature; system=:si)`.
"""
te_to_eV_factor(file::M3DC1File) = unit_factor(normalization(file), :temperature; system = :si)
