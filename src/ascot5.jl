# ascot5.jl — ASCOT5 input-file writer (Pipeline ①).
#
# Writes one a5py-compatible input HDF5 per time slice: bfield/B_3DS (3D
# field snapshot), plasma/plasma_1D (FSA background), wall/wall_2D (mesh
# boundary), efield/E_TC (zero). Layouts copied verbatim from the ASCOT5
# sources (a5py/ascot5io/{bfield,plasma,wall,efield}.py and
# coreio/fileapi.py) — see docs/ascot5_writer.md.
#
# Time handling: ASCOT5 traces particles in a STATIC field, so each MHD
# slice gets its own input file (one run per snapshot). The default name is keyed
# on the slice INDEX, mirroring M3D-C1's own `time_%03d.h5` layout (time_024.h5 ↔
# ascot_input_024.h5) so inputs pair 1:1 with their source slice. The DB
# pipeline's step-from-filename parsers read the FIRST digits after `ascot` as
# the MHD-time key, so no digit may touch the prefix (`ascot5_…` would be read as
# step 5, disagreeing with the end-anchored parser) — hence `ascot_input_`.

using Dates: now, format

"""
    _collapse_zeta(coef, ζ; dphi=false) -> Matrix (20 × nelms)

Collapse an 80-coefficient 3D Hermite block to the effective 20-coefficient
poloidal representation at local toroidal angle `ζ` (radians from the wedge
start): `c_eff = c₁ + ζc₂ + ζ²c₃ + ζ³c₄` over the four 20-row layers.
With `dphi=true` return ∂/∂ζ instead (`c₂ + 2ζc₃ + 3ζ²c₄`) — the toroidal
derivative used for legacy files whose `f` dataset is the potential itself.
"""
function _collapse_zeta(coef::AbstractMatrix, ζ::Real; dphi::Bool = false)
    size(coef, 1) >= 80 || error("_collapse_zeta: expected ≥80 coefficient rows")
    c1 = view(coef, 1:20, :);  c2 = view(coef, 21:40, :)
    c3 = view(coef, 41:60, :);  c4 = view(coef, 61:80, :)
    return dphi ? c2 .+ (2ζ) .* c3 .+ (3ζ^2) .* c4 :
        c1 .+ ζ .* c2 .+ ζ^2 .* c3 .+ ζ^3 .* c4
end

"""
    ascot5_bfield(file, ts; nR=150, nZ=150, nphi=0, efield=false) -> NamedTuple

Assemble the ASCOT5 `B_3DS` field data for one time slice, in SI units:

- `Rg`, `Zg` [m]: uniform grids over the mesh bounding box, padded by
  `margin` (fraction of the R/Z span; default 2% — just enough that the
  wall polyline is strictly inside the grid); `phi_deg`:
  the `nphi` periodic toroidal nodes (`linspace(0, 360, nphi+1)[1:end-1]`,
  the a5py storage convention). `nphi=0` (default) means `4 × nplanes` —
  8 points per wavelength of the highest mode the run resolves.
- `psi` [Wb/rad] (nR×nZ): the n=0 poloidal-flux map (per-radian — exactly
  ASCOT5's `Vs/m`), with `psi0`/`psi1` (axis/separatrix) and `axisr`/`axisz`
  from the per-slice `find_lcfs` recomputation.
- `br`, `bz` [T] (nR×nphi×nZ): **perturbation only** — B_3DS's convention is
  that the equilibrium poloidal field is reconstructed from ∇ψ, so the n=0
  part is subtracted here. `bphi` [T] is the total toroidal field.
- `er`, `ephi`, `ez` [V/m] (nR×nphi×nZ): the **total** electric field from
  M3D-C1's stored `E_R`/`E_PHI`/`E_Z` (evaluated as values, not gradients),
  returned only when `efield=true` and those datasets exist (else `nothing`).
  Unlike B_3DS there is no equilibrium reconstruction to pair with (ASCOT5's
  E_3D is standalone), so E is written in full — no n=0 subtraction.

The 3D field is evaluated from the exact FEM representation
`B = ∇ψ×∇φ + F∇φ − ∇⊥f′` at each φ node via [`_collapse_zeta`](@ref)
(docs/m3dc1_field_representation.md). Grid points outside the mesh are
`NaN` — the fill strategy for the rectangle's corners is deliberately left
to the caller/operator for now (see docs/ascot5_writer.md §fill).
"""
function ascot5_bfield(
        file::M3DC1File, ts::Integer;
        nR::Integer = 150, nZ::Integer = 150, nphi::Integer = 0,
        margin::Real = 0.02, efield::Bool = false
    )
    nplanes = file.nplanes
    npp = file.npp
    nphi <= 0 && (nphi = 4 * nplanes)
    norm = normalization(file)
    ulen = unit_factor(norm, :length; system = :si)
    ub = unit_factor(norm, :magnetic_field; system = :si)
    uflux = unit_factor(norm, :magnetic_flux; system = :si)
    ue = unit_factor(norm, :electric_field; system = :si)   # V/m = v0·b0 (fusion-io Phi0/L0)

    ep = elems_plane(file)
    Rb, Zb = mesh_boundary_rz(ep)
    ΔR = maximum(Rb) - minimum(Rb);  ΔZ = maximum(Zb) - minimum(Zb)
    Rg_n = collect(
        range(
            max(minimum(Rb) - margin * ΔR, 0.01 * ΔR),
            maximum(Rb) + margin * ΔR; length = nR
        )
    )
    Zg_n = collect(
        range(
            minimum(Zb) - margin * ΔZ,
            maximum(Zb) + margin * ΔZ; length = nZ
        )
    )
    id_map = build_grid_to_element_map(Rg_n, Zg_n, ep)

    # ASCOT B_3DS is evaluated at arbitrary φ (ζ-collapse), so it needs the
    # FULL 80-coefficient Hermite blocks — not the ζ=0 leading rows the FSA
    # export gets away with.
    sl = read_timeslice(file, ts; fields = (:psi, :I))
    psi3 = sl.fields[:psi];  I3 = sl.fields[:I]
    fprime = _fprime_stored(file)
    f3 = _try_field(file, ts, fprime ? :fp : :f; rows = Colon())

    # Electric field (E_3D). M3D-C1 stores E_R/E_PHI/E_Z directly as value fields
    # (same 80-coef Hermite blocks); only read when requested and present, else the
    # writer falls back to a zero E_TC. These are the TOTAL field — no n=0 split.
    ER3 = efield ? _try_field(file, ts, :E_R; rows = Colon()) : nothing
    EP3 = efield ? _try_field(file, ts, :E_PHI; rows = Colon()) : nothing
    EZ3 = efield ? _try_field(file, ts, :E_Z; rows = Colon()) : nothing
    have_E = ER3 !== nothing && EP3 !== nothing && EZ3 !== nothing

    # n=0 ψ map + axis/separatrix flux (same recipe as reduce_axisym_slice)
    psi_av = average_toroidal_axisymmetric(psi3, nplanes)
    pg0 = interpolate_axisym_gradient_to_grid(psi_av, ep, Rg_n, Zg_n; id_map = id_map)
    I_av = average_toroidal_axisymmetric(I3, nplanes)
    I0v = interpolate_axisym_to_grid(I_av, ep, Rg_n, Zg_n; id_map = id_map)
    ψ0 = Float64(sl.psi_axis);  ψ1 = Float64(sl.psi_lcfs)
    R_ax = Float64(sl.xmag);    Z_ax = Float64(sl.zmag)
    lim = limiter_points(file)
    lc = try
        find_lcfs(
            psi_av, ep; xmag = R_ax, zmag = Z_ax,
            xnull = sl.xnull, znull = sl.znull,
            xnull2 = sl.xnull2, znull2 = sl.znull2,
            lim.xlim, lim.zlim, lim.xlim2, lim.zlim2
        )
    catch err
        @warn "ascot5_bfield: axis/LCFS recomputation failed; using stored plane-1 scalars" exception = err
        nothing
    end
    if lc !== nothing
        ψ0 = lc.psi_axis;  R_ax = lc.axis.R;  Z_ax = lc.axis.Z
        isfinite(lc.psi_bound) && (ψ1 = lc.psi_bound)
    end

    Δφ = file.period / nplanes
    phis = [(j - 1) * file.period / nphi for j in 1:nphi]
    br = fill(NaN, nR, nphi, nZ)
    bphi = fill(NaN, nR, nphi, nZ)
    bz = fill(NaN, nR, nphi, nZ)
    er = have_E ? fill(NaN, nR, nphi, nZ) : nothing
    ephi = have_E ? fill(NaN, nR, nphi, nZ) : nothing
    ez = have_E ? fill(NaN, nR, nphi, nZ) : nothing
    for (j, φ) in enumerate(phis)
        p = min(floor(Int, φ / Δφ) + 1, nplanes)
        ζ = φ - (p - 1) * Δφ
        cols = ((p - 1) * npp + 1):(p * npp)
        ψeff = _collapse_zeta(view(psi3, :, cols), ζ)
        Ieff = _collapse_zeta(view(I3, :, cols), ζ)
        pgp = interpolate_axisym_gradient_to_grid(ψeff, ep, Rg_n, Zg_n; id_map = id_map)
        Iv = interpolate_axisym_to_grid(Ieff, ep, Rg_n, Zg_n; id_map = id_map)
        fg = nothing
        if f3 !== nothing
            feff = _collapse_zeta(view(f3, :, cols), ζ; dphi = !fprime)
            fg = interpolate_axisym_gradient_to_grid(feff, ep, Rg_n, Zg_n; id_map = id_map)
        end
        @inbounds for k in 1:nZ, i in 1:nR
            R = Rg_n[i]
            brt = -pgp.dZ[i, k] / R
            bzt = pgp.dR[i, k] / R
            if fg !== nothing
                brt -= fg.dR[i, k]
                bzt -= fg.dZ[i, k]
            end
            # B_3DS: br/bz carry only the non-equilibrium part (ψ carries n=0)
            br[i, j, k] = (brt - (-pg0.dZ[i, k] / R)) * ub
            bz[i, j, k] = (bzt - (pg0.dR[i, k] / R)) * ub
            bphi[i, j, k] = (Iv[i, k] / R) * ub
        end
        if have_E
            # E is a stored value field (not a gradient); total field, written in
            # full. Same φ-collapse + on-grid value evaluation as I, ×V/m factor.
            ERv = interpolate_axisym_to_grid(_collapse_zeta(view(ER3, :, cols), ζ), ep, Rg_n, Zg_n; id_map = id_map)
            EPv = interpolate_axisym_to_grid(_collapse_zeta(view(EP3, :, cols), ζ), ep, Rg_n, Zg_n; id_map = id_map)
            EZv = interpolate_axisym_to_grid(_collapse_zeta(view(EZ3, :, cols), ζ), ep, Rg_n, Zg_n; id_map = id_map)
            @inbounds for k in 1:nZ, i in 1:nR
                er[i, j, k] = ERv[i, k] * ue
                ephi[i, j, k] = EPv[i, k] * ue
                ez[i, j, k] = EZv[i, k] * ue
            end
        end
    end

    return (;
        Rg = Rg_n .* ulen, Zg = Zg_n .* ulen,
        phi_deg = phis .* (360 / file.period),
        psi = Array{Float64}(pg0.val) .* uflux,
        psi0 = ψ0 * uflux, psi1 = ψ1 * uflux,
        axisr = R_ax * ulen, axisz = Z_ax * ulen,
        br = br, bphi = bphi, bz = bz,
        er = er, ephi = ephi, ez = ez,        # total E [V/m], or nothing if not requested/absent
        bphi0_rz = (I0v ./ Rg_n) .* ub,       # n=0 Bφ map (for diagnostics)
        time = sl.time * unit_factor(norm, :time; system = :si),
        nstep = sl.nstep,
    )
end

# ---------------------------------------------------------------------------
# a5py HDF5 conventions (coreio/fileapi.py): every input lives in
# <parent>/<TYPE>_<qid> with a random zero-padded 10-digit uint32 QID,
# byte-string `description`/`date` attrs, and the parent carries
# `active` = qid of the input to use. Attributes must be ASCII-typed so
# h5py hands them back as `bytes` (a5py calls `.decode()` on them).
# ---------------------------------------------------------------------------

function _a5_ascii_attr!(obj, name::AbstractString, s::AbstractString)
    buf = Vector{UInt8}(codeunits(String(s)))
    dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dt, max(length(buf), 1))
    HDF5.API.h5t_set_cset(dt, HDF5.API.H5T_CSET_ASCII)
    attr = create_attribute(obj, String(name), dt, dataspace(()))
    try
        # low-level write: the fixed-length string is ONE element of `size(dt)`
        # bytes, which the high-level length check would reject
        HDF5.API.h5a_write(attr, dt, buf)
    finally
        close(attr)
    end
    return nothing
end

_a5_qid() = lpad(string(rand(UInt32)), 10, '0')

function _a5_group(
        f::HDF5.File, parent::String, gtype::String;
        desc::AbstractString = "M3DC1Reader export", qid::String = _a5_qid()
    )
    pg = haskey(f, parent) ? f[parent] : create_group(f, parent)
    g = create_group(pg, "$(gtype)_$(qid)")
    _a5_ascii_attr!(g, "description", desc)
    _a5_ascii_attr!(g, "date", format(now(), "yyyy-mm-dd HH:MM:SS"))
    haskey(HDF5.attrs(pg), "active") || _a5_ascii_attr!(pg, "active", qid)
    return g
end

# a5py stores everything through h5py (row-major); HDF5.jl reverses dims on
# write, so a Julia (a,b,…) array lands on disk as (…,b,a) — the shapes below
# are chosen so the DISK layout matches a5py's exactly.
_a5_i4(g, name, v::Integer) = (g[name] = fill(Int32(v), 1, 1))   # disk (1,1)
_a5_scalar(g, name, v::Real) = (g[name] = [Float64(v)])         # disk (1,)
_a5_scalar_i(g, name, v::Integer) = (g[name] = [Int32(v)])           # disk (1,)
_a5_col(g, name, v::AbstractVector) = (g[name] = reshape(Float64.(v), 1, :))  # disk (n,1)
_a5_col_i(g, name, v::AbstractVector) = (g[name] = reshape(Int32.(v), 1, :))    # disk (n,1)

"""
    write_ascot5(file, ts, out_path=""; nR=150, nZ=150, nphi=0, nbins=128,
                 fsa_method=:bin, fsa_window=4.0, desc="") -> out_path

Write one ASCOT5 input HDF5 for time slice `ts`: `bfield/B_3DS`
([`ascot5_bfield`](@ref)), `plasma/plasma_1D` (FSA profiles on ρ_pol, ions =
main D-like species + fully-stripped KPRAD impurity when active),
`wall/wall_2D` ([`mesh_boundary_rz`](@ref)) and a zero `efield/E_TC`.
Markers and options are the ASCOT operator's domain and are not written.

The `bfield` block is the exact FEM field sampled pointwise (no FSA — smooth by
construction). The `plasma_1D` background (ne/Te/Ti/ni), which drives ASCOT5's
collision operator, is flux-surface-averaged: `fsa_method` picks the estimator
(`:bin`, per-bin kernel average; `:cumulative`, the smoother `W′/V′` estimator
that removes bin-to-bin shot noise — a jagged n_e/T_e would inject unphysical
noise into ν∝n_e/T_e^{3/2}). `fsa_window` (default 4) tunes the `:cumulative`
regression width. See [`reduce_axisym_slice`](@ref); `scripts/export_run` passes
`:cumulative` by default.

`out_path=""` names the file `ascot_input_<slice>.h5` next to the C1.h5, using the
zero-padded slice INDEX (e.g. `ascot_input_024.h5` for slice 24), mirroring
M3D-C1's `time_%03d.h5` so each input lines up 1:1 with its source slice. (No
digit abuts the `ascot` prefix — the DB pipeline reads the first digits after
`ascot` as the MHD-time key.) The ntimestep is recorded in the group
`description`, not the filename. Off-mesh grid
points in the B field are `NaN` (fill strategy pending — docs/ascot5_writer.md);
a5py reads the file fine, but the field must be filled before an actual ASCOT run.
"""
function write_ascot5(
        file::M3DC1File, ts::Integer, out_path::AbstractString = "";
        nR::Integer = 150, nZ::Integer = 150, nphi::Integer = 0,
        margin::Real = 0.02, nbins::Integer = 128,
        fsa_method::Symbol = :bin, fsa_window::Real = 4.0,
        desc::AbstractString = ""
    )
    norm = normalization(file)
    bf = ascot5_bfield(file, ts; nR = nR, nZ = nZ, nphi = nphi, margin = margin, efield = true)
    isempty(out_path) && (out_path = _ascot5_default_path(file, ts))
    isempty(desc) && (desc = _ascot5_default_desc(file, ts, bf))

    # FSA background plasma on its own ρ_pol grid (standalone path). When called
    # from export_imas the per-slice FSA is reused instead — see there.
    ep = elems_plane(file)
    Rg_n = collect(range(extrema(ep[5, :])..., length = 120))
    Zg_n = collect(range(extrema(ep[6, :])..., length = 120))
    idm = build_grid_to_element_map(Rg_n, Zg_n, ep)
    sl = read_timeslice(file, ts; fields = (:psi,), rows = 1:20)
    fields = Dict{Symbol, Matrix{Float64}}(:psi => sl.fields[:psi])
    kz = _kprad_z(file)
    flds = kz >= 0 ? (_OPT_FIELDS..., :I, (Symbol("kprad_n_" * lpad(iz, 2, '0')) for iz in 0:kz)...) :
        (_OPT_FIELDS..., :I)
    for fld in flds
        v = _try_field(file, ts, fld)
        v === nothing || (fields[fld] = v)
    end
    lim = limiter_points(file)
    prof = reduce_axisym_slice(
        fields, ep, file.nplanes, norm,
        sl.psi_axis, sl.psi_lcfs, sl.xmag, sl.zmag,
        sl.time * unit_factor(norm, :time; system = :si),
        Rg_n, Zg_n, idm;
        nbins = nbins, fsa_method = fsa_method, fsa_window = fsa_window,
        kprad_z = kz,
        sl.xnull, sl.znull, sl.xnull2, sl.znull2,
        lim.xlim, lim.zlim, lim.xlim2, lim.zlim2
    )
    return _write_ascot5_hdf5(out_path, file, ep, norm, kz, bf, prof, desc)
end

# Default output path: name by the zero-padded slice INDEX, mirroring M3D-C1's
# `time_%03d.h5` so inputs pair 1:1 with their source slice (time_024.h5 ↔
# ascot_input_024.h5). No digit may abut the `ascot` prefix: the DB pipeline's
# step-from-filename parsers read the first digits after `ascot` as the MHD-time
# key, so `ascot5_…` would parse as step 5. ntimestep is kept in `desc`.
_ascot5_default_path(file::M3DC1File, ts::Integer) =
    joinpath(dirname(String(file.path)), "ascot_input_$(lpad(ts, 3, '0')).h5")

_ascot5_default_desc(file::M3DC1File, ts::Integer, bf) =
    "M3DC1Reader $(basename(String(file.path))) slice $ts (ntimestep=$(bf.nstep), t=$(bf.time) s)"

# Shared ASCOT5 HDF5 writer. Given a precomputed B-field `bf` ([`ascot5_bfield`])
# and FSA plasma profiles `prof` (a [`reduce_axisym_slice`] result — SI `.ne`,
# `.te`, `.ti`, `.ni`, `.imp_dens`, `.rho`), fill + floor the profiles, pick the
# ion species, and write the a5py-layout file. Both `write_ascot5` (standalone,
# 120² grid) and `export_imas` (reusing its per-slice `res` on the export grid)
# call this, so the `plasma_1D` block is byte-identical between the two pipelines.
function _write_ascot5_hdf5(
        out_path::AbstractString, file::M3DC1File, ep::AbstractMatrix,
        norm::M3DNormalization, kz::Integer, bf, prof, desc::AbstractString
    )
    nR = length(bf.Rg);  nZ = length(bf.Zg)

    # nearest-finite fill + positivity floor (plasma_1D must be finite > 0)
    function _fillprof(v, floor_)
        v === nothing && return nothing
        x = Float64.(v)
        lastf = NaN
        for i in eachindex(x)
            isfinite(x[i]) ? (lastf = x[i]) : (x[i] = lastf)
        end
        for i in reverse(eachindex(x))
            isfinite(x[i]) ? (lastf = x[i]) : (x[i] = lastf)
        end
        return max.(x, floor_)
    end
    ne = _fillprof(prof.ne, 1.0e6)          # m⁻³
    te = _fillprof(prof.te, 1.0)          # eV
    ti = _fillprof(prof.ti !== nothing ? prof.ti : prof.te, 1.0)
    ni = _fillprof(prof.ni !== nothing ? prof.ni : prof.ne, 1.0e6)
    (ne === nothing || te === nothing) &&
        error("write_ascot5: ne/te FSA profiles unavailable")

    # ion species: main D-like ion + total KPRAD impurity (fully-stripped
    # charge assumed — the collision operator needs one Z per species)
    imp = kz >= 0 && prof.imp_dens !== nothing ?
        _fillprof(
            reduce(
                .+, (p for p in prof.imp_dens[2:end] if p !== nothing);
                init = zeros(length(prof.rho))
            ), 0.0
        ) : nothing
    imp_label, imp_a = _impurity_species(max(kz, 0))
    anum = imp === nothing ? Int32[round(Int32, file.ion_mass)] :
        Int32[round(Int32, file.ion_mass), round(Int32, imp_a)]
    znum = imp === nothing ? Int32[1] : Int32[1, kz]
    charge = znum
    mass = imp === nothing ? Float64[file.ion_mass] : Float64[file.ion_mass, imp_a]

    isfile(out_path) && rm(out_path)
    h5open(String(out_path), "w") do f
        # ---- bfield/B_3DS ----
        g = _a5_group(f, "bfield", "B_3DS"; desc = desc)
        _a5_scalar(g, "b_rmin", bf.Rg[1]);    _a5_scalar(g, "b_rmax", bf.Rg[end])
        _a5_scalar_i(g, "b_nr", nR)
        _a5_scalar(g, "b_zmin", bf.Zg[1]);    _a5_scalar(g, "b_zmax", bf.Zg[end])
        _a5_scalar_i(g, "b_nz", nZ)
        _a5_scalar(g, "b_phimin", 0.0);       _a5_scalar(g, "b_phimax", 360.0)
        _a5_scalar_i(g, "b_nphi", length(bf.phi_deg))
        _a5_scalar(g, "psi_rmin", bf.Rg[1]);  _a5_scalar(g, "psi_rmax", bf.Rg[end])
        _a5_scalar_i(g, "psi_nr", nR)
        _a5_scalar(g, "psi_zmin", bf.Zg[1]);  _a5_scalar(g, "psi_zmax", bf.Zg[end])
        _a5_scalar_i(g, "psi_nz", nZ)
        _a5_scalar(g, "axisr", bf.axisr);     _a5_scalar(g, "axisz", bf.axisz)
        _a5_scalar(g, "psi0", bf.psi0);       _a5_scalar(g, "psi1", bf.psi1)
        g["psi"] = bf.psi                     # disk (nz,nr), a5py reads back (nr,nz)
        g["br"] = bf.br                      # disk (nz,nphi,nr)
        g["bphi"] = bf.bphi
        g["bz"] = bf.bz

        # ---- plasma/plasma_1D ----
        g = _a5_group(f, "plasma", "plasma_1D"; desc = desc)
        nrho = length(prof.rho);  nion = length(znum)
        _a5_i4(g, "nion", nion);  _a5_i4(g, "nrho", nrho)
        _a5_col_i(g, "znum", znum);  _a5_col_i(g, "anum", anum)
        _a5_col_i(g, "charge", charge)
        _a5_col(g, "mass", mass)
        _a5_col(g, "rho", prof.rho)
        _a5_col(g, "vtor", zeros(nrho))
        _a5_col(g, "etemperature", te);  _a5_col(g, "edensity", ne)
        _a5_col(g, "itemperature", ti)
        idens = imp === nothing ? reshape(ni, :, 1) : hcat(ni, imp)
        g["idensity"] = Array{Float64}(idens)  # Julia (nrho,nion) → disk (nion,nrho)

        # ---- wall/wall_2D ----
        ulen = unit_factor(norm, :length; system = :si)
        Rb, Zb = mesh_boundary_rz(ep)
        g = _a5_group(f, "wall", "wall_2D"; desc = desc)
        _a5_i4(g, "nelements", length(Rb))
        _a5_col(g, "r", Rb .* ulen);  _a5_col(g, "z", Zb .* ulen)
        _a5_col_i(g, "flag", zeros(Int32, length(Rb)))

        # ---- efield: E_3D (total MHD E from M3D-C1) when available, else a zero
        # E_TC (a5py's recommended "no electric field" input). E_3D scalar keys are
        # UNprefixed (unlike B_3DS's b_*); er/ephi/ez land disk (nz,nphi,nr) like br.
        if bf.er !== nothing
            g = _a5_group(f, "efield", "E_3D"; desc = desc)
            _a5_scalar(g, "rmin", bf.Rg[1]);    _a5_scalar(g, "rmax", bf.Rg[end])
            _a5_scalar_i(g, "nr", nR)
            _a5_scalar(g, "zmin", bf.Zg[1]);    _a5_scalar(g, "zmax", bf.Zg[end])
            _a5_scalar_i(g, "nz", nZ)
            _a5_scalar(g, "phimin", 0.0);       _a5_scalar(g, "phimax", 360.0)
            _a5_scalar_i(g, "nphi", length(bf.phi_deg))
            g["er"] = bf.er                     # disk (nz,nphi,nr)
            g["ephi"] = bf.ephi
            g["ez"] = bf.ez
        else
            g = _a5_group(f, "efield", "E_TC"; desc = desc)
            _a5_col(g, "exyz", zeros(3))
        end
    end
    return String(out_path)
end
