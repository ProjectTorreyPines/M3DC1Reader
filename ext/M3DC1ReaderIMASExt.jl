module M3DC1ReaderIMASExt

using M3DC1Reader
using IMASdd, IMAS

"""
    to_imas(eq::M3DAxisymField; time=0.0, trace=true, wall_r=Float64[], wall_z=Float64[]) -> IMASdd.dd

Adapt a `M3DAxisymField` into an in-memory IMAS `dd`, populating
`dd.equilibrium.time_slice[1]` with the 2D ψ(R,Z) map (rectangular grid), the
1D ψ / F(ψ) / pressure profiles, and `vacuum_toroidal_field` (r0/b0). All SI.
Off-mesh `NaN`s in `eq.psi_rz` are filled (`_fill_nan_dilate`) so the tracer can
close the outer surfaces; `pressure` is set (zeros when `eq.p1d===nothing`)
because `flux_surfaces` computes `j_tor`/`dpressure_dpsi` unconditionally even
though q itself is pressure-independent.

When `trace=true`, `IMAS.flux_surfaces(eqt, wall_r, wall_z)` recomputes the axis
and traces the flux surfaces to fill q + geometric coefficients on
`profiles_1d`. `global_quantities.free_boundary` is set to 0 (fixed boundary):
we supply the 2D ψ map directly rather than a free-boundary coil solve, and
`flux_surfaces` requires the flag to be present. Unlike the EFIT adapter
(`to_mxh`), IMAS handles arbitrary ψ (recomputing axis/boundary itself), so no
EFIT-style ψ-shift is applied here. The `flux_surfaces` call is wrapped in
try/catch so a tracer failure on degenerate input (e.g. non-nested surfaces)
leaves q empty rather than erroring.

!!! note "COCOS convention"
    The dd is populated with M3D-C1's native SI quantities (per-radian ψ,
    COCOS≈5) and is NOT transformed to the IMAS-standard COCOS 11. IMAS's
    derived q therefore carries the native convention and differs from the
    MXHEquilibrium (`to_mxh`) q by a COCOS sign/2π factor. Reconciling the two
    conventions is the subject of the q cross-validation; emitting a
    spec-correct COCOS-11 dd is a known refinement (see the design spec).
"""
function M3DC1Reader.to_imas(
        eq::M3DC1Reader.M3DAxisymField; time::Real = 0.0,
        trace::Bool = true, wall_r = Float64[], wall_z = Float64[]
    )
    M3DC1Reader._assert_native(eq)
    dd = IMASdd.dd()
    dd.global_time = Float64(time)
    eqt = resize!(dd.equilibrium.time_slice, Float64(time))   # stamps eqt.time
    resize!(eqt.profiles_2d, 1)
    eqt.profiles_2d[1].grid.dim1 = collect(Float64, eq.R)
    eqt.profiles_2d[1].grid.dim2 = collect(Float64, eq.Z)
    eqt.profiles_2d[1].grid_type.index = 1            # 1 => rectangular (R,Z ala eqdsk)
    eqt.profiles_2d[1].grid_type.name = "rectangular"
    # Fill off-mesh NaNs so flux_surfaces can close the outer surfaces (its ψ
    # interpolant/gradient would otherwise be NaN beyond the plasma mesh).
    eqt.profiles_2d[1].psi = M3DC1Reader._fill_nan_dilate(eq.psi_rz)
    eqt.profiles_1d.psi = collect(Float64, eq.psi1d)
    eqt.profiles_1d.f = collect(Float64, eq.F1d)
    # flux_surfaces computes j_tor (→ dpressure_dpsi) unconditionally, so pressure
    # must be present even though q is pressure-independent; zeros when unavailable.
    eqt.profiles_1d.pressure = eq.p1d === nothing ? zeros(length(eq.psi1d)) :
        collect(Float64, eq.p1d)
    isfinite(eq.r0) && (dd.equilibrium.vacuum_toroidal_field.r0 = eq.r0)
    isfinite(eq.b0) && (dd.equilibrium.vacuum_toroidal_field.b0 = [eq.b0])
    dd.equilibrium.time = [Float64(time)]
    if trace
        # fixed-boundary: flux_surfaces requires this flag; we supply the 2D ψ
        # map directly (no free-boundary coil solve).
        eqt.global_quantities.free_boundary = 0
        try
            IMAS.flux_surfaces(eqt, collect(Float64, wall_r), collect(Float64, wall_z))
        catch err
            @warn "to_imas: flux_surfaces failed (non-nested surfaces?); q left empty" exception = err
        end
    end
    return dd
end

# Bilinear sample of a field on a regular (R,Z) grid; NaN off-grid or when any
# corner is NaN (off-mesh). R and Z are the grid nodes (monotone increasing).
function _bilinear(
        R::AbstractVector{Float64}, Z::AbstractVector{Float64},
        F::AbstractMatrix, r::Real, z::Real
    )
    (r < R[1] || r > R[end] || z < Z[1] || z > Z[end]) && return NaN
    i = clamp(searchsortedlast(R, r), 1, length(R) - 1)
    j = clamp(searchsortedlast(Z, z), 1, length(Z) - 1)
    hr = R[i + 1] - R[i];  hz = Z[j + 1] - Z[j]
    tr = hr > 0 ? (r - R[i]) / hr : 0.0
    tz = hz > 0 ? (z - Z[j]) / hz : 0.0
    f00 = F[i, j];  f10 = F[i + 1, j];  f01 = F[i, j + 1];  f11 = F[i + 1, j + 1]
    (isfinite(f00) && isfinite(f10) && isfinite(f01) && isfinite(f11)) || return NaN
    return (1 - tr) * (1 - tz) * f00 + tr * (1 - tz) * f10 +
        (1 - tr) * tz * f01 + tr * tz * f11
end

function M3DC1Reader.fsa_imas(
        eq::M3DC1Reader.M3DAxisymField, field_rz::AbstractMatrix,
        R::AbstractVector, Z::AbstractVector;
        wall_r = Float64[], wall_z = Float64[], clip = (0.02, 0.995)
    )
    M3DC1Reader._assert_native(eq)
    size(field_rz) == (length(R), length(Z)) ||
        error("fsa_imas: field_rz must be (length(R), length(Z)) matching eq's grid")

    # ρ_pol = √ψ_N for every node (the full returned grid). The IMAS tracer aborts
    # the WHOLE call if ANY surface fails, and the axis-degenerate innermost
    # surfaces (ψ_N→0) and the separatrix (ψ_N→1, open at the X-point) are exactly
    # those that fail on a real diverted equilibrium — so only the interior band
    # `clip` is traced; clipped nodes return NaN.
    ψ0 = Float64(eq.psi_axis);  ψ1 = Float64(eq.psi_boundary)
    psin = [(Float64(p) - ψ0) / (ψ1 - ψ0) for p in eq.psi1d]
    ρ = [sqrt(clamp(x, 0.0, 1.0)) for x in psin]
    keep = [i for i in eachindex(psin) if clip[1] <= psin[i] <= clip[2]]

    # Build the equilibrium dd (2D ψ map), seed the magnetic axis from eq.axis so
    # trace_surfaces need not re-find it, and restrict the 1D ψ/f to the traceable
    # interior band before tracing.
    dd = M3DC1Reader.to_imas(eq; trace = false, wall_r = wall_r, wall_z = wall_z)
    eqt = dd.equilibrium.time_slice[1]
    eqt.global_quantities.free_boundary = 0
    eqt.global_quantities.magnetic_axis.r = Float64(eq.axis[1])
    eqt.global_quantities.magnetic_axis.z = Float64(eq.axis[2])
    eqt.profiles_1d.psi = Float64.(eq.psi1d[keep])
    eqt.profiles_1d.f = Float64.(eq.F1d[keep])
    eqt.profiles_1d.pressure = zeros(length(keep))
    surfaces = IMAS.trace_surfaces(eqt, collect(Float64, wall_r), collect(Float64, wall_z))

    Rv = collect(Float64, R);  Zv = collect(Float64, Z)
    avg = fill(NaN, length(psin))
    for (kk, i) in enumerate(keep)
        kk <= length(surfaces) || break
        s = surfaces[kk]
        isempty(s.r) && continue
        vals = Vector{Float64}(undef, length(s.r))
        ok = true
        @inbounds for m in eachindex(s.r)
            v = _bilinear(Rv, Zv, field_rz, s.r[m], s.z[m])
            isfinite(v) || (ok = false; break)
            vals[m] = v
        end
        ok && (avg[i] = IMAS.flux_surface_avg(vals, s))
    end
    return (; rho = ρ, avg = avg)
end

end # module
