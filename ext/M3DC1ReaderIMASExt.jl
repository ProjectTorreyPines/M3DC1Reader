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

end # module
