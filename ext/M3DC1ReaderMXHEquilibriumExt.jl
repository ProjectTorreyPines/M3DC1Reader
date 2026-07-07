module M3DC1ReaderMXHEquilibriumExt

using M3DC1Reader
using MXHEquilibrium

"""
    to_mxh(eq::M3DAxisymField; q, V, sigma=1) -> EFITEquilibrium

Adapt a `M3DAxisymField` into a `MXHEquilibrium.EFITEquilibrium`, which provides
`safety_factor` (q via FSA), `Bfield` (|B| → δB/B), and `flux_surface`/`average`
(FSA) for free.

`MXHEquilibrium.efit` requires `r`, `z`, and `psi` to be uniform `AbstractRange`s
of the same type. `eq.R/eq.Z` are vectors (wrapped as ranges) and `eq.psi1d` is
non-uniform (quadratic in ρ), so a uniform ψ range is built and the F(ψ)/p(ψ)
profiles are linearly resampled onto it before constructing the equilibrium.

Two data-conditioning steps make MXHEquilibrium's flux-surface tracer work on
M3D-C1 fields:
 1. Off-mesh grid points from `interpolate_axisym_to_grid` are `NaN`; a global
    cubic spline would propagate them to every evaluation, so they are filled
    (average of finite 4-neighbours, multi-pass dilation). The in-mesh
    core/axis is untouched.
 2. ψ is shifted so the boundary is 0 and the axis is the largest-|ψ| point
    (the EFIT/GEQDSK convention `flux_surface` assumes: `abs(ψ) >= abs(ψ_axis)`
    selects the axis). M3D-C1 ψ instead has ψ_axis≈0 growing outward, which
    makes the tracer return a degenerate single-point surface for ~90% of ψ.
    The shift is a pure additive offset: ∇ψ (hence B) is unchanged, and F(ψ)/
    p(ψ) mappings are preserved because the ψ grid is shifted identically. This
    was validated empirically on real SPI data — q rises 1.0→3.3 outward only
    after the shift (flat q≈1 on the innermost 10% of surfaces without it).
"""
function M3DC1Reader.to_mxh(
        eq::M3DC1Reader.M3DAxisymField;
        q = zeros(length(eq.psi1d)),
        V = zeros(length(eq.psi1d)), sigma::Int = 1
    )
    M3DC1Reader._assert_native(eq)
    cc = MXHEquilibrium.cocos(eq.cocos)
    R = range(first(eq.R), last(eq.R); length = length(eq.R))
    Z = range(first(eq.Z), last(eq.Z); length = length(eq.Z))
    # Shift ψ so the boundary is at 0 and the axis is the max-|ψ| extremum.
    shift = eq.psi_boundary
    psi1d = eq.psi1d .- shift
    # efit needs psi as a uniform AbstractRange; eq.psi1d is non-uniform → resample.
    psi_u = range(first(psi1d), last(psi1d); length = length(psi1d))
    F = M3DC1Reader._resample(psi1d, eq.F1d, psi_u)
    p = eq.p1d === nothing ? zeros(length(psi_u)) : M3DC1Reader._resample(psi1d, eq.p1d, psi_u)
    # Fill off-mesh NaNs (see docstring) and apply the same additive ψ shift.
    psi_rz = M3DC1Reader._fill_nan_dilate(eq.psi_rz) .- shift
    return MXHEquilibrium.efit(
        cc, R, Z, psi_u, psi_rz, F, p,
        collect(Float64, q), collect(Float64, V), eq.axis, sigma
    )
end

end # module
