# Analytic validation of deltab_over_b_rz (docs/deltab_over_b.md §5): a
# plane-stacked mini-mesh with one reference element per plane and
# hand-picked coefficient layers whose δB/B has a closed form.
#
# Geometry: a=b=c=1, θ=0, origin (x,z)=(2,0) → local ξ = R−3, η = Z; the
# triangle spans R∈[2,4], Z∈[0,1]. Poloidal term 3 is ξ⁰η¹ (MI/NI tables),
# so coef[3]=α gives ψ = α·η ⇒ ∂ψ/∂Z = α exactly. Term 1 is the constant.
#
# With ψ_p = (α₀ + ε cos φ_p)·η and F_p = F₀ + δ cos φ_p over nplanes=4
# equally spaced planes (mean cos = 0, mean cos² = ½ exactly):
#   B_R = −(α₀ + ε cos φ_p)/R,  Bφ = (F₀ + δ cos φ_p)/R,  B_Z = 0
#   δB  = √(2·(ε²/2 + δ²/2))/R = √(ε²+δ²)/R,  |B̄| = √(α₀²+F₀²)/R
#   δB/B = √(ε²+δ²)/√(α₀²+F₀²)      — R-independent.
# Adding f′_p = γ sin φ_p · η contributes B_Z = −γ sin φ_p (no 1/R):
#   δB/B = √(ε² + δ² + γ²R²)/√(α₀²+F₀²).

@testitem "deltab_over_b_rz: analytic n=1 fluctuation (ψ, F)" setup = [DeltaBFixture] begin
    nplanes = 4
    α0, ε, F0, δ = 0.8, 0.05, 2.4, 0.03
    psi3 = _stacked_coefs(nplanes, (c, φ) -> c[3] = α0 + ε * cos(φ))
    I3 = _stacked_coefs(nplanes, (c, φ) -> c[1] = F0 + δ * cos(φ))
    Rg = [2.8, 3.1];  Zg = [0.05, 0.15]

    r = deltab_over_b_rz(psi3, I3, nothing, nplanes, _EP1, Rg, Zg)
    expected = sqrt(ε^2 + δ^2) / sqrt(α0^2 + F0^2)
    for i in 1:2, j in 1:2
        @test r.db[i, j] ≈ expected  rtol = 1.0e-12
        @test r.br0[i, j] ≈ -α0 / Rg[i]  rtol = 1.0e-12   # n=0 = axisym assembler
        @test r.bphi0[i, j] ≈ F0 / Rg[i]  rtol = 1.0e-12
        @test abs(r.bz0[i, j]) < 1.0e-15
    end

    # single plane (axisymmetric run): fluctuation identically zero
    r1 = deltab_over_b_rz(psi3[:, 1:1], I3[:, 1:1], nothing, 1, _EP1, Rg, Zg)
    @test all(iszero, filter(isfinite, r1.db))

    # coefficient-table size mismatch is an informative error
    @test_throws ErrorException deltab_over_b_rz(
        psi3, I3[:, 1:2], nothing,
        nplanes, _EP1, Rg, Zg
    )
    @test_throws ErrorException deltab_over_b_rz(
        psi3[:, 1:3], I3[:, 1:3], nothing,
        2, _EP1, Rg, Zg
    )
end

@testitem "deltab_over_b_rz: f′ contribution, fp vs legacy ζ-layer" setup = [DeltaBFixture] begin
    nplanes = 4
    α0, ε, F0, δ, γ = 0.8, 0.05, 2.4, 0.03, 0.02
    psi3 = _stacked_coefs(nplanes, (c, φ) -> c[3] = α0 + ε * cos(φ))
    I3 = _stacked_coefs(nplanes, (c, φ) -> c[1] = F0 + δ * cos(φ))
    Rg = [2.8, 3.1];  Zg = [0.05, 0.15]

    # modern layout: f′ handed directly (the `fp` dataset), fprime=true
    fp3 = _stacked_coefs(nplanes, (c, φ) -> c[3] = γ * sin(φ))
    r = deltab_over_b_rz(psi3, I3, fp3, nplanes, _EP1, Rg, Zg; fprime = true)
    for i in 1:2, j in 1:2
        expected = sqrt(ε^2 + δ^2 + γ^2 * Rg[i]^2) / sqrt(α0^2 + F0^2)
        @test r.db[i, j] ≈ expected  rtol = 1.0e-12
        @test abs(r.bz0[i, j]) < 1.0e-15          # mean of sin over planes = 0
    end

    # legacy layout: same f′ encoded as the second ζ-layer of f (row 20+3),
    # fprime=false must reproduce the modern result exactly
    f3 = _stacked_coefs(nplanes, (c, φ) -> c[23] = γ * sin(φ))
    rl = deltab_over_b_rz(psi3, I3, f3, nplanes, _EP1, Rg, Zg; fprime = false)
    @test rl.db ≈ r.db  rtol = 1.0e-14
end
