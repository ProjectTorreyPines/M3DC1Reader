@testitem "reduce_axisym_slice (synthetic single element)" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    # single reference element a=b=c=1, θ=0, origin (0,0): global_to_local gives
    # ξ = R - 1, η = Z. Bowl ψ = (ξ-0.2)² + (η-0.3)² has its minimum at R=1.2, Z=0.3.
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    te = zeros(80, 1); te[1, 1] = 3.0          # constant Te = 3.0 (M3D units) in-element
    den = zeros(80, 1); den[1, 1] = 2.0      # constant ion density = 2.0 (M3D units)
    fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :te => te, :den => den)

    Rg = collect(range(1.05, 1.35, length = 6))
    Zg = collect(range(0.15, 0.45, length = 6))
    id_map = build_grid_to_element_map(Rg, Zg, ep)

    res = reduce_axisym_slice(
        fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.5, Rg, Zg, id_map;
        nbins = 16, adj = :linear
    )

    f_te = unit_factor(norm, :temperature; system = :si)
    f_den = unit_factor(norm, :density; system = :si)
    f_len = unit_factor(norm, :length; system = :si)
    @test res.rho == collect(range(0.0, 1.0, length = 16))
    @test length(res.te) == 16
    @test res.ne === nothing && res.ti === nothing && res.pressure === nothing
    @test any(isfinite, res.te)
    @test all(res.te[isfinite.(res.te)] .≈ 3.0 * f_te)   # FSA of a constant = the constant
    @test res.ni !== nothing
    @test all(res.ni[isfinite.(res.ni)] .≈ 2.0 * f_den)
    # outer ρ bins are legitimately empty (grid covers ρ≈0..0.95), but most bins fill:
    @test count(isfinite, res.te) ≥ 5
    @test size(res.psi_rz) == (6, 6)
    @test any(isfinite, res.psi_rz)
    @test res.time == 0.5
    @test res.R_axis ≈ 1.2 * f_len atol = 1.0e-3 * f_len
    @test res.Z_axis ≈ 0.3 * f_len atol = 1.0e-3 * f_len

    # no kprad_z passed (default -1) → impurity/Prad outputs are nothing
    @test res.imp_dens === nothing && res.prad1d === nothing

    # degenerate path: a present field that evaluates to all-NaN must return `nothing`
    nan_fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :ne => fill(NaN, 80, 1))
    res_deg = reduce_axisym_slice(nan_fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.5, Rg, Zg, id_map; nbins = 16)
    @test res_deg.ne === nothing
end

@testitem "reduce_axisym_slice kprad impurities + Prad1D" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    cf(c) = (a = zeros(80, 1); a[1, 1] = c; a)   # constant field
    fields = Dict{Symbol, Matrix{Float64}}(
        :psi => psi,
        :kprad_n_00 => cf(1.0), :kprad_n_01 => cf(2.0),       # neutral + Ne1+
        :kprad_rad => cf(0.5), :kprad_brem => cf(0.3)
    )       # reck/recp absent
    Rg = collect(range(1.05, 1.35, length = 6))
    Zg = collect(range(0.15, 0.45, length = 6))
    id_map = build_grid_to_element_map(Rg, Zg, ep)

    res = reduce_axisym_slice(
        fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.0, Rg, Zg, id_map;
        nbins = 16, kprad_z = 1
    )
    f_den = unit_factor(norm, :density; system = :si)
    f_pd = unit_factor(norm, :power_density; system = :si)

    @test res.imp_dens !== nothing
    @test length(res.imp_dens) == 2                                   # states 0 and 1
    @test all(res.imp_dens[1][isfinite.(res.imp_dens[1])] .≈ 1.0 * f_den)   # neutral (FSA of constant)
    @test all(res.imp_dens[2][isfinite.(res.imp_dens[2])] .≈ 2.0 * f_den)   # Ne1+
    @test res.prad1d !== nothing
    @test all(res.prad1d[isfinite.(res.prad1d)] .≈ (0.5 + 0.3) * f_pd)      # rad+brem summed
end

@testitem "reduce_axisym_slice D3 — axis-not-converged fallback" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    # Same synthetic bowl element (a=b=c=1, θ=0, origin (0,0)): ξ = R-1, η = Z;
    # the element triangle spans R∈[0,2], Z∈[0,1].
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    Rg = collect(range(1.05, 1.35, length = 6))
    Zg = collect(range(0.15, 0.45, length = 6))
    id_map = build_grid_to_element_map(Rg, Zg, ep)

    f_flux = unit_factor(norm, :magnetic_flux; system = :si)
    f_len = unit_factor(norm, :length; system = :si)

    # Seed the Newton axis search far OUTSIDE any element so locate_element
    # returns 0 on the first iteration → find_axis_newton(...).converged == false.
    # (Verified directly: elem_idx==0, converged==false for this seed.)
    psi_axis_plane1 = -0.123
    xmag = 7.7        # off-mesh seed → also the fallback magnetic-axis R
    zmag = -2.2       # off-mesh seed → also the fallback magnetic-axis Z

    ax = M3DC1Reader.find_axis_newton(
        average_toroidal_axisymmetric(psi, 1), ep,
        Float64(xmag), Float64(zmag)
    )
    @test ax.converged == false                                   # exercising the else-branch
    @test ax.elem_idx == 0                                        # walked off the mesh immediately

    res = reduce_axisym_slice(
        Dict{Symbol, Matrix{Float64}}(:psi => psi), ep, 1, norm,
        psi_axis_plane1, 0.05, xmag, zmag, 0.0, Rg, Zg, id_map; nbins = 16
    )
    # else-branch: scalar fallbacks flow straight through (× unit factor), exact equality
    @test res.psi_axis == psi_axis_plane1 * f_flux
    @test res.R_axis == xmag * f_len
    @test res.Z_axis == zmag * f_len

    # Contrast: the on-mesh bowl seed DOES converge to the analytic minimum (R=1.2, Z=0.3),
    # so reduce_axisym_slice uses the Newton result, NOT the fallback scalars.
    res_c = reduce_axisym_slice(
        Dict{Symbol, Matrix{Float64}}(:psi => psi), ep, 1, norm,
        psi_axis_plane1, 0.05, 1.2, 0.3, 0.0, Rg, Zg, id_map; nbins = 16
    )
    @test res_c.R_axis ≈ 1.2 * f_len atol = 1.0e-3 * f_len
    @test res_c.Z_axis ≈ 0.3 * f_len atol = 1.0e-3 * f_len
    @test res_c.psi_axis != psi_axis_plane1 * f_flux             # converged path overrides fallback
end

@testitem "reduce_axisym_slice D4 — KPRAD missing mid-range charge state" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    cf(c) = (a = zeros(80, 1); a[1, 1] = c; a)
    # kprad_z = 2 (states 0,1,2) but provide ONLY state 0 and state 2 (omit state 1):
    # the imp_dens comprehension must leave a `nothing` "hole" at the missing state.
    fields = Dict{Symbol, Matrix{Float64}}(
        :psi => psi,
        :kprad_n_00 => cf(1.0), :kprad_n_02 => cf(3.0)
    )
    Rg = collect(range(1.05, 1.35, length = 6))
    Zg = collect(range(0.15, 0.45, length = 6))
    id_map = build_grid_to_element_map(Rg, Zg, ep)

    res = reduce_axisym_slice(
        fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.0, Rg, Zg, id_map;
        nbins = 16, kprad_z = 2
    )
    f_den = unit_factor(norm, :density; system = :si)

    @test res.imp_dens !== nothing
    @test length(res.imp_dens) == 3                              # states 0, 1, 2
    @test res.imp_dens[2] === nothing                            # the missing state-1 "hole"
    @test res.imp_dens[1] !== nothing && res.imp_dens[3] !== nothing
    @test all(res.imp_dens[1][isfinite.(res.imp_dens[1])] .≈ 1.0 * f_den)   # neutral (state 0)
    @test all(res.imp_dens[3][isfinite.(res.imp_dens[3])] .≈ 3.0 * f_den)   # state 2
    @test any(isfinite, res.imp_dens[1]) && any(isfinite, res.imp_dens[3])  # finite where mapped
end

@testitem "reduce_axisym_slice D8 — single finite grid point reduces to nothing" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    cf(c) = (a = zeros(80, 1); a[1, 1] = c; a)
    # Element triangle spans R∈[0,2], Z∈[0,1]. Build a grid whose ONLY interior
    # point is (1.2, 0.3) so the reduction sees exactly one finite sample
    # (count(m) == 1 < 2 → reduce_si returns nothing).
    Rg = collect(range(1.2, 50.0, length = 6))   # only first node R=1.2 is inside
    Zg = collect(range(0.3, 50.0, length = 6))   # only first node Z=0.3 is inside
    id_map = build_grid_to_element_map(Rg, Zg, ep)
    @test count(!=(0), id_map) == 1               # exactly one interior grid point

    res = reduce_axisym_slice(
        Dict{Symbol, Matrix{Float64}}(:psi => psi, :te => cf(3.0)),
        ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.0, Rg, Zg, id_map; nbins = 16
    )
    @test res.te === nothing                      # < 2 finite points ⇒ profile is nothing
end

@testitem "reduce_axisym_slice recomputes boundary flux via find_lcfs" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    f_flux = unit_factor(norm, :magnetic_flux; system = :si)
    # bowl+saddle element (see test_find_critical_points.jl):
    # ψ = ξ² + (η−½)² − (η−½)³ on a=b=c=2, θ=0, origin (0,−1)
    #   → O-point at global (2, −1/2), ψ0 = 0; saddle at (2, +1/6), ψx = 4/27
    ep = reshape([2.0, 2.0, 2.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    psi = zeros(80, 1)
    psi[1, 1] = 0.375; psi[3, 1] = -1.75; psi[4, 1] = 1.0
    psi[6, 1] = 2.5;   psi[10, 1] = -1.0
    Rg = collect(range(1.7, 2.3, length = 8))
    Zg = collect(range(-0.9, 0.05, length = 8))
    id_map = build_grid_to_element_map(Rg, Zg, ep)
    stored_lcfs = 0.5                       # deliberately wrong plane-1 value

    # xnull seed given → ψ1 = recomputed saddle flux (4/27), NOT the stored 0.5;
    # ψ0 = recomputed O-point (0, not the stored 0.01). Deviation @warn expected.
    res = reduce_axisym_slice(
        Dict{Symbol, Matrix{Float64}}(:psi => psi), ep, 1, norm,
        0.01, stored_lcfs, 2.05, -0.45, 0.0, Rg, Zg, id_map;
        nbins = 8, xnull = 2.0, znull = 0.1
    )
    @test res.psi_boundary ≈ (4 / 27) * f_flux atol = 1.0e-8 * f_flux
    @test res.psi_axis ≈ 0.0 atol = 1.0e-8 * f_flux
    # psi1d endpoints follow the recomputed pair
    @test res.psi1d[1] ≈ res.psi_axis atol = 1.0e-12 * f_flux
    @test res.psi1d[end] ≈ res.psi_boundary atol = 1.0e-12 * f_flux

    # no boundary candidate at all → stored psi_lcfs flows through (axis still recomputed)
    res0 = reduce_axisym_slice(
        Dict{Symbol, Matrix{Float64}}(:psi => psi), ep, 1, norm,
        0.01, stored_lcfs, 2.05, -0.45, 0.0, Rg, Zg, id_map; nbins = 8
    )
    @test res0.psi_boundary == stored_lcfs * f_flux
    @test res0.psi_axis ≈ 0.0 atol = 1.0e-8 * f_flux
end

@testitem "reduce_axisym_slice FSA weighting + axis augmentation options" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    te = zeros(80, 1); te[2, 1] = 1.0            # te(ξ,η) = ξ = R − 1 (varies with R)
    fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :te => te)
    Rg = collect(range(1.05, 1.35, length = 6))
    Zg = collect(range(0.15, 0.45, length = 6))
    id_map = build_grid_to_element_map(Rg, Zg, ep)
    args = (fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.0, Rg, Zg, id_map)

    res_w = reduce_axisym_slice(args...; nbins = 8)                                  # defaults on
    res_u = reduce_axisym_slice(args...; nbins = 8, weighted = false, axis_aug = false)
    @test any(isfinite, res_w.te) && any(isfinite, res_u.te)
    # w ∝ R correlates positively with te = R−1 within a ρ ring, so the weighted
    # FSA exceeds the unweighted bin mean in at least one bin
    both = isfinite.(res_w.te) .& isfinite.(res_u.te)
    @test any(res_w.te[both] .> res_u.te[both] .+ 1.0e-12)

    # axis augmentation fills at least as many (typically more inner) ρ bins
    res_aug = reduce_axisym_slice(args...; nbins = 8, axis_aug = true)
    res_noaug = reduce_axisym_slice(args...; nbins = 8, axis_aug = false)
    @test count(isfinite, res_aug.te) ≥ count(isfinite, res_noaug.te)

    # gauss kernel runs and stays within the sample value range (convex average)
    res_g = reduce_axisym_slice(args...; nbins = 8, adj = :gauss)
    gv = filter(isfinite, res_g.te)
    f_te = unit_factor(norm, :temperature; system = :si)
    @test !isempty(gv)
    @test all(v -> 0.0 * f_te ≤ v ≤ 0.36 * f_te, gv)   # te = ξ ∈ [0.05, 0.35] on the grid
end

@testitem "reduce_axisym_slice q1d — circular analytic check" begin
    # bowl ψ = (ξ−ξ0)² + (η−η0)² ⇒ exactly circular surfaces of radius a=√ψ
    # around the axis (R₀=1.2, Z₀=0.3), and constant F=I0. Analytic safety factor
    #   q(a) = F / (2√(R₀²−a²))      (∮dθ/(R₀+a·cosθ) = 2π/√(R₀²−a²))
    # (norm b0=1e4, l0=100 ⇒ ulen=uB=uflux=1, so SI == normalized numbers).
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0, R0 = 0.2, 0.3, 1.2
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    I0 = 4.0
    Ic = zeros(80, 1); Ic[1, 1] = I0
    fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :I => Ic)
    ψ1s = 0.04                                     # a_max = 0.2: circles fully in-element
    Rg = collect(range(0.95, 1.45, length = 80))   # V' is a shell-volume DENSITY —
    Zg = collect(range(0.05, 0.55, length = 80))   # needs enough samples per bin
    idm = build_grid_to_element_map(Rg, Zg, ep)

    res = reduce_axisym_slice(
        fields, ep, 1, norm, 0.0, ψ1s, R0, η0, 0.0,
        Rg, Zg, idm; nbins = 16
    )
    @test res.F1d !== nothing && res.q1d !== nothing && res.phi1d !== nothing
    # F is the constant I0 wherever finite
    @test all(x -> isapprox(x, I0; rtol = 1.0e-6), filter(isfinite, res.F1d))
    # separatrix node NaN; the axis value comes from the integral V-fit and is
    # exact for this circular field: q0 = F/(2R0)
    @test isnan(res.q1d[end])
    @test res.q1d[1] ≈ I0 / (2R0) rtol = 0.02
    # interior bins match the analytic q within 5%
    ρ = res.rho
    q_an = [I0 / (2 * sqrt(R0^2 - ψ1s * r^2)) for r in ρ]
    m = isfinite.(res.q1d) .& (ρ .> 0.15) .& (ρ .< 0.95)
    @test count(m) ≥ 10
    @test maximum(abs.((res.q1d[m] .- q_an[m]) ./ q_an[m])) < 0.02
    # toroidal flux is cumulative from the axis and increasing (q > 0 here)
    @test res.phi1d[1] == 0.0
    @test issorted(res.phi1d)
    # no X-point in this bowl → no recomputed x_points, no confined-mask effect
    @test isempty(res.x_points)

    # ---- axisymmetric B assembly on the same fixture (exact for a quadratic ψ):
    # ψ(R,Z) = (R−R₀)² + (Z−Z₀)² ⇒ B_R = −2(Z−Z₀)/R, B_Z = 2(R−R₀)/R, Bφ = I0/R
    @test res.br_rz !== nothing && res.bz_rz !== nothing && res.bphi_rz !== nothing
    br_an = [-2 * (Zg[j] - η0) / Rg[i] for i in eachindex(Rg), j in eachindex(Zg)]
    bz_an = [ 2 * (Rg[i] - R0) / Rg[i] for i in eachindex(Rg), j in eachindex(Zg)]
    bphi_an = [ I0 / Rg[i]               for i in eachindex(Rg), j in eachindex(Zg)]
    fin = isfinite.(res.br_rz)
    @test count(fin) > 1000                      # most of the grid is in-element
    @test maximum(abs.(res.br_rz[fin] .- br_an[fin])) < 1.0e-9
    @test maximum(abs.(res.bz_rz[fin] .- bz_an[fin])) < 1.0e-9
    @test maximum(abs.(res.bphi_rz[fin] .- bphi_an[fin])) < 1.0e-9
    # ⟨|B|⟩: on the circle a(ρ)=√ψ1s·ρ, |B|·R = √(I0²+4a²) is constant, so the
    # volume-weighted FSA is exactly √(I0²+4a²)/⟨R⟩ring = √(I0²+4a²)/R₀
    @test res.babs1d !== nothing
    b_an = [sqrt(I0^2 + 4 * ψ1s * r^2) / R0 for r in ρ]
    mB = isfinite.(res.babs1d) .& (ρ .> 0.1) .& (ρ .< 0.95)
    @test count(mB) ≥ 10
    @test maximum(abs.((res.babs1d[mB] .- b_an[mB]) ./ b_an[mB])) < 0.02
    # Φ(ψ(R,Z)) map: finite & ≥0 inside the LCFS, NaN outside (open surfaces)
    @test res.phi_rz !== nothing
    inside = isfinite.(res.psi_rz) .& (res.psi_rz .<= ψ1s) .& (res.psi_rz .>= 0.0)
    @test all(x -> isfinite(x) && x >= 0, res.phi_rz[inside])
    outside = isfinite.(res.psi_rz) .& (res.psi_rz .> 1.0001 * ψ1s)
    @test count(outside) > 0 && all(isnan, res.phi_rz[outside])

    # unweighted mode disables q (V' needs the volume measure) but keeps F —
    # and keeps the B maps + ⟨|B|⟩ (plain bin mean), minus the Φ map (needs q)
    res_u = reduce_axisym_slice(
        fields, ep, 1, norm, 0.0, ψ1s, R0, η0, 0.0,
        Rg, Zg, idm; nbins = 16, weighted = false
    )
    @test res_u.q1d === nothing
    @test res_u.F1d !== nothing
    @test res_u.bphi_rz !== nothing
    @test res_u.babs1d !== nothing
    @test res_u.phi_rz === nothing
end

@testitem "reduce_axisym_slice fsa_method=:cumulative (analytic correctness)" begin
    # Same circular fixture as the q1d test: bowl ψ=(ξ-ξ0)²+(η-η0)² ⇒ circular
    # surfaces, constant F=I0 and a constant Te field. The cumulative W′/V′
    # estimator must (a) reproduce a constant exactly and (b) match the analytic
    # ⟨|B|⟩ = √(I0²+4·ψ1s·ρ²)/R0. (The smoothness advantage over :bin is a
    # shot-noise-regime property of sparse/irregular REAL data — on this dense,
    # well-sampled synthetic grid :bin is already smooth — so it is asserted in
    # the gated real-data cross-validation against :imas, not here.)
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0, R0 = 0.2, 0.3, 1.2
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    I0 = 4.0
    Ic = zeros(80, 1); Ic[1, 1] = I0
    te = zeros(80, 1); te[1, 1] = 3.0                # constant Te (M3D units)
    fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :I => Ic, :te => te)
    ψ1s = 0.04
    Rg = collect(range(0.95, 1.45, length = 100))
    Zg = collect(range(0.05, 0.55, length = 100))
    idm = build_grid_to_element_map(Rg, Zg, ep)
    args = (fields, ep, 1, norm, 0.0, ψ1s, R0, η0, 0.0, Rg, Zg, idm)

    res_bin = reduce_axisym_slice(args...; nbins = 64, fsa_method = :bin)
    res_cum = reduce_axisym_slice(args...; nbins = 64, fsa_method = :cumulative)

    # (a) constant field ⇒ cumulative FSA is the constant wherever defined
    f_te = unit_factor(norm, :temperature; system = :si)
    cvals = filter(isfinite, res_cum.te)
    @test !isempty(cvals)
    @test all(v -> isapprox(v, 3.0 * f_te; rtol = 2.0e-3), cvals)

    # (b) ⟨|B|⟩ matches the analytic circular profile (interior nodes)
    ρ = res_cum.rho
    b_an = [sqrt(I0^2 + 4 * ψ1s * r^2) / R0 for r in ρ]
    mB = isfinite.(res_cum.babs1d) .& (ρ .> 0.1) .& (ρ .< 0.9)
    @test count(mB) ≥ 20
    @test maximum(abs.((res_cum.babs1d[mB] .- b_an[mB]) ./ b_an[mB])) < 0.02

    # :bin path is untouched (still returns a profile on the same fixture)
    @test res_bin.babs1d !== nothing
    @test count(isfinite, res_bin.babs1d) ≥ 20

    # invalid method is a clear error
    @test_throws ArgumentError reduce_axisym_slice(args...; nbins = 64, fsa_method = :bogus)
end
