# IMASdd/IMAS are weakdeps only (not in test/Project.toml), so these items SKIP
# under the standard suite / cc-julia-test-runner. Real verification of to_imas is
# done via a direct `julia --project=.` run in an env where IMASdd/IMAS resolve
# (see docs/superpowers/... task reports).
@testitem "to_imas" begin
    if Base.find_package("IMASdd") === nothing || Base.find_package("IMAS") === nothing
        @info "skipping to_imas (IMAS/IMASdd unavailable)"
    else
        @eval using IMASdd, IMAS
        R = range(1.0, 2.0; length = 33); Z = range(-1.0, 1.0; length = 33)
        psi_rz = [ (r - 1.5)^2 + z^2 for r in R, z in Z ]
        psi1d = collect(range(0.0, 0.25; length = 16)); F1d = fill(3.4, 16)
        eq = M3DAxisymField(;
            cocos = 1, time = 0.0, R = R, Z = Z, psi_rz = psi_rz,
            psi1d = psi1d, F1d = F1d, p1d = nothing, axis = (1.5, 0.0),
            psi_axis = 0.0, psi_boundary = 0.25, r0 = 1.5, b0 = 2.27
        )
        dd = to_imas(eq; time = 0.0, trace = false)     # no trace: just check population
        eqt = dd.equilibrium.time_slice[1]
        @test eqt.profiles_2d[1].psi == psi_rz
        @test eqt.profiles_1d.f == F1d
        @test dd.equilibrium.vacuum_toroidal_field.r0 == 1.5
        # with trace=true, flux_surfaces fills q (gated; may degrade on non-nested input)
        dd2 = to_imas(eq; time = 0.0, trace = true)
        @test hasproperty(dd2.equilibrium.time_slice[1].profiles_1d, :q)
    end
end

# Graceful degradation: a non-nested ψ (monotonic in R, no closed flux surface)
# makes IMAS.flux_surfaces fail; to_imas's try/catch must warn and return a dd (q
# left unfilled/zeros) rather than throwing. Gated like the testset above; verified
# via a direct run: warns "flux_surfaces failed", no throw, q all-zeros.
@testitem "to_imas graceful degradation (non-nested ψ)" begin
    if Base.find_package("IMASdd") === nothing || Base.find_package("IMAS") === nothing
        @info "skipping to_imas degradation (IMAS/IMASdd unavailable)"
    else
        @eval using IMASdd, IMAS
        R = range(1.0, 2.0; length = 17); Z = range(-1.0, 1.0; length = 17)
        psi_rz = [ r + 0.0 * z for r in R, z in Z ]     # monotonic: no closed surface
        eq = M3DAxisymField(;
            cocos = 5, time = 0.0, R = R, Z = Z, psi_rz = psi_rz,
            psi1d = collect(range(1.0, 2.0; length = 8)), F1d = fill(3.0, 8),
            p1d = nothing, axis = (1.5, 0.0), psi_axis = 1.0, psi_boundary = 2.0,
            r0 = 1.5, b0 = 2.0
        )
        dd = @test_logs (:warn,) match_mode = :any to_imas(eq; trace = true)
        @test dd isa IMASdd.dd
        # tracer failure leaves q at its default (zeros), not filled
        @test all(iszero, dd.equilibrium.time_slice[1].profiles_1d.q)
    end
end

# Cross-validation of the cumulative-U q (FSA identity, reduce_axisym_slice)
# against IMAS.flux_surfaces' contour-integral q on real data. Doubly gated:
# needs IMASdd/IMAS resolvable AND a real C1.h5. The load-bearing part of the
# recipe is the node-window CLIP to ψ_N ∈ [0.02, 0.995]: the tracer throws on
# the axis-degenerate innermost surfaces and on the separatrix, and one failed
# surface aborts the whole flux_surfaces call. The node COUNT is free —
# verified working down to 32 nodes (an earlier "≥256 required" note was
# outdated; 340 is used here only for a fine-grained comparison, and accuracy
# of the comparison degrades below ~64 nodes). Expected relations:
# q_imas = 2π·q (COCOS-11 dd vs per-radian identity; median within 1%), and
# since the cumulative-U estimator BOTH q's sit at the same smoothness level.
@testitem "q cross-validation vs IMAS flux_surfaces (real C1.h5)" begin
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if Base.find_package("IMASdd") === nothing || Base.find_package("IMAS") === nothing
        @info "skipping q cross-validation (IMAS/IMASdd unavailable)"
    elseif !isfile(c1)
        @info "skipping q cross-validation (no C1.h5 at $c1)"
    else
        @eval using IMASdd, IMAS, Statistics
        file = M3DC1File(c1)
        ts = last(list_timeslices(file))
        eq = axisym_field(file, ts; nbins = 340)
        ψn = (eq.psi1d .- eq.psi_axis) ./ (eq.psi_boundary - eq.psi_axis)
        rng = findfirst(>=(0.02), ψn):findlast(<=(0.995), ψn)
        @test length(rng) >= 200     # fine comparison grid (the count itself is free)
        eqc = M3DAxisymField(;
            cocos = 1, time = eq.time, R = eq.R, Z = eq.Z,
            psi_rz = eq.psi_rz, psi1d = eq.psi1d[rng], F1d = eq.F1d[rng],
            p1d = eq.p1d === nothing ? nothing : eq.p1d[rng],
            q1d = eq.q1d === nothing ? nothing : eq.q1d[rng],
            axis = eq.axis, psi_axis = eq.psi_axis, psi_boundary = eq.psi_boundary,
            r0 = eq.r0, b0 = eq.b0, x_points = eq.x_points
        )
        dd = to_imas(eqc; trace = true)
        q_im = dd.equilibrium.time_slice[1].profiles_1d.q
        q_bin = eq.q1d[rng];  ψnc = ψn[rng]
        m = [
            i for i in eachindex(q_im)
                if isfinite(q_im[i]) && isfinite(q_bin[i]) && 0.05 <= ψnc[i] <= 0.95
        ]
        @test length(m) > 200
        # unbiasedness: the 2π relation holds in the median within 2%
        @test abs(median(q_im[m] ./ q_bin[m]) / 2π - 1) < 0.02
        # smoothness: since the cumulative-U estimator, BOTH q's are at the
        # contour-reference level (~1% node-to-node)
        noise(q) = std(diff(q)) / mean(abs.(q))
        @test noise(q_im[m]) < 0.02
        @test noise(q_bin[m]) < 0.02
        # pointwise agreement: two independent methods within 2% for 95% of nodes
        rel = abs.(q_im[m] ./ (2π .* q_bin[m]) .- 1)
        @test quantile(rel, 0.95) < 0.02
    end
end

# fsa_imas plumbing: an independent IMAS contour FSA of a 2D field map. A
# constant field must average to that constant on every traced surface (the
# ∮f·dl/∮dl identity), and the ρ nodes must run axis→boundary. Gated like the
# rest of this file (skips without IMASdd/IMAS).
@testitem "fsa_imas contour FSA (constant-field plumbing)" begin
    if Base.find_package("IMASdd") === nothing || Base.find_package("IMAS") === nothing
        @info "skipping fsa_imas plumbing (IMAS/IMASdd unavailable)"
    else
        @eval using IMASdd, IMAS
        R = range(1.0, 2.0; length = 65);  Z = range(-1.0, 1.0; length = 65)
        psi_rz = [(r - 1.5)^2 + z^2 for r in R, z in Z]   # nested circular surfaces
        psi1d = collect(range(0.0, 0.2; length = 48));  F1d = fill(3.4, 48)
        eq = M3DAxisymField(;
            cocos = 1, time = 0.0, R = R, Z = Z, psi_rz = psi_rz,
            psi1d = psi1d, F1d = F1d, p1d = nothing, axis = (1.5, 0.0),
            psi_axis = 0.0, psi_boundary = 0.2, r0 = 1.5, b0 = 2.27
        )
        fld = ones(length(R), length(Z))              # constant ⇒ ⟨f⟩ = 1 everywhere
        res = fsa_imas(eq, fld, R, Z)
        @test length(res.rho) == length(psi1d)
        @test res.rho[1] ≈ 0.0 atol = 1.0e-9
        @test res.rho[end] ≈ 1.0 atol = 1.0e-9
        fin = isfinite.(res.avg)
        @test count(fin) ≥ 20                          # most surfaces traced
        @test all(v -> isapprox(v, 1.0; atol = 1.0e-6), res.avg[fin])
        # a fully off-mesh (NaN) field cannot be sampled → all-NaN, no crash
        res_nan = fsa_imas(eq, fill(NaN, length(R), length(Z)), R, Z)
        @test all(isnan, res_nan.avg)
        # grid/shape mismatch is a clear error
        @test_throws ErrorException fsa_imas(eq, ones(3, 3), R, Z)
    end
end

# Cross-validation of the :cumulative estimator against the IMAS contour FSA on
# the circular analytic fixture: ψ=(ξ-ξ0)²+(η-η0)², constant F=I0 ⇒ analytic
# ⟨|B|⟩ = √(I0²+4·ψ1s·ρ²)/R0. Both independent methods must match the closed form
# AND each other. Deterministic (no data file) — runs whenever IMAS is present.
@testitem "fsa_imas vs :cumulative ⟨|B|⟩ (circular cross-val)" begin
    if Base.find_package("IMASdd") === nothing || Base.find_package("IMAS") === nothing
        @info "skipping fsa_imas cross-val (IMAS/IMASdd unavailable)"
    else
        @eval using IMASdd, IMAS
        norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
        ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
        ξ0, η0, R0 = 0.2, 0.3, 1.2
        psi = zeros(80, 1)
        psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
        psi[4, 1] = 1.0;         psi[6, 1] = 1.0
        I0 = 4.0;  Ic = zeros(80, 1);  Ic[1, 1] = I0
        fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :I => Ic)
        ψ1s = 0.04
        Rg = collect(range(0.95, 1.45, length = 100))
        Zg = collect(range(0.05, 0.55, length = 100))
        idm = build_grid_to_element_map(Rg, Zg, ep)

        res = reduce_axisym_slice(
            fields, ep, 1, norm, 0.0, ψ1s, R0, η0, 0.0, Rg, Zg, idm;
            nbins = 64, fsa_method = :cumulative
        )
        # rebuild the equilibrium + |B| map from the returned SI arrays (norm makes
        # the SI factors unity, so mesh numbers == SI numbers here)
        eq = M3DAxisymField(;
            cocos = 1, time = res.time, R = res.Rg, Z = res.Zg, psi_rz = res.psi_rz,
            psi1d = res.psi1d, F1d = res.F1d, p1d = nothing,
            axis = (res.R_axis, res.Z_axis), psi_axis = res.psi_axis,
            psi_boundary = res.psi_boundary, r0 = R0, b0 = I0 / R0
        )
        babs_rz = sqrt.(res.br_rz .^ 2 .+ res.bz_rz .^ 2 .+ res.bphi_rz .^ 2)
        im = fsa_imas(eq, babs_rz, res.Rg, res.Zg)

        ρ = res.rho
        b_an = [sqrt(I0^2 + 4 * ψ1s * r^2) / R0 for r in ρ]
        m = isfinite.(im.avg) .& isfinite.(res.babs1d) .& (ρ .> 0.15) .& (ρ .< 0.85)
        @test count(m) ≥ 10
        # both methods match the analytic ⟨|B|⟩ within a few %
        @test maximum(abs.((im.avg[m] .- b_an[m]) ./ b_an[m])) < 0.03
        @test maximum(abs.((res.babs1d[m] .- b_an[m]) ./ b_an[m])) < 0.03
        # and agree with each other
        @test maximum(abs.((im.avg[m] .- res.babs1d[m]) ./ res.babs1d[m])) < 0.03
    end
end
