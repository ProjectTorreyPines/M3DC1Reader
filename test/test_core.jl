# Core FEM / axis / reduction math on synthetic single-element meshes + analytic
# ψ — no C1.h5 needed. The reference triangle a=b=c=1, θ=0, origin (R0,Z0) with
# ψ = (ξ-ξ0)² + (η-η0)² has a unique minimum at (ξ0,η0), exercising the FEM math
# without a data file.

@testitem "element geometry round-trip" begin
    R0, Z0, b, θ = 2.0, 0.3, 0.4, 0.7
    for (R, Z) in ((2.1, 0.35), (1.95, 0.28), (2.0, 0.3))
        ξ, η = global_to_local(R, Z, R0, Z0, b, θ)
        # invert: R = R0 + (ξ+b)cosθ - η sinθ ; Z = Z0 + (ξ+b)sinθ + η cosθ
        co, sn = cos(θ), sin(θ)
        Rb = R0 + (ξ + b) * co - η * sn
        Zb = Z0 + (ξ + b) * sn + η * co
        @test Rb ≈ R atol = 1.0e-12
        @test Zb ≈ Z atol = 1.0e-12
    end
end

@testitem "quintic polynomial + analytic derivatives" begin
    # ψ = c1 + c2 ξ + c3 η + c4 ξ² + c5 ξη + c6 η²  (terms k=1..6)
    coef = zeros(20)
    coef[1] = 0.5; coef[2] = -1.0; coef[3] = 2.0
    coef[4] = 3.0; coef[5] = 0.7;  coef[6] = -1.5
    ξ, η = 0.13, -0.21
    ψ, pξ, pη, pξξ, pξη, pηη = eval_psi_and_derivs(coef, ξ, η)
    @test ψ ≈ 0.5 - ξ + 2η + 3ξ^2 + 0.7ξ * η - 1.5η^2
    @test pξ ≈ -1.0 + 6ξ + 0.7η
    @test pη ≈ 2.0 + 0.7ξ - 3η
    @test pξξ ≈ 6.0
    @test pξη ≈ 0.7
    @test pηη ≈ -3.0
    # eval_axisym_at_local should match ψ
    @test eval_axisym_at_local(coef, ξ, η) ≈ ψ
end

@testitem "find_axis_newton on synthetic bowl" begin
    # one reference triangle, large enough; ψ = (ξ-ξ0)²+(η-η0)² min at (ξ0,η0)
    a = b = c = 1.0;  θ = 0.0;  R0 = Z0 = 0.0
    ξ0, η0 = 0.2, 0.3
    # ψ = ξ² - 2ξ0 ξ + η² - 2η0 η + (ξ0²+η0²)
    coef = zeros(20)
    coef[1] = ξ0^2 + η0^2     # const
    coef[2] = -2ξ0            # ξ
    coef[3] = -2η0            # η
    coef[4] = 1.0             # ξ²
    coef[6] = 1.0             # η²
    elems = reshape([a, b, c, θ, R0, Z0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    cmat = reshape(coef, 20, 1)
    # global minimum location: ξ=ξ0, η=η0 → (R,Z)
    Rmin = R0 + (ξ0 + b) * 1.0 - η0 * 0.0
    Zmin = Z0 + (ξ0 + b) * 0.0 + η0 * 1.0
    ax = find_axis_newton(cmat, elems, Rmin + 0.05, Zmin - 0.05)
    @test ax.converged
    @test ax.R ≈ Rmin atol = 1.0e-3
    @test ax.Z ≈ Zmin atol = 1.0e-3
    @test ax.grad_norm < 1.0e-6
end

@testitem "toroidal average linearity" begin
    npp, nplanes = 4, 3
    coef3d = reshape(Float64.(1:(80 * npp * nplanes)), 80, npp * nplanes)
    ax = average_toroidal_axisymmetric(coef3d, nplanes)
    @test size(ax) == (80, npp)
    # check one element/coef equals the plane mean
    k, c = 2, 7
    manual = (coef3d[c, k] + coef3d[c, npp + k] + coef3d[c, 2npp + k]) / 3
    @test ax[c, k] ≈ manual

    # in-place variant writes the same result into a provided buffer, and
    # a dirty buffer is fully overwritten (fill!(0) guard) — this is what
    # lets the export reuse a pooled coefficient buffer across slices.
    buf = fill(9.9, 80, npp)
    ax2 = average_toroidal_axisymmetric!(buf, coef3d, nplanes)
    @test ax2 === buf
    @test ax2 == ax
    @test_throws ErrorException average_toroidal_axisymmetric!(zeros(80, npp + 1), coef3d, nplanes)
end

@testitem "flux coordinate helpers" begin
    @test psi_to_psi_norm(0.5, 0.0, 1.0) ≈ 0.5
    @test psi_n_to_rho_pol(0.25) ≈ 0.5
    @test psi_n_to_rho_pol(-1.0e-12) == 0.0        # clamp
end

@testitem "reduce_1d_psi_func: weights + gauss kernel" begin
    ρg = collect(range(0.0, 1.0; length = 6))    # h = 0.2, node 3 at ψ = 0.4
    ψp = [0.4, 0.4, 0.4];  f = [1.0, 1.0, 4.0];  w = [1.0, 1.0, 2.0]
    # on-node cluster: linear kernel gives the plain / weighted mean exactly
    ru = reduce_1d_psi_func(ψp, f; psi_grid = ρg)
    rw = reduce_1d_psi_func(ψp, f; psi_grid = ρg, weights = w)
    @test ru.func_bin[3] ≈ 2.0                   # (1+1+4)/3
    @test rw.func_bin[3] ≈ 2.5                   # Σwf/Σw = 10/4
    @test rw.den[3] ≈ 4.0                        # kernel-weighted mass Σw at the node
    @test rw.den[2] ≈ 0.0 atol = 1.0e-12           # nothing deposits off-node
    # a NaN weight drops that sample
    rn = reduce_1d_psi_func(ψp, f; psi_grid = ρg, weights = [1.0, 1.0, NaN])
    @test rn.func_bin[3] ≈ 1.0
    # gauss kernel: same weighted mean on the cluster (all samples at one ψ)
    rgw = reduce_1d_psi_func(ψp, f; psi_grid = ρg, adj = :gauss, weights = w)
    @test rgw.func_bin[3] ≈ 2.5
    # gauss on dense uniform samples: constant is exact; a linear ramp is
    # reproduced at a fully-interior node (symmetric truncated kernel)
    ψs = collect(range(0.0, 1.0; length = 401))
    rc = reduce_1d_psi_func(ψs, fill(3.5, length(ψs)); psi_grid = ρg, adj = :gauss)
    @test all(x -> isapprox(x, 3.5; rtol = 1.0e-12), filter(isfinite, rc.func_bin))
    rl = reduce_1d_psi_func(ψs, ψs; psi_grid = ρg, adj = :gauss, sigma_cells = 0.75)
    @test rl.func_bin[3] ≈ ρg[3] atol = 0.01
end

@testitem "mesh_boundary_rz" begin
    # single element (a=b=c=1, θ=0, origin (0,0)) → its own triangle
    ep1 = reshape([1.0, 1, 1, 0, 0, 0, 0, 1, 0, 0], 10, 1)
    bd1 = mesh_boundary_rz(ep1)
    @test length(bd1.R) == 3
    @test Set(zip(bd1.R, bd1.Z)) == Set([(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)])
    # two elements sharing the (0,0)–(2,0) edge → 4-point diamond, the
    # shared edge is interior and must not appear in the outline
    ep2 = hcat(
        [1.0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [1.0, 1, 1, π, 2, 0, 0, 1, 0, 0]
    )
    bd2 = mesh_boundary_rz(ep2)
    @test length(bd2.R) == 4
    pts = collect(zip(bd2.R, bd2.Z))
    for want in ((0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, -1.0))
        @test any(
            p -> isapprox(p[1], want[1]; atol = 1.0e-9) &&
                isapprox(p[2], want[2]; atol = 1.0e-9), pts
        )
    end
    # consecutive outline points never form the interior edge
    n = length(pts)
    for i in 1:n
        p, q = pts[i], pts[mod1(i + 1, n)]
        @test !(Set([p[1], q[1]]) == Set([0.0, 2.0]) && p[2] == 0.0 == q[2])
    end
    # row-7 boundary-edge flags path: tag every edge of the single element
    # as boundary (bits 1|2|4 = 7) → same triangle outline via the fast path
    ep1f = reshape([1.0, 1, 1, 0, 0, 0, 7, 1, 0, 0], 10, 1)
    bd1f = mesh_boundary_rz(ep1f)
    @test Set(zip(bd1f.R, bd1f.Z)) == Set([(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)])
end

@testitem "mesh_zone_boundary_rz + elem_zones" begin
    # elem_zones reads the constant Hermite coefficient as the region id
    @test elem_zones(Float64[1.0 2.0 3.0]) == [1, 2, 3]
    @test elem_zones(zeros(80, 4) .+ 2.0)[1] == 2

    diamond(shift) = hcat(
        [1.0, 1, 1, 0, shift, 0, 2, 1, 0, 0],
        [1.0, 1, 1, π, shift + 2, 0, 2, 1, 0, 0]
    )
    # one conductor (zone 2) diamond → one 4-point loop; absent zone → []
    ep = diamond(0.0);  zones = [2, 2]
    loops = mesh_zone_boundary_rz(ep, zones, 2)
    @test length(loops) == 1 && length(loops[1].R) == 4
    lpts = collect(zip(loops[1].R, loops[1].Z))
    for want in ((0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, -1.0))
        @test any(
            p -> isapprox(p[1], want[1]; atol = 1.0e-9) &&
                isapprox(p[2], want[2]; atol = 1.0e-9), lpts
        )
    end
    @test mesh_zone_boundary_rz(ep, zones, 3) == NamedTuple[]

    # two disjoint conductor diamonds → two loops (the annular / multi-
    # surface case a real resistive wall produces: inner + outer surface)
    ep2 = hcat(diamond(0.0), diamond(10.0));  zones2 = [2, 2, 2, 2]
    loops2 = mesh_zone_boundary_rz(ep2, zones2, 2)
    @test length(loops2) == 2
    @test all(l -> length(l.R) == 4, loops2)

    # a zone filter picks only its own elements: mark the second diamond
    # zone 1 (plasma) → zone-2 query returns just the first loop
    zones3 = [2, 2, 1, 1]
    @test length(mesh_zone_boundary_rz(ep2, zones3, 2)) == 1
    @test length(mesh_zone_boundary_rz(ep2, zones3, 1)) == 1
end

@testitem "interpolate_axisym_gradient_to_grid (rotation + FD pin)" begin
    # rotated element (θ=0.7) so the local→global gradient rotation is
    # nontrivial; generic cross-term quadratic (same fixture as the
    # critical_point_system rotation test)
    a = b = c = 1.0;  θ = 0.7;  R0 = 2.0;  Z0 = 0.3
    ep = reshape([a, b, c, θ, R0, Z0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    coef = zeros(20)
    coef[1] = 0.3; coef[2] = 0.5; coef[3] = -0.7
    coef[4] = 1.2; coef[5] = 0.9; coef[6] = -0.4
    cm = reshape(coef, 20, 1)
    Rg = collect(range(2.7, 2.78, length = 4))
    Zg = collect(range(1.08, 1.16, length = 4))
    g = interpolate_axisym_gradient_to_grid(cm, ep, Rg, Zg)
    @test isapprox(g.val, interpolate_axisym_to_grid(cm, ep, Rg, Zg); nans = true)
    @test any(isfinite, g.val)                   # grid lands in the element
    ψofRZ(Rp, Zp) = (
        lc = M3DC1Reader.global_to_local(Rp, Zp, R0, Z0, b, θ);
        M3DC1Reader.eval_psi_and_derivs(coef, lc[1], lc[2])[1]
    )
    h = 1.0e-6
    for j in eachindex(Zg), i in eachindex(Rg)
        isfinite(g.val[i, j]) || continue
        @test g.dR[i, j] ≈ (ψofRZ(Rg[i] + h, Zg[j]) - ψofRZ(Rg[i] - h, Zg[j])) / 2h atol = 1.0e-6
        @test g.dZ[i, j] ≈ (ψofRZ(Rg[i], Zg[j] + h) - ψofRZ(Rg[i], Zg[j] - h)) / 2h atol = 1.0e-6
    end
    # off-mesh grid → NaN in all three outputs
    g0 = interpolate_axisym_gradient_to_grid(cm, ep, [10.0, 11.0], [5.0, 6.0])
    @test all(isnan, g0.val) && all(isnan, g0.dR) && all(isnan, g0.dZ)
end

@testitem "unit system" begin
    # m3d_smoke normalization: b0=1e4 G, n0=1e14 cm^-3, l0=100 cm, mi=2
    # These conveniently make several SI factors unit-valued.
    u = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)

    @test unit_factor(u, :magnetic_field; system = :si) ≈ 1.0      # 1e4 G = 1 T
    @test unit_factor(u, :length; system = :si) ≈ 1.0      # 100 cm = 1 m
    @test unit_factor(u, :density; system = :si) ≈ 1.0e20     # 1e14 cm^-3
    @test unit_factor(u, :magnetic_field; system = :cgs) ≈ 1.0e4     # Gauss
    @test unit_factor(u, :length; system = :cgs) ≈ 100.0   # cm

    # :m3d and :mks alias
    @test unit_factor(u, :temperature; system = :m3d) == 1.0
    @test unit_factor(u, :density; system = :mks) == unit_factor(u, :density; system = :si)

    # temperature factor matches the closed form B0²/(4π N0 e_erg)
    @test unit_factor(u, :temperature; system = :si) ≈ 1.0e8 / (4π * 1.0e14 * 1.6022e-12)

    # to_units broadcasts
    @test to_units(u, [1.0, 2.0], :magnetic_field; system = :si) ≈ [1.0, 2.0]

    # power_density = energy-density / time (erg/cm^3/s → W/m^3)
    @test unit_factor(u, :power_density; system = :cgs) ≈ unit_factor(u, :pressure; system = :cgs) / u.t0
    @test unit_label(:power_density; system = :si) == "W/m^3"
    @test :power_density in available_quantities()

    # power = energy / time, in both systems (erg/s → W)
    @test unit_factor(u, :power; system = :cgs) ≈ unit_factor(u, :energy; system = :cgs) / u.t0
    @test unit_factor(u, :power; system = :si) ≈ unit_factor(u, :energy; system = :si) / u.t0
    @test unit_label(:power; system = :si) == "W"

    # labels + discovery + error paths
    @test unit_label(:temperature; system = :si) == "eV"
    @test unit_label(:magnetic_field; system = :si) == "T"
    @test unit_label(:magnetic_field; system = :m3d) == "normalized"
    @test :temperature in available_quantities()
    @test_throws ErrorException unit_factor(u, :nonexistent)
    @test_throws ErrorException unit_factor(u, :temperature; system = :bogus)
end

@testitem "reduce_1d_psi_func basic" begin
    # points on a line Te = 10*(1-ρ): bin-mean should recover the line
    ρ = collect(range(0, 1, length = 500))
    Te = 10 .* (1 .- ρ)
    r = reduce_1d_psi_func(ρ, Te; n_bins = 20, psi_range = (0.0, 1.0), adj = :linear)
    @test length(r.func_bin) == 20
    good = isfinite.(r.func_bin)
    @test all(abs.(r.func_bin[good] .- 10 .* (1 .- r.psi_grid[good])) .< 0.5)

    # NaN handling + weights: NaN samples dropped, weights honoured — must
    # match a hand-filtered weighted bin mean (guards the pooled finite-gather).
    ρn = copy(ρ);  ρn[1:10] .= NaN
    Ten = copy(Te);  Ten[20:25] .= NaN
    wv = 1.0 .+ ρ
    rn = reduce_1d_psi_func(
        ρn, Ten; psi_grid = collect(range(0, 1, length = 20)),
        weights = wv, adj = :constant
    )
    m = isfinite.(ρn) .& isfinite.(Ten)
    ref = reduce_1d_psi_func(
        ρn[m], Ten[m]; psi_grid = collect(range(0, 1, length = 20)),
        weights = wv[m], adj = :constant
    )
    @test rn.func_bin == ref.func_bin            # pre-filtering must not change the result
end
