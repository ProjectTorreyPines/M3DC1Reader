# Tests for the globalized (damped) Newton critical-point finder.
#   * pure-solver tests use hand-written analytic (residual, Jacobian) systems;
#   * FEM tests use the synthetic single-element mesh pattern (no C1.h5);
#   * a rotation test pins the local→global gradient/Hessian transform;
#   * real-data O/X reproduction is guarded on the presence of a C1.h5 file.
# The bowl+saddle fixture (lelems/lcoef/lc/ψsad) lives in test/setup.jl
# (@testsnippet LcfsFixture).

@testitem "damped_newton_2d (analytic systems)" begin
    # 1) quadratic bowl F=∇[(R-1)²+(Z-2)²] — one Newton step nails it
    bowl(R, Z) = (2(R - 1), 2(Z - 2), 2.0, 0.0, 0.0, 2.0, true)
    s = damped_newton_2d(bowl, 5.0, -3.0)
    @test s.converged
    @test s.R ≈ 1.0 atol = 1.0e-9
    @test s.Z ≈ 2.0 atol = 1.0e-9

    # 2) saddle F=∇[(R-1)²-(Z-2)²], det J = -4 < 0 — still converges
    saddle(R, Z) = (2(R - 1), -2(Z - 2), 2.0, 0.0, 0.0, -2.0, true)
    s2 = damped_newton_2d(saddle, -4.0, 7.0)
    @test s2.converged
    @test s2.R ≈ 1.0 atol = 1.0e-9
    @test s2.Z ≈ 2.0 atol = 1.0e-9

    # 3) line search RESCUES a case where a plain Newton step diverges:
    #    F=(atan R, atan Z); undamped x←x-atan(x)(1+x²) blows up for |x|>1.39.
    at(R, Z) = (atan(R), atan(Z), 1 / (1 + R^2), 0.0, 0.0, 1 / (1 + Z^2), true)
    s3 = damped_newton_2d(at, 5.0, -5.0; maxit = 100)
    @test s3.converged
    @test s3.R ≈ 0.0 atol = 1.0e-6
    @test s3.Z ≈ 0.0 atol = 1.0e-6

    # 4) singular J (det≡0) → the gradient-descent fallback still drives R to
    #    the root instead of dividing by zero. The fallback step is a small
    #    normalized 1e-2 (deliberately conservative — real singularities are
    #    transient), so seed near the root; the point is that it makes safe,
    #    monotone progress and converges rather than blowing up.
    nearsing(R, Z) = (2(R - 1), 0.0, 2.0, 0.0, 0.0, 0.0, true)
    s4 = damped_newton_2d(nearsing, 1.05, 0.0; maxit = 200)
    @test s4.converged
    @test s4.R ≈ 1.0 atol = 1.0e-5
    @test isfinite(s4.R)                   # never divided by the zero det

    # 5) domain guard: the only root (R=-1) is outside the valid region R>0,
    #    so the solver must stay in and fail gracefully (no escape, no crash)
    onlypos(R, Z) = R > 0 ? (R + 1, Z, 1.0, 0.0, 0.0, 1.0, true) :
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)
    s5 = damped_newton_2d(onlypos, 2.0, 0.0; maxit = 50)
    @test s5.R > 0
    @test !s5.converged
end

@testitem "critical_point_system global gradient/Hessian (rotation)" begin
    using M3DC1Reader: eval_psi_and_derivs, global_to_local   # non-exported helpers
    # rotated element (θ=0.7) so the local→global transform is nontrivial
    a = b = c = 1.0;  θ = 0.7;  R0 = 2.0;  Z0 = 0.3
    elems = reshape([a, b, c, θ, R0, Z0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    # a generic (cross-term) quadratic in local coords
    coef = zeros(20)
    coef[1] = 0.3; coef[2] = 0.5; coef[3] = -0.7; coef[4] = 1.2; coef[5] = 0.9; coef[6] = -0.4
    eqs = critical_point_system(reshape(coef, 20, 1), elems)
    co, sn = cos(θ), sin(θ)
    ξ, η = 0.1, 0.15                        # interior local point
    R = R0 + (ξ + b) * co - η * sn
    Z = Z0 + (ξ + b) * sn + η * co
    F1, F2, J11, J12, J21, J22, ok = eqs(R, Z)
    @test ok
    @test J12 == J21                         # symmetric Hessian
    # finite-difference the GLOBAL gradient/Hessian of ψ(R,Z)
    function ψofRZ(Rp, Zp)
        ξ2, η2 = global_to_local(Rp, Zp, R0, Z0, b, θ)
        return eval_psi_and_derivs(coef, ξ2, η2)[1]
    end
    h = 1.0e-6
    @test F1 ≈ (ψofRZ(R + h, Z) - ψofRZ(R - h, Z)) / 2h atol = 1.0e-6
    @test F2 ≈ (ψofRZ(R, Z + h) - ψofRZ(R, Z - h)) / 2h atol = 1.0e-6
    gR(R, Z) = eqs(R, Z)[1]
    gZ(R, Z) = eqs(R, Z)[2]
    @test J11 ≈ (gR(R + h, Z) - gR(R - h, Z)) / 2h atol = 1.0e-5
    @test J22 ≈ (gZ(R, Z + h) - gZ(R, Z - h)) / 2h atol = 1.0e-5
    @test J12 ≈ (gR(R, Z + h) - gR(R, Z - h)) / 2h atol = 1.0e-5
end

@testitem "find_critical_point on synthetic FEM element" begin
    a = b = c = 1.0;  θ = 0.7;  R0 = 2.0;  Z0 = 0.3
    elems = reshape([a, b, c, θ, R0, Z0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    co, sn = cos(θ), sin(θ)
    loc2glob(ξ, η) = (R0 + (ξ + b) * co - η * sn, Z0 + (ξ + b) * sn + η * co)
    ξ0, η0 = 0.15, 0.2

    # (a) BOWL ⇒ extremum (O-point-like): ψ=(ξ-ξ0)²+(η-η0)², det H = +4
    cb = zeros(20); cb[1] = ξ0^2 + η0^2; cb[2] = -2ξ0; cb[3] = -2η0; cb[4] = 1.0; cb[6] = 1.0
    Rm, Zm = loc2glob(ξ0, η0)
    o = find_critical_point(reshape(cb, 20, 1), elems, Rm + 0.03, Zm - 0.03)
    @test o.converged
    @test o.R ≈ Rm atol = 1.0e-8
    @test o.Z ≈ Zm atol = 1.0e-8
    @test o.grad_norm < 1.0e-9
    @test o.kind === :extremum
    @test o.D ≈ 4.0 atol = 1.0e-9
    # find_o_point is the same solve, seeded as the axis
    @test find_o_point(reshape(cb, 20, 1), elems, Rm + 0.03, Zm - 0.03).kind === :extremum

    # (b) SADDLE ⇒ X-point-like: ψ=(ξ-ξ0)²-(η-η0)², det H = -4
    cs = zeros(20); cs[1] = ξ0^2 - η0^2; cs[2] = -2ξ0; cs[3] = 2η0; cs[4] = 1.0; cs[6] = -1.0
    Rs, Zs = loc2glob(ξ0, η0)
    x = find_critical_point(reshape(cs, 20, 1), elems, Rs - 0.03, Zs + 0.03)
    @test x.converged
    @test x.R ≈ Rs atol = 1.0e-8
    @test x.Z ≈ Zs atol = 1.0e-8
    @test x.kind === :saddle
    @test x.D ≈ -4.0 atol = 1.0e-9
    @test find_x_point(reshape(cs, 20, 1), elems, Rs - 0.03, Zs + 0.03).kind === :saddle
end

@testitem "classify_critical discriminant" begin
    @test classify_critical(4.0) === :extremum
    @test classify_critical(-4.0) === :saddle
    @test classify_critical(0.0) === :degenerate
    @test classify_critical(1.0e-3; tol = 1.0e-2) === :degenerate
end

@testitem "eval_axisym_at (global point)" setup = [LcfsFixture] begin
    # matches the local evaluation at a mapped point; NaN off-mesh
    @test eval_axisym_at(lcoef, lelems, 2.3, -0.6) ≈
        eval_axisym_at_local(lc, 0.3, 0.4) atol = 1.0e-14
    @test isnan(eval_axisym_at(lcoef, lelems, 10.0, 10.0))
end

@testitem "find_lcfs candidate selection (synthetic bowl+saddle)" setup = [LcfsFixture] begin
    seedO = (2.05, -0.45);  seedX = (2.0, 0.1)

    # (a) DIVERTED: X-point saddle beats a far limiter point
    #     limiter at local (0.5, 0.2) = global (2.5, −0.8): ψ = 0.367 > 4/27
    r = find_lcfs(
        lcoef, lelems; xmag = seedO[1], zmag = seedO[2],
        xnull = seedX[1], znull = seedX[2], xlim = 2.5, zlim = -0.8
    )
    @test r.is_diverted
    @test r.limited_by === :xpoint1
    @test r.psi_axis ≈ 0.0 atol = 1.0e-10
    @test r.psi_bound ≈ ψsad atol = 1.0e-8
    @test r.x1.kind === :saddle
    @test r.x1.R ≈ 2.0 atol = 1.0e-8
    @test r.x1.Z ≈ 1 / 6 atol = 1.0e-8
    @test r.x2 === nothing
    @test isfinite(r.psilim)                     # limiter evaluated, just lost

    # (b) LIMITED: limiter #2 closer to ψ0 than the saddle
    #     lim1 far (0.367); lim2 at local (0.1, 0.6) = global (2.1, −0.4): ψ = 0.019
    r2 = find_lcfs(
        lcoef, lelems; xmag = seedO[1], zmag = seedO[2],
        xnull = seedX[1], znull = seedX[2],
        xlim = 2.5, zlim = -0.8, xlim2 = 2.1, zlim2 = -0.4
    )
    @test !r2.is_diverted
    @test r2.limited_by === :limiter2
    @test r2.psi_bound ≈ eval_axisym_at(lcoef, lelems, 2.1, -0.4) atol = 1.0e-12
    @test abs(r2.psi_bound - r2.psi_axis) < abs(r2.x1.ψ - r2.psi_axis)

    # (c) no candidates at all → :none, NaN boundary
    r3 = find_lcfs(lcoef, lelems; xmag = seedO[1], zmag = seedO[2])
    @test r3.limited_by === :none
    @test isnan(r3.psi_bound)
    @test r3.x1 === nothing && r3.x2 === nothing

    # (d) WALL scan wins: wall node at (2, −0.4) has ψ = 0.009 < 4/27;
    #     off-mesh wall point is skipped silently
    r4 = find_lcfs(
        lcoef, lelems; xmag = seedO[1], zmag = seedO[2],
        xnull = seedX[1], znull = seedX[2],
        wall_rz = [(2.0, -0.4), (10.0, 10.0)]
    )
    @test !r4.is_diverted
    @test r4.limited_by === :wall
    @test r4.psi_bound ≈ eval_axisym_at(lcoef, lelems, 2.0, -0.4) atol = 1.0e-12

    # (e) Z·Z₀ wall filter: the point (2, +0.6) has |ψ−ψ0| = 0.121 < 4/27 and
    #     WOULD win, but sits in the opposite vertical half (Z₀ = −1/2) → it
    #     must be filtered out, leaving the X-point as the boundary
    @test abs(eval_axisym_at(lcoef, lelems, 2.0, 0.6) - 0.0) < ψsad   # would-be winner
    r5 = find_lcfs(
        lcoef, lelems; xmag = seedO[1], zmag = seedO[2],
        xnull = seedX[1], znull = seedX[2], wall_rz = [(2.0, 0.6)]
    )
    @test r5.is_diverted
    @test r5.limited_by === :xpoint1
    @test isnan(r5.psib)

    # (f) xnull ≤ 0 ⇒ "not tracked" (M3D-C1 convention): no X candidate
    r6 = find_lcfs(
        lcoef, lelems; xmag = seedO[1], zmag = seedO[2],
        xnull = 0.0, znull = 0.0, xlim = 2.5, zlim = -0.8
    )
    @test r6.x1 === nothing
    @test r6.limited_by === :limiter1
end

# ---- real-data O/X reproduction (guarded) ----
@testitem "C1.h5 O-point + X-point (damped Newton on n=0 ψ)" begin
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping C1.h5 critical-point tests (no file at $c1)"
    else
        file = M3DC1File(c1);  ep = elems_plane(file)
        ts = last(list_timeslices(file))
        sl = read_timeslice(file, ts; fields = (:psi,))
        psi_avg = average_toroidal_axisymmetric(read_field(file, ts, :psi), file.nplanes)
        Δψ = abs(sl.psi_lcfs - sl.psi_axis)

        # O-point: seed at stored (xmag,zmag) → an extremum matching find_axis_newton
        o = find_o_point(psi_avg, ep, sl.xmag, sl.zmag)
        oa = find_axis_newton(psi_avg, ep, sl.xmag, sl.zmag)
        @test o.converged
        @test o.kind === :extremum
        @test o.grad_norm < 1.0e-8
        @test o.R ≈ oa.R atol = 1.0e-4
        @test o.Z ≈ oa.Z atol = 1.0e-4
        @test o.ψ ≈ oa.ψ atol = 1.0e-6 * max(1.0, Δψ)

        # X-point: seed at stored (xnull,znull) when present & inside the mesh
        if isfinite(sl.xnull) && minimum(ep[5, :]) < sl.xnull < maximum(ep[5, :])
            x = find_x_point(psi_avg, ep, sl.xnull, sl.znull)
            @test x.converged
            @test x.kind === :saddle                         # genuine saddle
            @test x.grad_norm < 1.0e-8
            # recomputed X-point ψ sits at the LCFS flux (few-% of the axis→edge span)
            @test abs(x.ψ - sl.psi_lcfs) < 0.05 * Δψ
            # stays near the stored plane-1 X-point (didn't wander to the O-point)
            @test hypot(x.R - sl.xnull, x.Z - sl.znull) < 0.1
        end

        # full LCFS determination reproduces M3D-C1's stored psi_lcfs
        lim = limiter_points(file)
        if isfinite(lim.xlim) && lim.xlim > 0 && isfinite(sl.xnull) && sl.xnull > 0
            r = find_lcfs(file, ts)
            @test r.limited_by in (:xpoint1, :xpoint2, :limiter1, :limiter2)
            # stored psi_lcfs (plane-1) vs our n=0 recomputation: sub-% of span
            @test abs(r.psi_bound - sl.psi_lcfs) < 0.01 * Δψ
            @test r.x1 !== nothing && r.x1.kind === :saddle
            # limiter candidates were evaluated (attrs present in this file)
            @test isfinite(r.psilim)
            if r.is_diverted
                # diverted ⇒ every limiter candidate is farther from ψ0 than the winner
                @test abs(r.psilim - r.psi_axis) > abs(r.psi_bound - r.psi_axis)
            end
        end
    end
end
