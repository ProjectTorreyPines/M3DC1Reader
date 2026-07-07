# to_geqdsk builds an in-memory EFIT.GEQDSKFile from a M3DAxisymField, reusing
# to_mxh (MXHEquilibrium) for q / Ip / boundary / rhovn. EFIT + MXHEquilibrium are
# in test/Project.toml, so the SYNTHETIC testitem RUNS under the runner. The
# real-data testitem is gated on the SPI C1.h5 and skips otherwise.
@testitem "to_geqdsk (synthetic)" begin
    if Base.find_package("EFIT") === nothing || Base.find_package("MXHEquilibrium") === nothing
        @info "skipping to_geqdsk (EFIT/MXHEquilibrium unavailable)"
    else
        @eval using EFIT, MXHEquilibrium
        R = range(1.0, 2.0; length = 33); Z = range(-1.0, 1.0; length = 33)
        psi_rz = [ (r - 1.5)^2 + z^2 for r in R, z in Z ]   # nested closed surfaces
        psi1d = collect(range(0.0, 0.25; length = 16)); F1d = fill(3.4, 16)
        eq = M3DAxisymField(;
            cocos = 1, time = 0.5, R = R, Z = Z, psi_rz = psi_rz,
            psi1d = psi1d, F1d = F1d, p1d = nothing, axis = (1.5, 0.0),
            psi_axis = 0.0, psi_boundary = 0.25, r0 = 1.5, b0 = 3.4 / 1.5
        )
        g = to_geqdsk(eq)

        @test g isa EFIT.GEQDSKFile
        # struct/grid consistency mapped straight from the field
        @test (g.nw, g.nh) == (33, 33)
        @test size(g.psirz) == (33, 33)
        @test g.simag == 0.0 && g.sibry == 0.25
        @test (g.rmaxis, g.zmaxis) == (1.5, 0.0)
        @test g.rcentr == 1.5
        @test g.time == 0.5
        @test g.psirz == psi_rz                       # unshifted ψ map preserved
        # every 1D profile has length npsi and is fully finite
        npsi = length(g.psi)
        @test all(length.((g.fpol, g.pres, g.ffprim, g.pprime, g.qpsi, g.rhovn)) .== npsi)
        @test all(isfinite, g.qpsi)
        @test all(isfinite, g.rhovn)
        # normalized toroidal flux runs axis→boundary
        @test g.rhovn[1] ≈ 0.0 atol = 1.0e-6
        @test g.rhovn[end] ≈ 1.0 atol = 1.0e-6
        # q is single-signed and finite (co-current synthetic → positive)
        @test all(g.qpsi .> 0)
        # LCFS boundary is a real, non-trivial polygon
        @test g.nbbbs > 10
        @test length(g.rlim) == length(g.zlim) == 5   # box limiter
    end
end

@testitem "to_geqdsk (real SPI data)" begin
    c1 = get(
        ENV, "M3DC1_TEST_FILE",
        "/Users/yoo/Research/M3D-C1/test_Data/spi_example/C1.h5"
    )
    have = isfile(c1) && Base.find_package("EFIT") !== nothing &&
        Base.find_package("MXHEquilibrium") !== nothing
    if !have
        @info "skipping to_geqdsk real-data (need C1.h5 + EFIT + MXHEquilibrium)"
    else
        @eval using EFIT, MXHEquilibrium
        file = M3DC1File(c1)
        eq = axisym_field(file, first(list_timeslices(file)); ngrid = 129, nbins = 64)
        g = to_geqdsk(eq)
        # validated: Ip ≈ +2.41 MA (field frame), |q| rises 1.0 → ~3.8 with the
        # COCOS-1 sign (counter-current: Ip>0, F<0 ⇒ q<0), rmaxis ≈ 2.975 m
        @test g isa EFIT.GEQDSKFile
        @test g.rmaxis ≈ 2.975 atol = 0.05
        @test (g.rcentr, g.bcentr) == (3.0, 2.5)
        @test 1.0e6 < abs(g.current) < 5.0e6          # ~2.4 MA
        @test all(isfinite, g.qpsi)
        @test g.qpsi[1] ≈ -1.0 atol = 0.3             # q0 ≈ -1 (COCOS-1 sign)
        @test minimum(g.qpsi) < -2.0                  # |q| rises outward
        @test sign(g.qpsi[1]) == sign(g.current * g.fpol[end])   # EFIT sign rule
        @test g.fpol[end] ≈ -7.5 atol = 0.3           # F ≈ -b0·r0
        @test g.rhovn[1] ≈ 0.0 atol = 1.0e-6
        @test g.rhovn[end] ≈ 1.0 atol = 1.0e-6
    end
end
