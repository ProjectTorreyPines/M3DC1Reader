# Cross-validation of the two equilibrium adapters on real SPI data. Fed from the
# same `M3DAxisymField`, MXHEquilibrium's FSA `safety_factor` (`to_mxh`) and
# IMAS's `flux_surfaces` q (`to_imas`) must agree at the same physical surface, up
# to the known COCOS convention factor: the dd is populated in M3D-C1's native
# per-radian ψ (COCOS≈5) and IMAS interprets it as COCOS 11 (2π-in-ψ), so
# q_dd = -2π · q_efit (the sign is the co-current convention). Two independent
# implementations agreeing to that constant validates F(ψ), the equilibrium
# assembly, and both adapters at once.
#
# Gated on the real C1.h5 (env `M3DC1_TEST_FILE`) AND MXHEquilibrium + IMASdd +
# IMAS being loadable, so it SKIPS under the standard runner (IMAS is weakdep-only
# and too heavy for the package test env). Validated by hand on spi_example/C1.h5:
# q_dd/q_efit = -6.283 ± 0.005 across ψ_norm∈[0.1,0.9] (= -2π to 4 sig figs);
# q_efit = 1.01 → 2.94 (physical: q0≈1, rising outward).
@testitem "q cross-validation (to_mxh vs to_imas)" begin
    c1 = get(
        ENV, "M3DC1_TEST_FILE",
        "/Users/yoo/Research/M3D-C1/test_Data/spi_example/C1.h5"
    )
    have = isfile(c1) &&
        Base.find_package("MXHEquilibrium") !== nothing &&
        Base.find_package("IMASdd") !== nothing &&
        Base.find_package("IMAS") !== nothing
    if !have
        @info "skipping q cross-validation (need C1.h5 + MXHEquilibrium + IMASdd + IMAS)"
    else
        @eval using MXHEquilibrium, IMASdd, IMAS
        file = M3DC1File(c1)
        eq = axisym_field(file, first(list_timeslices(file)); ngrid = 129, nbins = 64)

        # efit FSA q via the GENERIC AbstractEquilibrium method (EFITEquilibrium
        # overrides safety_factor to return the stored profile, which is zeros here).
        Me = to_mxh(eq)
        pa, pb = MXHEquilibrium.psi_limits(Me)
        q_efit(pn) = invoke(
            MXHEquilibrium.safety_factor,
            Tuple{MXHEquilibrium.AbstractEquilibrium, Any},
            Me, pa + pn * (pb - pa)
        )

        # IMAS q vs its own psi_norm, linearly interpolated onto shared ψ_norm.
        p1d = to_imas(eq; trace = true).equilibrium.time_slice[1].profiles_1d
        xq = collect(Float64, p1d.psi_norm); yq = collect(Float64, p1d.q)
        pp = sortperm(xq); xq = xq[pp]; yq = yq[pp]
        function q_dd(pn)
            pn <= xq[1] && return yq[1]
            pn >= xq[end] && return yq[end]
            k = searchsortedlast(xq, pn); t = (pn - xq[k]) / (xq[k + 1] - xq[k])
            return (1 - t) * yq[k] + t * yq[k + 1]
        end

        pns = 0.1:0.1:0.9
        qe = [q_efit(pn) for pn in pns]
        qd = [q_dd(pn) for pn in pns]

        # to_mxh q under the native COCOS-1 label: negative (counter-current:
        # field-frame Ip>0, F<0 ⇒ sign(q)=sign(Ip·F)<0), |q0|≈1, |q| rising.
        @test all(qe .< 0)
        @test qe[1] ≈ -1.0 atol = 0.3
        @test issorted(abs.(qe))
        # the two independent implementations agree up to the +2π factor
        # (per-radian ψ fed to IMAS as-if-total-flux; same σ_ρθφ as COCOS 1).
        @test all(r -> isapprox(r, 2π; rtol = 0.02), qd ./ qe)
    end
end
