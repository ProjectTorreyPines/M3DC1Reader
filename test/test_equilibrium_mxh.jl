@testitem "to_mxh" begin
    if Base.find_package("MXHEquilibrium") === nothing
        @info "skipping to_mxh (MXHEquilibrium unavailable)"
    else
        @eval using MXHEquilibrium
        # Synthetic nested-bowl ψ(R,Z) with a single interior minimum (axis):
        # ψ = (R - 1.5)² + Z², minimum at (1.5, 0). F = R·Bφ ≈ const (vacuum-like).
        R = range(1.0, 2.0; length = 33)
        Z = range(-1.0, 1.0; length = 33)
        psi_rz = [(r - 1.5)^2 + z^2 for r in R, z in Z]
        # Non-uniform 1D ψ grid (quadratic in ρ) exercises the resample path.
        ρ = range(0.0, 1.0; length = 16)
        psi1d = collect(0.25 .* ρ .^ 2)
        F1d = fill(3.4, 16)                      # F = R·Bφ ~ const (vacuum-like)
        eq = M3DAxisymField(;
            cocos = 1, time = 0.0, R = R, Z = Z, psi_rz = psi_rz,
            psi1d = psi1d, F1d = F1d, p1d = nothing, axis = (1.5, 0.0),
            psi_axis = 0.0, psi_boundary = 0.25, r0 = 1.5, b0 = 3.4 / 1.5
        )

        M = to_mxh(eq)
        @test M isa MXHEquilibrium.EFITEquilibrium

        # |B| evaluator works (the δB/B enabler). At the axis |B| ≈ |Bφ| = F/R.
        B = Bfield(M, 1.5, 0.0)
        @test all(isfinite, B)
        @test sqrt(sum(abs2, B)) ≈ 3.4 / 1.5 rtol = 0.05     # F/R at axis

        # EFIT-native safety_factor returns the stored profile (to_mxh passes
        # q=zeros); real q comes from the FSA path below.
        @test safety_factor(M, -0.05) == 0.0

        # COCOS-driven FSA q on an interior surface: flux_surface must trace a
        # closed surface (validates the ψ-shift) and the FSA integral is finite.
        # ψ is shifted so boundary→0, axis→-0.25; sample partway out.
        psi_i = -0.15
        fs = MXHEquilibrium.flux_surface(M, psi_i)
        @test fs !== nothing
        g = MXHEquilibrium.poloidal_current(M, psi_i)
        qfsa = (g / (2π)) *
            MXHEquilibrium.average(fs, (x, y) -> inv(MXHEquilibrium.poloidal_Bfield(M, x, y) * x^2)) *
            MXHEquilibrium.circumference(fs)
        @test isfinite(qfsa)
        @test qfsa > 0   # co-current COCOS=5 gives positive q

        @testset "NaN-fill off-mesh (dilation)" begin
            # Poke NaNs into the four corners of the synthetic bowl's psi_rz,
            # well outside the plasma region (ψ_boundary = 0.25; the corners
            # have ψ = (R-1.5)² + Z² = 1.25, i.e. off-mesh). This mirrors the
            # real M3D-C1 data where interpolate_axisym_to_grid leaves NaNs
            # outside the mesh. The interior/axis must stay untouched and the
            # NaN-fill (dilation) must keep downstream evaluations finite.
            psi_rz_nan = copy(psi_rz)
            psi_rz_nan[1, 1] = NaN
            psi_rz_nan[1, end] = NaN
            psi_rz_nan[end, 1] = NaN
            psi_rz_nan[end, end] = NaN
            @test any(isnan, psi_rz_nan)

            eq_nan = M3DAxisymField(;
                cocos = 1, time = 0.0, R = R, Z = Z, psi_rz = psi_rz_nan,
                psi1d = psi1d, F1d = F1d, p1d = nothing, axis = (1.5, 0.0),
                psi_axis = 0.0, psi_boundary = 0.25, r0 = 1.5, b0 = 3.4 / 1.5
            )
            M_nan = to_mxh(eq_nan)
            @test M_nan isa MXHEquilibrium.EFITEquilibrium

            # Axis |B| must remain finite and physical despite the off-mesh NaNs.
            B_nan = Bfield(M_nan, 1.5, 0.0)
            @test all(isfinite, B_nan)
            @test sqrt(sum(abs2, B_nan)) ≈ 3.4 / 1.5 rtol = 0.05

            # An interior flux surface + FSA q must also stay finite: the
            # off-mesh NaNs did not poison the interior spline.
            fs_nan = MXHEquilibrium.flux_surface(M_nan, psi_i)
            @test fs_nan !== nothing
            g_nan = MXHEquilibrium.poloidal_current(M_nan, psi_i)
            qfsa_nan = (g_nan / (2π)) *
                MXHEquilibrium.average(fs_nan, (x, y) -> inv(MXHEquilibrium.poloidal_Bfield(M_nan, x, y) * x^2)) *
                MXHEquilibrium.circumference(fs_nan)
            @test isfinite(qfsa_nan)
        end
    end
end
