@testitem "_collapse_zeta: value + toroidal-derivative layers" begin
    # one element, layers with distinct constants: u(ζ) = 1 + 2ζ + 3ζ² + 4ζ³
    coef = zeros(80, 1)
    coef[1, 1] = 1.0;  coef[21, 1] = 2.0;  coef[41, 1] = 3.0;  coef[61, 1] = 4.0
    for ζ in (0.0, 0.1, 0.39)
        c = M3DC1Reader._collapse_zeta(coef, ζ)
        @test c[1, 1] ≈ 1 + 2ζ + 3ζ^2 + 4ζ^3
        d = M3DC1Reader._collapse_zeta(coef, ζ; dphi = true)
        @test d[1, 1] ≈ 2 + 6ζ + 12ζ^2
        # collapsed evaluation ≡ the full 3D Hermite evaluation
        @test M3DC1Reader.eval_axisym_at_local(vec(c), 0.2, 0.3) ≈
            M3DC1Reader.eval_at_local(vec(coef), 0.2, 0.3, ζ)
    end
    @test_throws ErrorException M3DC1Reader._collapse_zeta(zeros(20, 1), 0.1)
end

@testitem "write_ascot5 e2e (real C1.h5)" begin
    using HDF5
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping ascot5 e2e (no C1.h5 at $c1)"
    else
        file = M3DC1File(c1)
        ts = last(list_timeslices(file))
        out = tempname() * ".h5"
        write_ascot5(file, ts, out; nR = 60, nZ = 60, nphi = 16)

        h5open(out, "r") do f
            for parent in ("bfield", "plasma", "wall", "efield")
                @test haskey(f, parent)
                act = read_attribute(f[parent], "active")
                gname = only(keys(f[parent]))
                @test occursin(r"_\d{10}$", gname)
                @test endswith(gname, act)                 # active points at the group
                g = f[parent][gname]
                @test read_attribute(g, "date") isa AbstractString
            end
            g = f["bfield"][only(keys(f["bfield"]))]
            nR = Int(read(g["b_nr"])[1]);  nphi = Int(read(g["b_nphi"])[1])
            nZ = Int(read(g["b_nz"])[1])
            @test (nR, nphi, nZ) == (60, 16, 60)
            br = read(g["br"]);  bphi = read(g["bphi"])
            @test size(br) == (nR, nphi, nZ)               # Julia view of disk (nz,nphi,nr)
            bphif = filter(isfinite, vec(bphi))
            @test count(isfinite, vec(bphi)) / length(bphi) > 0.5   # mesh covers most of bbox
            @test all(x -> 1.5 < abs(x) < 5.0, bphif)      # |Bφ| ~ b0·r0/R
            @test length(unique(sign.(bphif))) == 1
            # br is the PERTURBATION: its scale is δB, a few % of |B| at most
            brf = filter(isfinite, vec(br))
            @test maximum(abs, brf) < 0.2 * maximum(abs, bphif)
            # ψ endpoints: per-radian Vs/m, axis inside the grid
            psi0 = read(g["psi0"])[1];  psi1 = read(g["psi1"])[1]
            @test 0.05 < abs(psi1 - psi0) < 5.0
            @test read(g["b_rmin"])[1] < read(g["axisr"])[1] < read(g["b_rmax"])[1]

            g = f["plasma"][only(keys(f["plasma"]))]
            nrho = Int(read(g["nrho"])[1]);  nion = Int(read(g["nion"])[1])
            ne = vec(read(g["edensity"]));  te = vec(read(g["etemperature"]))
            @test length(ne) == nrho
            @test all(isfinite, ne) && all(>(0), ne)       # NaN-filled + floored
            @test all(isfinite, te) && all(>(0), te)
            @test size(read(g["idensity"])) == (nrho, nion)

            g = f["wall"][only(keys(f["wall"]))]
            wr = vec(read(g["r"]))
            @test Int(read(g["nelements"])[1]) == length(wr) > 20
            @test all(x -> 0.5 < x < 10.0, wr)

            # efield is now the total MHD E as E_3D (M3D-C1 stores E_R/E_PHI/E_Z)
            ekey = only(keys(f["efield"]))
            @test startswith(ekey, "E_3D")
            g = f["efield"][ekey]
            @test (Int(read(g["nr"])[1]), Int(read(g["nphi"])[1]), Int(read(g["nz"])[1])) == (60, 16, 60)
            er = read(g["er"])
            @test size(er) == (60, 16, 60)                 # disk (nz,nphi,nr)
            erf = filter(isfinite, vec(er))
            @test count(isfinite, vec(er)) / length(er) > 0.5   # mesh covers most of bbox
            @test maximum(abs, erf) < 1.0e7                # sane V/m magnitude (not blown up)
        end
        rm(out; force = true)
    end
end

# Integrated pipeline: export_imas(ascot5=true) writes one ASCOT5 input per slice
# reusing that slice's FSA, so the ASCOT5 plasma_1D background must be IDENTICAL
# to the IMAS core_profiles (not just close). Same real-C1.h5 gate as above.
@testitem "export_imas ascot5=true reuses FSA (plasma_1D == core_profiles)" begin
    using HDF5
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping ascot5 integration e2e (no C1.h5 at $c1)"
    else
        file = M3DC1File(c1)
        ts = last(list_timeslices(file))
        apath = joinpath(dirname(abspath(c1)), "ascot_input_$(lpad(ts, 3, '0')).h5")
        imas_out = tempname() * ".h5"
        preexisting = isfile(apath)          # never delete a user's real ascot file
        try
            export_imas(
                file, imas_out; slices = [ts], fsa_method = :cumulative,
                ascot5 = true, ascot5_nphi = 8, verbose = false
            )
            @test isfile(apath)              # ascot input named by slice index

            ine, ite = h5open(imas_out) do f
                g = f["core_profiles"]["profiles_1d"]["0"]   # single slice → index 0
                vec(read(g["electrons"]["density"])), vec(read(g["electrons"]["temperature"]))
            end
            ane, ate = h5open(apath) do f
                g = f["plasma"][only(keys(f["plasma"]))]
                vec(read(g["edensity"])), vec(read(g["etemperature"]))
            end
            @test length(ane) == length(ine)
            mne = isfinite.(ine) .& isfinite.(ane)
            @test count(mne) > 10
            # reused FSA ⇒ bit-identical where IMAS is finite (ascot only fills
            # NaNs + applies a positivity floor real profiles never hit)
            @test all(i -> ine[i] == ane[i], findall(mne))
            mte = isfinite.(ite) .& isfinite.(ate)
            @test all(i -> ite[i] == ate[i], findall(mte))
        finally
            rm(imas_out; force = true)
            preexisting || rm(apath; force = true)
        end
    end
end
