@testitem "export_imas end-to-end" begin
    using HDF5
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping export_imas e2e (no C1.h5 at $c1)"
    else
        file = M3DC1File(c1)
        out = tempname() * ".h5"
        slices = list_timeslices(file)
        sample = slices[[1, 1 + length(slices) ÷ 2, length(slices)]] |> unique
        export_imas(file, out; slices = sample, nbins = 60, ngrid = 80, pulse = 123456)

        h5open(out, "r") do f
            @test haskey(f, "core_profiles")
            @test haskey(f, "equilibrium")
            @test read(f["core_profiles/ids_properties/homogeneous_time"]) === Int64(1)
            @test length(read(f["core_profiles/time"])) == length(sample)
            te = read(f["core_profiles/profiles_1d/0/electrons/temperature"])
            @test length(te) == 60
            @test maximum(filter(isfinite, te)) > 0          # peaked, positive Te
            d = f["equilibrium/time_slice/0/profiles_2d/0/psi"]
            @test ndims(d) == 2
            @test reverse(size(d)) == (80, 80)               # (nR, nZ) for h5py
            @test any(isfinite, read(d))
            # axisymmetric B maps + ⟨|B|⟩ (this device: b0=2.5 T at rzero=3 m)
            bphi = read(f["equilibrium/time_slice/0/profiles_2d/0/b_field_tor"])
            bfin = filter(isfinite, vec(bphi))
            @test !isempty(bfin)
            @test all(x -> 1.5 < abs(x) < 5.0, bfin)         # |Bφ| ~ b0·r0/R
            @test length(unique(sign.(bfin))) == 1            # single-signed Bφ
            bavg = read(f["core_profiles/profiles_1d/0/custom/b_field_torus_average"])
            bavg_f = filter(isfinite, bavg)
            @test !isempty(bavg_f)
            @test all(x -> 1.5 < x < 4.0, bavg_f)             # ⟨|B|⟩ around 2.5 T
            @test haskey(f, "equilibrium/time_slice/0/profiles_2d/0/phi")
            # δB/B fluctuation profile (√2·RMS mode-amplitude normalization,
            # docs/deltab_over_b.md): pre-injection slice ~ axisymmetric,
            # the disrupting slice carries a %-level stochastic field
            db_first = filter(
                isfinite,
                read(f["core_profiles/profiles_1d/0/custom/deltab_over_b"])
            )
            db_last = filter(
                isfinite,
                read(f["core_profiles/profiles_1d/$(length(sample) - 1)/custom/deltab_over_b"])
            )
            @test !isempty(db_last)
            @test all(x -> 0 <= x < 0.5, db_last)
            @test sum(db_first) / length(db_first) < 0.01
            @test sum(db_last) / length(db_last) >= sum(db_first) / length(db_first)
            # 0D globals: any real run carries these scalars/* traces
            @test read(f["summary/ids_properties/homogeneous_time"]) === Int64(1)
            ip = read(f["summary/global_quantities/ip/value"])
            @test length(ip) == length(sample)
            @test all(x -> 1.0e4 < abs(x) < 5.0e7, ip)           # sane tokamak Ip [A]
            wth = read(f["summary/global_quantities/energy_thermal/value"])
            @test all(x -> 0 < x < 1.0e9, wth)                 # thermal energy [J]
            @test read(f["equilibrium/time_slice/0/global_quantities/ip"]) == ip[1]
            if haskey(f, "disruption/global_quantities/power_ohm")
                vl = read(f["summary/global_quantities/v_loop/value"])
                @test read(f["disruption/global_quantities/power_ohm"]) ≈ ip .* vl
            end
            # COCOS-11 invariants (default export) — the DB's own validator rule
            # sign(q)=sign(Ip·B0), Ampère-consistent ψ orientation, total flux
            psia = read(f["equilibrium/time_slice/0/global_quantities/psi_axis"])
            psib = read(f["equilibrium/time_slice/0/global_quantities/psi_boundary"])
            @test sign(psib - psia) == sign(ip[1])
            @test 3.0 < abs(psib - psia) < 15.0       # 2π×(~1.1 Wb/rad) — not per-radian
            b0v = read(f["equilibrium/vacuum_toroidal_field/b0"])
            qcp = filter(isfinite, read(f["core_profiles/profiles_1d/0/q"]))
            @test !isempty(qcp)
            @test sign(qcp[end ÷ 2 + 1]) == sign(ip[1] * b0v[1])
            @test b0v[1] < 0                          # signed like F (this device)
            # wall IDS: mesh boundary as the limiter outline (metres)
            wr = read(f["wall/description_2d/0/limiter/unit/0/outline/r"])
            wz = read(f["wall/description_2d/0/limiter/unit/0/outline/z"])
            @test length(wr) == length(wz) > 20
            @test all(x -> 1.0 < x < 5.0, wr)          # sane machine radii
            @test read(f["wall/description_2d/0/limiter/type/name"]) == "first_wall"
            # pellets IDS: 30 SPI shards; ablation accumulates after injection
            @test haskey(f, "pellets/time_slice/0/pellet/29/path_profiles/ablation_rate")
            abl2 = sum(
                read(f["pellets/time_slice/$(length(sample) - 1)/pellet/$(ip0)/path_profiles/ablated_particles"])[1]
                    for ip0 in 0:29
            )
            @test abl2 > 1.0e19                          # ~1e21-1e22 Ne atoms injected
            # NIMROD SPI two-species layout: species.0 = D carrier, species.1 = impurity
            @test read(f["pellets/time_slice/0/pellet/0/species/0/label"]) == "D"
            @test read(f["pellets/time_slice/0/pellet/0/species/1/label"]) == "Ne"
            fr0 = read(f["pellets/time_slice/0/pellet/0/species/0/fraction"])
            fr1 = read(f["pellets/time_slice/0/pellet/0/species/1/fraction"])
            @test 0.0 <= fr0 <= 1.0
            @test fr0 + fr1 ≈ 1.0
            # recomputed X-point(s) recorded on the boundary IDS (diverted case)
            @test haskey(f, "equilibrium/time_slice/0/boundary/x_point/0/r")
            xr = read(f["equilibrium/time_slice/0/boundary/x_point/0/r"])
            @test 1.0 < xr < 5.0
            @test read(f["dataset_description/data_entry/pulse"]) === Int64(123456)
        end
        rm(out; force = true)
    end
end
