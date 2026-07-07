# End-to-end tests on a real C1.h5 if present (env M3DC1_TEST_FILE, else the
# default scratch path) — mirrors the ad-hoc validations developed under
# ~/_neo_test (verify_magaxis.jl, etc.). Each item self-skips when no file
# exists. A few slices spanning the run (start / mid / late) are sampled.

@testitem "C1.h5 file metadata" begin
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping C1.h5 metadata test (no file at $c1)"
    else
        file = M3DC1File(c1)
        ts_list = list_timeslices(file)
        @test file.nplanes ≥ 1
        @test file.npp * file.nplanes == size(file.elems, 2)
        @test !isempty(ts_list)
        @test te_to_eV_factor(file) > 0
    end
end

@testitem "C1.h5 plane-1 magaxis reproduces stored axis" begin
    # Canonical verify_magaxis.jl result: Newton on the PLANE-1 (φ=0) ψ, seeded
    # from the stored (xmag, zmag), reproduces M3D-C1's stored (xmag, zmag, psi0)
    # to ~machine precision in 1–few iterations, because psi0 IS the plane-1
    # magaxis value.
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping C1.h5 plane-1 magaxis test (no file at $c1)"
    else
        file = M3DC1File(c1)
        ep = elems_plane(file)
        ts_list = list_timeslices(file)
        sample_ts = unique(
            clamp.(
                [ts_list[1], ts_list[1 + length(ts_list) ÷ 2], ts_list[end]],
                ts_list[1], ts_list[end]
            )
        )
        @testset "plane-1 magaxis (ts=$ts)" for ts in sample_ts
            slice = read_timeslice(file, ts; fields = (:psi,))
            psi_p1 = view(slice.fields[:psi], :, 1:file.npp)
            ax = find_axis_newton(psi_p1, ep, slice.xmag, slice.zmag)
            @test ax.converged
            @test ax.n_iter ≤ 5
            @test ax.R ≈ slice.xmag     atol = 1.0e-4
            @test ax.Z ≈ slice.zmag     atol = 1.0e-4
            @test ax.ψ ≈ slice.psi_axis atol = 1.0e-4
            @test ax.grad_norm < 1.0e-6
        end
    end
end

@testitem "C1.h5 read_timeslice optional + rows_of" begin
    # read_timeslice single-open batch (`optional` + `rows_of`): reading ψ plus
    # optional fields in one open must produce byte-for-byte the same arrays as
    # the old per-field `read_field` loop, skip absent optionals, and honour the
    # per-field row range.
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping C1.h5 read_timeslice optional test (no file at $c1)"
    else
        file = M3DC1File(c1)
        ts_list = list_timeslices(file)
        sample_ts = unique(
            clamp.(
                [ts_list[1], ts_list[1 + length(ts_list) ÷ 2], ts_list[end]],
                ts_list[1], ts_list[end]
            )
        )
        @testset "read_timeslice optional + rows_of (ts=$ts)" for ts in sample_ts
            batch = read_timeslice(
                file, ts; fields = (:psi,),
                optional = (:te, :I, :nonexistent_field_xyz),
                rows = 1:20,
                rows_of = fld -> fld === :psi ? (1:40) : (1:20)
            )
            # absent optional silently skipped; present ones read
            @test !haskey(batch.fields, :nonexistent_field_xyz)
            @test haskey(batch.fields, :te) && haskey(batch.fields, :I)
            # rows_of honoured per field: ψ got 1:40, the rest 1:20
            @test size(batch.fields[:psi], 1) == 40
            @test size(batch.fields[:te], 1) == 20
            @test size(batch.fields[:I], 1) == 20
            # bit-identical to independent per-field reads
            @test batch.fields[:psi] == read_field(file, ts, :psi; rows = 1:40)
            @test batch.fields[:te] == read_field(file, ts, :te; rows = 1:20)
            @test batch.fields[:I] == read_field(file, ts, :I; rows = 1:20)
        end
    end
end

@testitem "C1.h5 read_timeslice! in-place + buffer reuse" begin
    # read_timeslice! (in-place): the low-level h5d_read path must yield
    # byte-for-byte the same fields as the allocating read_timeslice, reuse the
    # same buffer objects across calls, and drop a field that vanishes.
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping C1.h5 read_timeslice! test (no file at $c1)"
    else
        file = M3DC1File(c1)
        ts_list = list_timeslices(file)
        sample_ts = unique(
            clamp.(
                [ts_list[1], ts_list[1 + length(ts_list) ÷ 2], ts_list[end]],
                ts_list[1], ts_list[end]
            )
        )
        @testset "read_timeslice! in-place + buffer reuse (ts=$ts)" for ts in sample_ts
            opt = (:te, :ne, :I)
            ref = read_timeslice(file, ts; fields = (:psi,), optional = opt, rows = 1:20)
            fbuf = Dict{Symbol, Matrix{Float64}}()
            sbuf = Dict{Symbol, Matrix{Float32}}()
            s1 = read_timeslice!(fbuf, sbuf, file, ts; required = (:psi,), optional = opt, rows = 1:20)
            @test s1.fields === fbuf
            @test keys(s1.fields) == keys(ref.fields)
            for k in keys(ref.fields)
                @test s1.fields[k] == ref.fields[k]      # bit-identical (Float32→Float64)
            end
            @test s1.psi_axis == ref.psi_axis && s1.time == ref.time
            # per-field row range honoured through rows_of
            @test size(
                read_timeslice!(
                    fbuf, sbuf, file, ts; required = (:psi,),
                    rows_of = fld -> 1:40
                ).fields[:psi], 1
            ) == 40
            # same buffer objects reused on a repeat read (no realloc)
            ids = Dict(k => objectid(v) for (k, v) in fbuf)
            read_timeslice!(fbuf, sbuf, file, ts; required = (:psi,), optional = opt, rows = 1:20)
            @test all(objectid(fbuf[k]) == ids[k] for k in keys(ids) if haskey(fbuf, k))
            # a field absent this slice is dropped from the reused buffer
            read_timeslice!(
                fbuf, sbuf, file, ts; required = (:psi,),
                optional = (:nonexistent_xyz,), rows = 1:20
            )
            @test collect(keys(fbuf)) == [:psi]
        end
    end
end

@testitem "C1.h5 n=0 magaxis + full reduction" begin
    # n=0 (toroidally-averaged) axis: converges, and sits within a few % of |Δψ|
    # of the plane-1 stored value (they differ by 3D modes). Followed by a full
    # 2D → 1D Te(ρ_pol) reduction sanity pass.
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping C1.h5 n=0 magaxis test (no file at $c1)"
    else
        file = M3DC1File(c1)
        ep = elems_plane(file)
        ts_list = list_timeslices(file)
        sample_ts = unique(
            clamp.(
                [ts_list[1], ts_list[1 + length(ts_list) ÷ 2], ts_list[end]],
                ts_list[1], ts_list[end]
            )
        )
        @testset "n=0 magaxis + full reduction (ts=$ts)" for ts in sample_ts
            slice = read_timeslice(file, ts; fields = (:psi, :te))
            psi_axisym_coef = average_toroidal_axisymmetric(slice.fields[:psi], file.nplanes)
            te_ax = average_toroidal_axisymmetric(
                slice.fields[:te] .* te_to_eV_factor(file),
                file.nplanes
            )
            ax = find_axis_newton(psi_axisym_coef, ep, slice.xmag, slice.zmag)
            @test ax.converged
            Δψ = slice.psi_lcfs - slice.psi_axis
            @test abs(ax.ψ - slice.psi_axis) < 0.05 * abs(Δψ)

            # 2D evaluation on an R-Z grid
            Rg = collect(range(extrema(ep[5, :])..., length = 80))
            Zg = collect(range(extrema(ep[6, :])..., length = 80))
            id_map = build_grid_to_element_map(Rg, Zg, ep)
            @test count(>(0), id_map) > 0
            psi_rz = interpolate_axisym_to_grid(psi_axisym_coef, ep, Rg, Zg; id_map = id_map)
            te_2d = interpolate_axisym_to_grid(te_ax, ep, Rg, Zg; id_map = id_map)
            @test any(isfinite, psi_rz)
            # Te(eV) may have small quintic overshoots near steep edges; require
            # any undershoot to be a small fraction of the peak (sanity, not strict ≥0).
            te_fin = filter(isfinite, te_2d)
            @test minimum(te_fin) > -0.05 * maximum(te_fin)

            # 1D Te(ρ_pol): most bins filled, peak Te is a sane positive value
            ψn = psi_to_psi_norm.(vec(psi_rz), ax.ψ, slice.psi_lcfs)
            ρpol = psi_n_to_rho_pol.(ψn)
            Te1d = vec(te_2d)
            m = isfinite.(ρpol) .& isfinite.(Te1d)
            prof = reduce_1d_psi_func(
                ρpol[m], Te1d[m];
                n_bins = 60, psi_range = (0.0, 1.2), adj = :linear
            )
            nfilled = count(isfinite, prof.func_bin)
            @test nfilled ≥ 45                       # ≥ 75% of bins have data
            @test maximum(filter(isfinite, prof.func_bin)) > 0
        end
    end
end
