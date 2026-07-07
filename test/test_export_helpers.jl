# Direct coverage for the export_imas helper functions (_impurity_species,
# _kprad_z, _with_vacuum_field, _try_field). These are otherwise only touched by
# the file-gated end-to-end test, which is skipped unless a real C1.h5 exists.
#
# A minimal synthetic C1.h5 is cheap to build: the M3DC1File constructor only
# needs one `time_NNN` group holding a `mesh` subgroup (an `elements` (10×nelms)
# dataset + `nplanes`/`period`/`phi` attrs) plus the root normalization attrs
# `b0_norm`/`n0_norm`/`l0_norm`/`ion_mass`. The helpers re-open `file.path` and
# read root attributes (rzero/bzero, ikprad/kprad_z) and field datasets, so we
# write per-case files and toggle just those attrs/fields.

# The synthetic C1.h5 builder `_make_min_c1` lives in test/setup.jl
# (@testsnippet MinC1); items that need it opt in via `setup = [MinC1]`.

@testitem "_impurity_species table + fallback" begin
    # tabulated species (label, atomic mass [amu])
    @test M3DC1Reader._impurity_species(2) == ("He", 4.0026)
    @test M3DC1Reader._impurity_species(6) == ("C", 12.011)
    @test M3DC1Reader._impurity_species(10) == ("Ne", 20.1797)
    @test M3DC1Reader._impurity_species(18) == ("Ar", 39.948)
    # also the remaining tabulated entries for completeness
    @test M3DC1Reader._impurity_species(1) == ("H", 1.008)
    @test M3DC1Reader._impurity_species(4) == ("Be", 9.0122)
    @test M3DC1Reader._impurity_species(5) == ("B", 10.811)

    # fallback for Z not in the table: ("imp", 2·max(z,1))
    @test M3DC1Reader._impurity_species(0) == ("imp", 2.0)    # max(0,1)=1 → 2.0
    @test M3DC1Reader._impurity_species(3) == ("imp", 6.0)
    @test M3DC1Reader._impurity_species(7) == ("imp", 14.0)
end

@testitem "_kprad_z from root attrs" setup = [MinC1] begin
    # ikprad=0 → KPRAD inactive → -1 (kprad_z ignored even if present)
    p_off = tempname() * ".h5"
    _make_min_c1(p_off; root_attrs = Dict{String, Any}("ikprad" => 0, "kprad_z" => 10))
    file_off = M3DC1File(p_off)
    @test M3DC1Reader._kprad_z(file_off) == -1

    # ikprad=1 and kprad_z=10 → 10
    p_on = tempname() * ".h5"
    _make_min_c1(p_on; root_attrs = Dict{String, Any}("ikprad" => 1, "kprad_z" => 10))
    file_on = M3DC1File(p_on)
    @test M3DC1Reader._kprad_z(file_on) == 10

    # attrs absent entirely → ikprad defaults to 0 → -1
    p_none = tempname() * ".h5"
    _make_min_c1(p_none)
    file_none = M3DC1File(p_none)
    @test M3DC1Reader._kprad_z(file_none) == -1

    rm(p_off; force = true)
    rm(p_on; force = true)
    rm(p_none; force = true)
end

@testitem "_with_vacuum_field meta r0/b0" setup = [MinC1] begin
    # b0_norm=1e4 G, l0_norm=100 cm chosen so the SI factors are exactly 1.0,
    # but assert against the computed factors so the test stays correct if reused.
    base = (;
        source = "s", provider = "p", code_name = "c", comment = "x",
        ion_label = "D+", z_ion = 1.0, ion_a = 2.0,
        imp_label = "imp", imp_z = 0, imp_a = 2.0,
        recompute_ne = false, r0 = NaN, b0 = NaN,
    )

    # attrs present → r0/b0 are rzero·(SI length factor), bzero·(SI mag-field factor)
    p_vac = tempname() * ".h5"
    _make_min_c1(p_vac; root_attrs = Dict{String, Any}("rzero" => 1.7, "bzero" => 2.3))
    file_vac = M3DC1File(p_vac)
    norm = normalization(file_vac)
    ulen = unit_factor(norm, :length; system = :si)
    ub = unit_factor(norm, :magnetic_field; system = :si)
    meta_vac = M3DC1Reader._with_vacuum_field(file_vac, norm, base)
    @test meta_vac.r0 ≈ 1.7 * ulen
    @test meta_vac.b0 ≈ 2.3 * ub

    # attrs absent → r0/b0 stay NaN
    p_novac = tempname() * ".h5"
    _make_min_c1(p_novac)
    file_novac = M3DC1File(p_novac)
    meta_novac = M3DC1Reader._with_vacuum_field(file_novac, normalization(file_novac), base)
    @test isnan(meta_novac.r0)
    @test isnan(meta_novac.b0)

    rm(p_vac; force = true)
    rm(p_novac; force = true)
end

@testitem "scalar_names + read_scalar" setup = [MinC1] begin
    # traces long enough for ntimestep=1 (entry [2] is the sampled one)
    traces = Dict{String, Vector{Float64}}(
        "toroidal_current" => [0.5, 0.6, 0.7],
        "W_P" => [0.1, 0.12, 0.14],
        "loop_voltage" => [0.01, 0.02, 0.03]
    )
    p = tempname() * ".h5"
    _make_min_c1(p; scalars = traces)
    file = M3DC1File(p)

    @test scalar_names(file) == ["W_P", "loop_voltage", "toroidal_current"]
    @test read_scalar(file, "toroidal_current") == [0.5, 0.6, 0.7]      # full trace
    @test read_scalar(file, "toroidal_current", 1) == 0.6               # ntimestep=1 → [2]
    @test read_scalar(file, "W_P", 1) == 0.12
    # unknown scalar / missing group are informative errors
    @test_throws ErrorException read_scalar(file, "no_such_scalar")
    @test_throws ErrorException read_scalar(file, "no_such_scalar", 1)

    p_none = tempname() * ".h5"
    _make_min_c1(p_none)                       # no scalars group at all
    file_none = M3DC1File(p_none)
    @test scalar_names(file_none) == String[]
    @test_throws ErrorException read_scalar(file_none, "toroidal_current")

    rm(p; force = true)
    rm(p_none; force = true)
end

@testitem "read_pellets: orientation + absent group" setup = [MinC1] begin
    # 2 pellets × 3 steps, written TRANSPOSED (steps × pellets, the raw HDF5
    # C-order view) — read_pellets must use the npellets attr to fix it
    rate_t = [1.0 10.0; 2.0 20.0; 3.0 30.0]        # (3 steps, 2 pellets)
    p = tempname() * ".h5"
    _make_min_c1(
        p; root_attrs = Dict{String, Any}("npellets" => 2),
        pellet = Dict{String, Matrix{Float64}}(
            "pellet_rate" => rate_t, "r_p" => fill(0.1, 3, 2)
        )
    )
    file = M3DC1File(p)
    pel = read_pellets(file)
    @test pel.npellets == 2
    @test size(pel.rate) == (2, 3)                  # (npellets, nsteps)
    @test pel.rate == [1.0 2.0 3.0; 10.0 20.0 30.0]
    @test size(pel.r_p) == (2, 3)
    @test pel.mix === nothing                       # dataset absent → nothing
    rm(p; force = true)

    # already (npellets, nsteps) → untouched
    p2 = tempname() * ".h5"
    _make_min_c1(
        p2; root_attrs = Dict{String, Any}("npellets" => 2),
        pellet = Dict{String, Matrix{Float64}}("pellet_rate" => permutedims(rate_t))
    )
    pel2 = read_pellets(M3DC1File(p2))
    @test pel2.rate == [1.0 2.0 3.0; 10.0 20.0 30.0]
    rm(p2; force = true)

    # no /pellet group at all → nothing
    p3 = tempname() * ".h5"
    _make_min_c1(p3)
    @test read_pellets(M3DC1File(p3)) === nothing
    rm(p3; force = true)
end

@testitem "_fprime_stored: version/numvar rule" setup = [MinC1] begin
    # fusion-io m3dc1_source.cpp:37-40 — fp dataset exists iff
    # version ≥ 38, or (version ≥ 35 and numvar > 1)
    cases = [
        (Dict{String, Any}("version" => 39), true),
        (Dict{String, Any}("version" => 38, "numvar" => 1), true),
        (Dict{String, Any}("version" => 36, "numvar" => 3), true),
        (Dict{String, Any}("version" => 36, "numvar" => 1), false),
        (Dict{String, Any}("version" => 30, "numvar" => 3), false),
        (Dict{String, Any}(), false),
    ]                 # attrs absent → legacy
    for (attrs_, want) in cases
        p = tempname() * ".h5"
        _make_min_c1(p; root_attrs = attrs_)
        @test M3DC1Reader._fprime_stored(M3DC1File(p)) == want
        rm(p; force = true)
    end
end

@testitem "_read_globals: SI conversion + absent traces + short trace" setup = [MinC1] begin
    traces = Dict{String, Vector{Float64}}(
        "toroidal_current" => [0.5, 0.6, 0.7],
        "W_P" => [0.1, 0.12, 0.14],
        "loop_voltage" => [0.01, 0.02, 0.03],
        "radiation" => [1.0, 2.0, 3.0]
    )
    p = tempname() * ".h5"
    _make_min_c1(p; scalars = traces)
    file = M3DC1File(p)
    norm = normalization(file)

    g = M3DC1Reader._read_globals(file, norm, [1, 2])   # sample steps 1 and 2
    @test g.ip ≈ [0.6, 0.7] .* unit_factor(norm, :current; system = :si)
    @test g.energy_thermal ≈ [0.12, 0.14] .* unit_factor(norm, :energy; system = :si)
    @test g.v_loop ≈ [0.02, 0.03] .* unit_factor(norm, :voltage; system = :si)
    @test g.power_radiated ≈ [2.0, 3.0] .* unit_factor(norm, :power; system = :si)

    # a step beyond the trace length → that trace comes back nothing
    g_short = M3DC1Reader._read_globals(file, norm, [1, 5])
    @test g_short.ip === nothing

    # file without any scalars → all nothing (export omits the summary IDS)
    p_none = tempname() * ".h5"
    _make_min_c1(p_none)
    file_none = M3DC1File(p_none)
    g_none = M3DC1Reader._read_globals(file_none, normalization(file_none), [1])
    @test all(k -> getproperty(g_none, k) === nothing, keys(g_none))

    rm(p; force = true)
    rm(p_none; force = true)
end

@testitem "_try_field success + swallow-to-nothing" setup = [MinC1] begin
    # write one real field (:psi) so a read succeeds; omit :te so it is absent.
    psi = zeros(80, 1); psi[1, 1] = 0.5
    p = tempname() * ".h5"
    _make_min_c1(p; fields = Dict{String, Matrix{Float64}}("psi" => psi))
    file = M3DC1File(p)

    # present field → default reads only the leading ζ=0 block (rows 1:20)
    got = M3DC1Reader._try_field(file, 1, :psi)
    @test got isa Matrix{Float64}
    @test got == psi[1:20, :]
    # explicit full read via the rows kwarg gives the whole 80×nelms matrix
    @test M3DC1Reader._try_field(file, 1, :psi; rows = Colon()) == psi
    @test M3DC1Reader._try_field(file, 1, :psi; rows = 1:40) == psi[1:40, :]

    # absent field → read_field throws internally, helper swallows it → nothing
    @test M3DC1Reader._try_field(file, 1, :te) === nothing

    rm(p; force = true)
end
