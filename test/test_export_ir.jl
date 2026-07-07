@testitem "assemble_ir" begin
    s1 = (;
        time = 0.0, rho = [0.0, 0.5, 1.0], psi1d = [0.0, 0.25, 1.0],
        te = [10.0, 5.0, 1.0], ne = [3.0, 2.0, 1.0],
        ti = [8.0, 4.0, 1.0], ni = [3.0, 2.0, 1.0],
        pressure = [100.0, 40.0, 5.0],
        psi_rz = reshape(collect(1.0:6.0), 3, 2), Rg = [1.0, 1.1, 1.2], Zg = [-0.1, 0.1],
        psi_axis = 0.0, psi_boundary = 1.0, R_axis = 1.05, Z_axis = 0.0,
    )
    # second slice with a missing field (ti = nothing) must be omitted, not errored
    s2 = merge(s1, (; time = 1.0, ti = nothing))
    s3 = merge(s1, (; time = 2.0, ni = nothing))
    meta = (;
        source = "M3D-C1", provider = "M3DC1Reader.jl", code_name = "M3DC1Reader",
        comment = "test", ion_label = "D+", z_ion = 1.0, ion_a = 2.0, r0 = 1.7, b0 = 2.0,
    )

    ir = assemble_ir([s1, s2, s3], meta)

    @test ir["core_profiles.ids_properties.homogeneous_time"] == 1
    @test ir["core_profiles.ids_properties.source"] == "M3D-C1"
    @test ir["core_profiles.code.name"] == "M3DC1Reader"
    @test ir["core_profiles.time"] == [0.0, 1.0, 2.0]
    @test ir["core_profiles.profiles_1d.0.grid.rho_pol_norm"] == [0.0, 0.5, 1.0]
    @test ir["core_profiles.profiles_1d.0.electrons.temperature"] == [10.0, 5.0, 1.0]
    @test ir["core_profiles.profiles_1d.0.ion.0.density"] == [3.0, 2.0, 1.0]
    @test ir["core_profiles.profiles_1d.0.ion.0.label"] == "D+"
    @test ir["core_profiles.profiles_1d.0.ion.0.element.0.a"] == 2.0
    @test ir["equilibrium.time_slice.0.profiles_1d.pressure"] == [100.0, 40.0, 5.0]
    @test ir["equilibrium.time_slice.0.profiles_2d.0.psi"] == reshape(collect(1.0:6.0), 3, 2)
    @test ir["equilibrium.time_slice.0.profiles_2d.0.grid.dim1"] == [1.0, 1.1, 1.2]
    @test ir["equilibrium.time_slice.0.profiles_2d.0.grid_type.index"] == 1
    @test ir["equilibrium.time_slice.0.global_quantities.psi_axis"] == 0.0
    @test ir["equilibrium.vacuum_toroidal_field.r0"] == 1.7
    @test ir["equilibrium.vacuum_toroidal_field.b0"] == [2.0, 2.0, 2.0]
    # slice 1 (index 1) had ti=nothing → that leaf must be absent
    @test !haskey(ir, "core_profiles.profiles_1d.1.ion.0.temperature")
    # but its other fields are present
    @test ir["core_profiles.profiles_1d.1.electrons.temperature"] == [10.0, 5.0, 1.0]
    # ni=nothing slice (index 2): ion[0] density AND its metadata block must be omitted
    @test !haskey(ir, "core_profiles.profiles_1d.2.ion.0.density")
    @test !haskey(ir, "core_profiles.profiles_1d.2.ion.0.label")
    # per-time-slice `time` is a scalar (vs the IDS-level vector)
    @test ir["equilibrium.time_slice.0.time"] == 0.0
    @test ir["equilibrium.time_slice.1.time"] == 1.0
end

@testitem "assemble_ir impurities + disruption" begin
    rho = [0.0, 0.5, 1.0]; psi1d = [0.0, 0.25, 1.0]
    s = (;
        time = 0.0, rho = rho, psi1d = psi1d,
        te = [10.0, 5.0, 1.0], ne = [3.0, 2.0, 1.0], ti = [8.0, 4.0, 1.0],
        ni = [3.0, 2.0, 1.0], pressure = [100.0, 40.0, 5.0],
        psi_rz = reshape(collect(1.0:6.0), 3, 2), Rg = [1.0, 1.1, 1.2], Zg = [-0.1, 0.1],
        psi_axis = 0.0, psi_boundary = 1.0, R_axis = 1.05, Z_axis = 0.0,
        imp_dens = Union{Nothing, Vector{Float64}}[[0.1, 0.05, 0.01], [0.2, 0.1, 0.02]],
        prad1d = [1.0e5, 4.0e4, 1.0e3],
    )
    meta = (;
        source = "M3D-C1", provider = "M3DC1Reader.jl", code_name = "M3DC1Reader",
        comment = "t", ion_label = "D+", z_ion = 1.0, ion_a = 2.0,
        imp_label = "Ne", imp_z = 1, imp_a = 20.18, recompute_ne = false, r0 = 1.7, b0 = 2.0,
    )

    ir = assemble_ir([s], meta)

    # impurity: neutral.0 (state 0) + ion.1 (state 1)
    @test ir["core_profiles.profiles_1d.0.neutral.0.density"] == [0.1, 0.05, 0.01]
    @test ir["core_profiles.profiles_1d.0.neutral.0.element.0.z_n"] == 1
    @test ir["core_profiles.profiles_1d.0.ion.1.density"] == [0.2, 0.1, 0.02]
    @test ir["core_profiles.profiles_1d.0.ion.1.label"] == "Ne1+"
    @test ir["core_profiles.profiles_1d.0.ion.1.z_ion"] == 1.0
    @test ir["core_profiles.profiles_1d.0.ion.1.element.0.z_n"] == 1
    # quasi-neutral off → native ne kept
    @test ir["core_profiles.profiles_1d.0.electrons.density"] == [3.0, 2.0, 1.0]
    # disruption IDS: Prad1D
    @test ir["disruption.ids_properties.homogeneous_time"] == 1
    @test ir["disruption.code.name"] == "M3DC1Reader"
    @test ir["disruption.time"] == [0.0]
    @test ir["disruption.profiles_1d.0.grid.rho_pol_norm"] == rho
    @test ir["disruption.profiles_1d.0.power_density_radiative_losses"] == [1.0e5, 4.0e4, 1.0e3]

    # with recompute_ne=true → nₑ = n_main + 1·n_imp1 = [3.2, 2.1, 1.02]
    meta_qn = merge(meta, (; recompute_ne = true))
    ir_qn = assemble_ir([s], meta_qn)
    @test ir_qn["core_profiles.profiles_1d.0.electrons.density"] ≈ [3.2, 2.1, 1.02]
end

@testitem "assemble_ir B-field + phi map passthrough" begin
    s = (;
        time = 0.0, rho = [0.0, 0.5, 1.0], psi1d = [0.0, 0.25, 1.0],
        te = [10.0, 5.0, 1.0], ne = [3.0, 2.0, 1.0],
        ti = [8.0, 4.0, 1.0], ni = [3.0, 2.0, 1.0],
        pressure = [100.0, 40.0, 5.0],
        psi_rz = reshape(collect(1.0:6.0), 3, 2), Rg = [1.0, 1.1, 1.2], Zg = [-0.1, 0.1],
        psi_axis = 0.0, psi_boundary = 1.0, R_axis = 1.05, Z_axis = 0.0,
        br_rz = fill(0.1, 3, 2), bz_rz = fill(0.2, 3, 2), bphi_rz = fill(2.0, 3, 2),
        phi_rz = fill(0.5, 3, 2), babs1d = [2.0, 2.1, 2.2],
    )
    meta = (;
        source = "M3D-C1", provider = "M3DC1Reader.jl", code_name = "M3DC1Reader",
        comment = "t", ion_label = "D+", z_ion = 1.0, ion_a = 2.0, r0 = 1.7, b0 = 2.0,
    )
    ir = assemble_ir([merge(s, (; x_points = [(1.4, -1.1), (1.5, 1.2)]))], meta)
    @test ir["equilibrium.time_slice.0.boundary.x_point.0.r"] == 1.4
    @test ir["equilibrium.time_slice.0.boundary.x_point.0.z"] == -1.1
    @test ir["equilibrium.time_slice.0.boundary.x_point.1.r"] == 1.5
    # no x_points key (old tuples) → no boundary paths
    ir_nox = assemble_ir([s], meta)
    @test !haskey(ir_nox, "equilibrium.time_slice.0.boundary.x_point.0.r")

    ir = assemble_ir([s], meta)
    @test ir["equilibrium.time_slice.0.profiles_2d.0.b_field_r"] == s.br_rz
    @test ir["equilibrium.time_slice.0.profiles_2d.0.b_field_z"] == s.bz_rz
    @test ir["equilibrium.time_slice.0.profiles_2d.0.b_field_tor"] == s.bphi_rz
    @test ir["equilibrium.time_slice.0.profiles_2d.0.phi"] == s.phi_rz
    @test ir["core_profiles.profiles_1d.0.custom.b_field_torus_average"] == s.babs1d

    # old-style tuples (no B keys) still assemble, with the paths absent
    s_old = Base.structdiff(s, NamedTuple{(:br_rz, :bz_rz, :bphi_rz, :phi_rz, :babs1d)})
    ir_old = assemble_ir([s_old], meta)
    @test !haskey(ir_old, "equilibrium.time_slice.0.profiles_2d.0.b_field_r")
    @test !haskey(ir_old, "core_profiles.profiles_1d.0.custom.b_field_torus_average")
end

@testitem "assemble_ir 0D globals (summary + equilibrium ip + disruption)" begin
    s1 = (;
        time = 0.0, rho = [0.0, 0.5, 1.0], psi1d = [0.0, 0.25, 1.0],
        te = [10.0, 5.0, 1.0], ne = [3.0, 2.0, 1.0],
        ti = [8.0, 4.0, 1.0], ni = [3.0, 2.0, 1.0],
        pressure = [100.0, 40.0, 5.0],
        psi_rz = reshape(collect(1.0:6.0), 3, 2), Rg = [1.0, 1.1, 1.2], Zg = [-0.1, 0.1],
        psi_axis = 0.0, psi_boundary = 1.0, R_axis = 1.05, Z_axis = 0.0,
    )
    s2 = merge(s1, (; time = 1.0))
    base_meta = (;
        source = "M3D-C1", provider = "M3DC1Reader.jl", code_name = "M3DC1Reader",
        comment = "t", ion_label = "D+", z_ion = 1.0, ion_a = 2.0, r0 = 1.7, b0 = 2.0,
    )

    # all four traces present → summary + per-slice eq ip + disruption globals
    gq = (;
        ip = [1.0e6, 0.8e6], energy_thermal = [2.0e5, 1.5e5],
        v_loop = [1.0, 30.0], power_radiated = [5.0e6, 2.0e8],
    )
    ir = assemble_ir([s1, s2], merge(base_meta, (; globals = gq)))
    @test ir["summary.ids_properties.homogeneous_time"] == 1
    @test ir["summary.time"] == [0.0, 1.0]
    @test ir["summary.global_quantities.ip.value"] == gq.ip
    @test ir["summary.global_quantities.energy_thermal.value"] == gq.energy_thermal
    @test ir["summary.global_quantities.v_loop.value"] == gq.v_loop
    @test ir["summary.global_quantities.power_radiated.value"] == gq.power_radiated
    @test ir["equilibrium.time_slice.0.global_quantities.ip"] == 1.0e6
    @test ir["equilibrium.time_slice.1.global_quantities.ip"] == 0.8e6
    # disruption globals appear even without prad1d profiles
    @test ir["disruption.ids_properties.homogeneous_time"] == 1
    @test ir["disruption.time"] == [0.0, 1.0]
    @test ir["disruption.global_quantities.power_radiated_total"] == gq.power_radiated
    @test ir["disruption.global_quantities.power_ohm"] ≈ [1.0e6, 2.4e7]   # Ip·V_loop
    @test !haskey(ir, "disruption.profiles_1d.0.grid.rho_pol_norm")       # no profiles

    # only ip present → summary written; no power_ohm/power_radiated_total,
    # and therefore no disruption IDS at all
    ir_ip = assemble_ir(
        [s1, s2], merge(
            base_meta,
            (;
                globals = (;
                    ip = [1.0e6, 0.8e6], energy_thermal = nothing,
                    v_loop = nothing, power_radiated = nothing,
                ),
            )
        )
    )
    @test ir_ip["summary.global_quantities.ip.value"] == [1.0e6, 0.8e6]
    @test !haskey(ir_ip, "summary.global_quantities.v_loop.value")
    @test !haskey(ir_ip, "disruption.ids_properties.homogeneous_time")

    # no globals key (old-style meta) → no summary, no eq ip (backward compat)
    ir_old = assemble_ir([s1, s2], base_meta)
    @test !haskey(ir_old, "summary.ids_properties.homogeneous_time")
    @test !haskey(ir_old, "equilibrium.time_slice.0.global_quantities.ip")

    # all-nothing globals → same as absent
    ir_nil = assemble_ir(
        [s1, s2], merge(
            base_meta,
            (;
                globals = (;
                    ip = nothing, energy_thermal = nothing,
                    v_loop = nothing, power_radiated = nothing,
                ),
            )
        )
    )
    @test !haskey(ir_nil, "summary.ids_properties.homogeneous_time")
end

@testitem "assemble_ir recompute_ne edge cases + vacuum-field omit" begin
    rho = [0.0, 0.5, 1.0]; psi1d = [0.0, 0.25, 1.0]
    base = (;
        time = 0.0, rho = rho, psi1d = psi1d,
        te = [10.0, 5.0, 1.0], ne = [3.0, 2.0, 1.0], ti = [8.0, 4.0, 1.0],
        ni = [3.0, 2.0, 1.0], pressure = [100.0, 40.0, 5.0],
        psi_rz = reshape(collect(1.0:6.0), 3, 2), Rg = [1.0, 1.1, 1.2], Zg = [-0.1, 0.1],
        psi_axis = 0.0, psi_boundary = 1.0, R_axis = 1.05, Z_axis = 0.0,
    )
    meta = (;
        source = "M3D-C1", provider = "M3DC1Reader.jl", code_name = "M3DC1Reader",
        comment = "t", ion_label = "D+", z_ion = 1.0, ion_a = 2.0,
        imp_label = "Ne", imp_z = 1, imp_a = 20.18, recompute_ne = true, r0 = 1.7, b0 = 2.0,
    )

    # (1) A1 regression: a NaN impurity bin must NOT corrupt ne.
    # state0 (neutral) + state1 (ion); state-1 profile has NaN in the middle bin.
    s_nan = merge(
        base, (;
            imp_dens =
                Union{Nothing, Vector{Float64}}[[0.1, 0.05, 0.01], [0.2, NaN, 0.02]],
        )
    )
    ir_nan = assemble_ir([s_nan], meta)
    ne_nan = ir_nan["core_profiles.profiles_1d.0.electrons.density"]
    # NaN bin contributes 0 → ne = ni + 1·[0.2, 0.0, 0.02]
    @test ne_nan == [3.2, 2.0, 1.02]
    @test all(isfinite, ne_nan)

    # (2) D5: recompute_ne=true with ni === nothing must not crash; native ne kept.
    s_noni = merge(
        base, (;
            ni = nothing, imp_dens =
                Union{Nothing, Vector{Float64}}[[0.1, 0.05, 0.01], [0.2, NaN, 0.02]],
        )
    )
    ir_noni = assemble_ir([s_noni], meta)   # must not crash when ni === nothing
    # quasi-neutral path skipped (guard requires s.ni !== nothing) → native ne preserved
    @test ir_noni["core_profiles.profiles_1d.0.electrons.density"] == s_noni.ne

    # (3) A10: NaN r0/b0 must be omitted from the vacuum_toroidal_field block.
    meta_nan_vac = merge(meta, (; r0 = NaN, b0 = NaN))
    ir_vac = assemble_ir([base], meta_nan_vac)
    @test !haskey(ir_vac, "equilibrium.vacuum_toroidal_field.r0")
    @test !haskey(ir_vac, "equilibrium.vacuum_toroidal_field.b0")
end
