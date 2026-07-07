@testitem "write_omas_h5" begin
    using HDF5
    tmp = tempname() * ".h5"
    ir = Dict{String, Any}(
        "core_profiles.ids_properties.homogeneous_time" => 1,
        "core_profiles.ids_properties.source" => "M3D-C1",
        "core_profiles.ids_properties.some_flag" => true,
        "core_profiles.time" => [0.0, 1.5, 3.0],
        "core_profiles.profiles_1d.0.electrons.temperature" => [10.0, 5.0, 1.0],
        # intended numpy shape (nR=5, nZ=3):
        "equilibrium.time_slice.0.profiles_2d.0.psi" => reshape(collect(1.0:15.0), 5, 3),
    )
    write_omas_h5(tmp, ir)
    h5open(tmp, "r") do f
        @test read(f["core_profiles/ids_properties/homogeneous_time"]) === Int64(1)
        @test read(f["core_profiles/ids_properties/source"]) == "M3D-C1"
        @test read(f["core_profiles/ids_properties/some_flag"]) === Int64(1)
        @test read(f["core_profiles/time"]) ≈ [0.0, 1.5, 3.0]
        @test read(f["core_profiles/profiles_1d/0/electrons/temperature"]) ≈ [10.0, 5.0, 1.0]
        d = f["equilibrium/time_slice/0/profiles_2d/0/psi"]
        # h5py/numpy shape = reverse of Julia size; must be (nR, nZ) = (5, 3)
        @test reverse(size(d)) == (5, 3)
        @test permutedims(read(d)) ≈ reshape(collect(1.0:15.0), 5, 3)
    end
    rm(tmp; force = true)
end

@testitem "D2 — unsupported value type" begin
    using HDF5
    tmp = tempname() * ".h5"
    # A Tuple is not a String/Integer/Real/AbstractArray{<:Real} → error() branch.
    @test_throws ErrorException write_omas_h5(tmp, Dict("a.b" => (1, 2)))
    # A Symbol is likewise unsupported.
    @test_throws ErrorException write_omas_h5(tmp, Dict("a.b" => :sym))
    # A nested Dict value is unsupported (only flat dotted keys are allowed).
    @test_throws ErrorException write_omas_h5(tmp, Dict("a.b" => Dict("c" => 1.0)))
    rm(tmp; force = true)
end

@testitem "A2 — path collision raises a clear error" begin
    using HDF5
    tmp = tempname() * ".h5"
    # "a.b" => leaf and "a.b.c" => leaf collide: either iteration order must throw.
    # Order 1: leaf "a.b" first, then "a.b.c" tries to descend through the leaf.
    # Order 2: group "a.b" first (from "a.b.c"), then "a.b" leaf hits existing group.
    @test_throws ErrorException write_omas_h5(tmp, Dict("a.b" => 1.0, "a.b.c" => 2.0))
    # Confirm the raised message mentions the collision (robust to iteration order).
    err = try
        write_omas_h5(tmp, Dict("a.b" => 1.0, "a.b.c" => 2.0))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("collision", err.msg)
    rm(tmp; force = true)
end

@testitem "D7 — N-D (3D) transpose round-trip" begin
    using HDF5
    tmp = tempname() * ".h5"
    orig = reshape(collect(1.0:24.0), 2, 3, 4)
    write_omas_h5(tmp, Dict("equilibrium.time_slice.0.profiles_3d.0.b_field_r" => orig))
    h5open(tmp, "r") do f
        d = f["equilibrium/time_slice/0/profiles_3d/0/b_field_r"]
        # h5py/numpy shape = reverse of Julia size; must be (2, 3, 4).
        @test reverse(size(d)) == (2, 3, 4)
        @test permutedims(read(d), (3, 2, 1)) == orig
    end
    rm(tmp; force = true)
end
