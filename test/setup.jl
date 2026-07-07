# Shared @testsnippets for the M3DC1Reader test suite.
#
# TestItemRunner auto-discovers these (static AST scan) and evaluates each into a
# @testitem's isolated module when that item opts in via `setup = [Name]`.
# `using M3DC1Reader` and `using Test` are auto-injected into every @testitem, so
# only the extra imports / fixtures live here.

using TestItemRunner

# ── LcfsFixture ──────────────────────────────────────────────────────
# One synthetic field with BOTH an axis and a saddle, for the find_lcfs /
# eval_axisym_at tests. Element a=b=c=2, θ=0, origin (0,-1) → R=ξ+2, Z=η-1,
# η∈[0,2] ⇒ Z∈[-1,1]; ψ = ξ² + (η-½)² - (η-½)³:
#   O-point (min)    at local (0, 1/2) = global (2, -1/2), ψ0 = 0
#   X-point (saddle) at local (0, 7/6) = global (2, +1/6), ψx = 4/27
@testsnippet LcfsFixture begin
    ψsad = 4 / 27
    lelems = reshape([2.0, 2.0, 2.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    lc = zeros(20)
    lc[1] = 0.375;  lc[3] = -1.75;  lc[4] = 1.0;  lc[6] = 2.5;  lc[10] = -1.0
    lcoef = reshape(lc, 20, 1)
end

# ── DeltaBFixture ────────────────────────────────────────────────────
# A plane-stacked mini-mesh with one reference element per plane. Geometry:
# a=b=c=1, θ=0, origin (x,z)=(2,0) → local ξ = R-3, η = Z. Poloidal term 3 is
# ξ⁰η¹, so coef[3]=α gives ψ = α·η ⇒ ∂ψ/∂Z = α exactly; term 1 is the constant.
@testsnippet DeltaBFixture begin
    const _EP1 = reshape([1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)

    function _stacked_coefs(nplanes, setter!)
        A = zeros(80, nplanes)
        for p in 1:nplanes
            setter!(view(A, :, p), 2π * (p - 1) / nplanes)
        end
        return A
    end
end

# ── MinC1 ────────────────────────────────────────────────────────────
# Build a minimal valid synthetic C1.h5 for the export_imas helper tests. The
# M3DC1File constructor only needs one `time_NNN` group holding a `mesh` subgroup
# (an `elements` (10×nelms) dataset + `nplanes`/`period`/`phi` attrs) plus the
# root normalization attrs `b0_norm`/`n0_norm`/`l0_norm`/`ion_mass`. `root_attrs`
# / `fields` / `scalars` / `pellet` add optional extras. The slice stores
# ntimestep=1 (≠ 0 on purpose, so per-slice trace sampling hits entry [2],
# catching off-by-one indexing).
@testsnippet MinC1 begin
    using HDF5

    function _make_min_c1(
            path; root_attrs = Dict{String, Any}(),
            fields = Dict{String, Matrix{Float64}}(),
            scalars = Dict{String, Vector{Float64}}(),
            pellet = Dict{String, Matrix{Float64}}()
        )
        h5open(path, "w") do f
            # root normalization attrs required by the constructor
            attrs(f)["b0_norm"] = 1.0e4      # Gauss  → 1 T  (SI mag-field factor = 1.0)
            attrs(f)["n0_norm"] = 1.0e14     # cm^-3
            attrs(f)["l0_norm"] = 100.0    # cm     → 1 m  (SI length factor = 1.0)
            attrs(f)["ion_mass"] = 2.0      # mi/mp
            for (k, v) in root_attrs
                attrs(f)[k] = v
            end
            g = create_group(f, "time_001")
            attrs(g)["ntimestep"] = 1
            m = create_group(g, "mesh")
            # one reference element a=b=c=1, θ=0, origin (0,0); cols 7..10 padding
            m["elements"] = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
            attrs(m)["nplanes"] = 1
            attrs(m)["period"] = 2π
            attrs(m)["phi"] = [0.0]
            if !isempty(fields)
                fg = create_group(g, "fields")
                for (k, v) in fields
                    fg[k] = v
                end
            end
            if !isempty(scalars)
                sg = create_group(f, "scalars")
                for (k, v) in scalars
                    sg[k] = v
                end
            end
            if !isempty(pellet)
                pg = create_group(f, "pellet")
                for (k, v) in pellet
                    pg[k] = v
                end
            end
        end
        return path
    end
end
