# elements.jl — M3D-C1 reduced quintic Hermite element math (format-specific).
#
# Local-coordinate evaluation, analytic derivatives, element location, and the
# (R,Z)→element map. These encode the M3D-C1 C¹ reduced quintic Hermite basis
# and constitute the layer that would differ for another MHD code's backend.
#
# Element columns (HDF5 col-major, Julia 1-indexed rows):
#   1=a, 2=b, 3=c, 4=θ, 5=x, 6=z, 7=bound, 8=region, 9=d, 10=φ

# Reduced quintic Hermite poloidal monomial indices (0-indexed exponents).
# Term k contributes c_k * ξ^MI[k] * η^NI[k].
const MI = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 3, 2, 1, 0)
const NI = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 2, 3, 4, 5)

"""
    global_to_local(R, Z, x, z, b, θ) -> (ξ, η)

Map a global (R, Z) point into an element's local (ξ, η) coord system given
its origin (x, z), ξ-axis offset `b`, and rotation `θ`.
"""
@inline function global_to_local(R::Real, Z::Real, x::Real, z::Real, b::Real, θ::Real)
    co, sn = cos(θ), sin(θ)
    ξ = (R - x) * co + (Z - z) * sn - b
    η = -(R - x) * sn + (Z - z) * co
    return (ξ, η)
end

"""
    is_in_element_local(ξ, η, a, b, c; tol_factor=1e-4) -> Bool

Check whether (ξ, η) lies within the triangle (-b, 0)–(a, 0)–(0, c).
"""
@inline function is_in_element_local(
        ξ::Real, η::Real, a::Real, b::Real, c::Real;
        tol_factor::Real = 1.0e-4
    )
    small = (abs(a) + abs(b) + abs(c)) * tol_factor
    (η + small < 0.0)  && return false
    (η - small > c)    && return false
    x_lim = 1.0 - η / c
    (ξ + small < -b * x_lim) && return false
    (ξ - small > a * x_lim) && return false
    return true
end

"""
    eval_axisym_at_local(coef, ξ, η)

Evaluate the axisymmetric Hermite polynomial (ζ=0) at element-local (ξ, η)
using the first 20 coefficients. `coef` may be an 80-element vector
(`coef[1:20]` used) or a 20-element vector.
"""
@inline function eval_axisym_at_local(coef::AbstractVector, ξ::Real, η::Real)
    lpξ = (1.0, ξ, ξ^2, ξ^3, ξ^4, ξ^5)
    lpη = (1.0, η, η^2, η^3, η^4, η^5)
    s = 0.0
    @inbounds for k in 1:20
        s += Float64(coef[k]) * lpξ[MI[k] + 1] * lpη[NI[k] + 1]
    end
    return s
end

"""
    eval_axisym_at(coef_plane, elems_plane, R, Z) -> Float64

Evaluate the axisymmetric (ζ=0) field at the global point `(R, Z)`: locate the
containing element and evaluate its quintic polynomial there. Returns `NaN`
when `(R, Z)` is outside the mesh.
"""
function eval_axisym_at(
        coef_plane::AbstractMatrix, elems_plane::AbstractMatrix,
        R::Real, Z::Real
    )
    k = locate_element(R, Z, elems_plane)
    k == 0 && return NaN
    ξ, η = global_to_local(
        R, Z, Float64(elems_plane[5, k]), Float64(elems_plane[6, k]),
        Float64(elems_plane[2, k]), Float64(elems_plane[4, k])
    )
    return eval_axisym_at_local(view(coef_plane, :, k), ξ, η)
end

"""
    eval_at_local(coef, ξ, η, ζ)

Evaluate the full 3D Hermite polynomial at element-local (ξ, η, ζ) using all
80 coefficients. For axisymmetric data this gives the same value at any ζ
(since ζ¹, ζ², ζ³ layers are 0 in the averaged representation).
"""
@inline function eval_at_local(coef::AbstractVector, ξ::Real, η::Real, ζ::Real)
    lpξ = (1.0, ξ, ξ^2, ξ^3, ξ^4, ξ^5)
    lpη = (1.0, η, η^2, η^3, η^4, η^5)
    s = 0.0
    @inbounds for k in 1:20
        b_k = lpξ[MI[k] + 1] * lpη[NI[k] + 1]
        v_k = Float64(coef[k]) +
            Float64(coef[k + 20]) * ζ +
            Float64(coef[k + 40]) * ζ^2 +
            Float64(coef[k + 60]) * ζ^3
        s += v_k * b_k
    end
    return s
end

"""
    build_grid_to_element_map(R_grid, Z_grid, elems_plane) -> Matrix{Int}

Precompute the element id that contains each (R, Z) grid point, using a
bounding-box prune per element + triangle test. Returns an `Int` matrix of
shape `(length(R_grid), length(Z_grid))`, with `0` where the point is outside
the mesh.

# Arguments
- `R_grid, Z_grid`: rectilinear 1D coordinate vectors (sorted ascending).
- `elems_plane::AbstractMatrix`: shape `(10, npp)`, single plane mesh
  elements (use `elements[:, 1:npp]` from a 3D-stacked file).

# Notes
The element bounding box is computed from the three global vertices. Points
on shared edges between elements are assigned to the first element checked
that contains them (deterministic).
"""
function build_grid_to_element_map(
        R_grid::AbstractVector,
        Z_grid::AbstractVector,
        elems_plane::AbstractMatrix
    )
    nR, nZ = length(R_grid), length(Z_grid)
    npp = size(elems_plane, 2)
    id_map = zeros(Int, nR, nZ)

    a = view(elems_plane, 1, :);  b = view(elems_plane, 2, :);  c = view(elems_plane, 3, :)
    θ = view(elems_plane, 4, :);  x = view(elems_plane, 5, :);  z = view(elems_plane, 6, :)

    R_grid_arr = collect(R_grid);  Z_grid_arr = collect(Z_grid)

    @inbounds for k in 1:npp
        co, sn = cos(θ[k]), sin(θ[k])
        # Global vertex coordinates
        v1R = x[k];                       v1Z = z[k]
        v2R = x[k] + (a[k] + b[k]) * co;  v2Z = z[k] + (a[k] + b[k]) * sn
        v3R = x[k] + b[k] * co - c[k] * sn;  v3Z = z[k] + b[k] * sn + c[k] * co
        R_lo = min(v1R, v2R, v3R);  R_hi = max(v1R, v2R, v3R)
        Z_lo = min(v1Z, v2Z, v3Z);  Z_hi = max(v1Z, v2Z, v3Z)

        i_lo = searchsortedfirst(R_grid_arr, R_lo - 1.0e-6)
        i_hi = searchsortedlast(R_grid_arr, R_hi + 1.0e-6)
        j_lo = searchsortedfirst(Z_grid_arr, Z_lo - 1.0e-6)
        j_hi = searchsortedlast(Z_grid_arr, Z_hi + 1.0e-6)
        (i_lo > i_hi || j_lo > j_hi) && continue

        for j in j_lo:j_hi, i in i_lo:i_hi
            id_map[i, j] != 0 && continue
            ξ, η = global_to_local(
                R_grid_arr[i], Z_grid_arr[j],
                x[k], z[k], b[k], θ[k]
            )
            if is_in_element_local(ξ, η, a[k], b[k], c[k])
                id_map[i, j] = k
            end
        end
    end
    return id_map
end

"""
    locate_element(R, Z, elems_plane) -> Int

Linear search for the element index containing global point (R, Z) at one
toroidal plane. Returns 0 if (R, Z) is outside the mesh. O(npp); for large
meshes prefer using `build_grid_to_element_map` once and then a grid lookup.
"""
function locate_element(R::Real, Z::Real, elems_plane::AbstractMatrix)
    npp = size(elems_plane, 2)
    a = view(elems_plane, 1, :);  b = view(elems_plane, 2, :)
    c = view(elems_plane, 3, :);  θ = view(elems_plane, 4, :)
    x = view(elems_plane, 5, :);  z = view(elems_plane, 6, :)
    @inbounds for k in 1:npp
        ξ, η = global_to_local(R, Z, x[k], z[k], b[k], θ[k])
        if is_in_element_local(ξ, η, a[k], b[k], c[k])
            return k
        end
    end
    return 0
end

"""
    mesh_boundary_rz(elems_plane) -> (; R, Z)

Ordered outline of the single-plane mesh boundary — the M3D-C1 computational
wall (e.g. for the `wall` IDS limiter outline or an ASCOT5 wall). Returns the
loop vertices in order (last point not repeated), in mesh (normalized) units.

M3D-C1 tags each element's boundary edges directly: row 7 of the element
table is a bitmask (bit `k` set ⇒ triangle edge `k` lies on the domain
boundary). When present, that is the exact boundary — the flagged edges are
chained into the loop with a single vertex snap. For meshes without the flag
(e.g. synthetic test fixtures) it falls back to identifying boundary edges as
those used by exactly one element, with an adaptive snap-tolerance ladder
(adjacent elements reconstruct shared vertices only to ~1e-4·span).
"""
function mesh_boundary_rz(elems_plane::AbstractMatrix)
    xs = view(elems_plane, 5, :);  zs = view(elems_plane, 6, :)
    span = max(maximum(xs) - minimum(xs), maximum(zs) - minimum(zs), 1.0)

    # Preferred path: M3D-C1's own per-element boundary-edge flags (row 7).
    if size(elems_plane, 1) >= 7 && any(!iszero, view(elems_plane, 7, :))
        r = _boundary_from_flags(elems_plane, span * 1.0e-3)
        r.clean && return (; r.R, r.Z)
        @warn "mesh_boundary_rz: flagged boundary edges did not chain to one clean loop; falling back to edge counting"
    end

    # Fallback: boundary = edges used by exactly one element, snapped adaptively.
    tolfs = (1.0e-8, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
    result = nothing
    for (i, tolf) in enumerate(tolfs)
        # A non-manifold snap tolerance (some vertex on ≠2 boundary edges) is
        # rejected anyway, so skip its loop-chaining walk — at too-fine
        # tolerances nothing merges and that walk is quadratic. Force the walk
        # only on the last tolerance so the all-fail fallback still has a loop.
        r = _boundary_loop(elems_plane, span * tolf; force_walk = i == length(tolfs))
        result = r
        r.clean && return (; r.R, r.Z)
        i == length(tolfs) &&
            @warn "mesh_boundary_rz: boundary not a clean manifold at any snap tolerance; returning the longest loop" tolf
    end
    return (; result.R, result.Z)
end

# Reconstruct an element's three global triangle vertices from its stored
# (a, b, c, θ, x, z): local (−b,0), (a,0), (0,c) rotated by θ about (x, z).
@inline function _tri_verts(elems_plane::AbstractMatrix, k::Integer)
    a = Float64(elems_plane[1, k]);  b = Float64(elems_plane[2, k])
    c = Float64(elems_plane[3, k]);  θ = Float64(elems_plane[4, k])
    x0 = Float64(elems_plane[5, k]);  z0 = Float64(elems_plane[6, k])
    co = cos(θ);  sn = sin(θ)
    return (
        (x0, z0),
        (x0 + (a + b) * co, z0 + (a + b) * sn),
        (x0 + b * co - c * sn, z0 + b * sn + c * co),
    )
end

# Boundary from M3D-C1's row-7 edge bitmask: bit 1 → edge (v1,v2), bit 2 →
# (v2,v3), bit 4 → (v3,v1). Chains the flagged edges into an ordered loop.
function _boundary_from_flags(elems_plane::AbstractMatrix, tol::Float64)
    key(p) = (round(Int64, p[1] / tol), round(Int64, p[2] / tol))
    coords = Dict{NTuple{2, Int64}, NTuple{2, Float64}}()
    adj = Dict{NTuple{2, Int64}, Vector{NTuple{2, Int64}}}()
    nb = 0
    for k in axes(elems_plane, 2)
        t = Int(round(Float64(elems_plane[7, k])));  t == 0 && continue
        v1, v2, v3 = _tri_verts(elems_plane, k)
        for (bit, (p, q)) in ((1, (v1, v2)), (2, (v2, v3)), (4, (v3, v1)))
            (t & bit) == 0 && continue
            kp = key(p);  kq = key(q)
            coords[kp] = p;  coords[kq] = q
            push!(get!(() -> NTuple{2, Int64}[], adj, kp), kq)
            push!(get!(() -> NTuple{2, Int64}[], adj, kq), kp)
            nb += 1
        end
    end
    return _chain_boundary(adj, coords, nb)
end

function _boundary_loop(elems_plane::AbstractMatrix, tol::Float64; force_walk::Bool = false)
    key(p) = (round(Int64, p[1] / tol), round(Int64, p[2] / tol))
    edges = Dict{NTuple{2, NTuple{2, Int64}}, Int}()       # edge key → use count
    coords = Dict{NTuple{2, Int64}, NTuple{2, Float64}}()   # vertex key → coords
    for k in axes(elems_plane, 2)
        v1, v2, v3 = _tri_verts(elems_plane, k)
        for (p, q) in ((v1, v2), (v2, v3), (v3, v1))
            kp = key(p);  kq = key(q)
            coords[kp] = p;  coords[kq] = q
            ek = kp <= kq ? (kp, kq) : (kq, kp)
            edges[ek] = get(edges, ek, 0) + 1
        end
    end

    adj = Dict{NTuple{2, Int64}, Vector{NTuple{2, Int64}}}()
    nb = 0
    for (ek, n) in edges
        n == 1 || continue                                # boundary edge
        nb += 1
        push!(get!(() -> NTuple{2, Int64}[], adj, ek[1]), ek[2])
        push!(get!(() -> NTuple{2, Int64}[], adj, ek[2]), ek[1])
    end
    return _chain_boundary(adj, coords, nb; force_walk = force_walk)
end

# Chain an adjacency of boundary edges into an ordered loop. `clean` = the
# boundary is a single closed manifold (every vertex on exactly two edges).
function _chain_boundary(
        adj::Dict{NTuple{2, Int64}, Vector{NTuple{2, Int64}}},
        coords::Dict{NTuple{2, Int64}, NTuple{2, Float64}},
        nb::Int; force_walk::Bool = false
    )
    isempty(adj) && return (; R = Float64[], Z = Float64[], clean = false)
    manifold = all(v -> length(v) == 2, values(adj))
    # a non-manifold candidate is rejected; skip the (quadratic when degenerate)
    # chaining walk unless the caller forces it for the fallback outline
    (manifold || force_walk) ||
        return (; R = Float64[], Z = Float64[], clean = false)

    visited = Set{NTuple{2, Int64}}()
    best = NTuple{2, Int64}[]
    nloops = 0
    for start in keys(adj)
        start in visited && continue
        nloops += 1
        loop = [start];  push!(visited, start)
        prev = start;  cur = first(adj[start])
        closed = false
        for _ in 1:nb                                     # bounded walk
            if cur == start
                closed = true
                break
            end
            push!(loop, cur);  push!(visited, cur)
            nexts = adj[cur]
            nxt = nexts[1] == prev ? (length(nexts) > 1 ? nexts[2] : start) : nexts[1]
            prev, cur = cur, nxt
        end
        manifold &= closed
        length(loop) > length(best) && (best = loop)
    end
    return (;
        R = Float64[coords[k][1] for k in best],
        Z = Float64[coords[k][2] for k in best],
        clean = manifold && nloops == 1,
    )
end

"""
    mesh_zone_boundary_rz(elems_plane, zone_of_elem, zone) -> Vector

Ordered boundary loop(s) of the mesh elements whose region id `zone_of_elem[k]`
equals `zone`. M3D-C1 multi-region meshes (`imulti_region=1`) tag elements by
zone — `ZONE_PLASMA=1`, `ZONE_CONDUCTOR=2` (the resistive wall), `ZONE_VACUUM=3`
(see [`elem_zones`](@ref)); single-region runs are all plasma. A simply
connected zone yields one loop; an annular zone (the conductor between plasma
and vacuum) yields two — its inner and outer surfaces. Each loop is a
`(; R, Z)` NamedTuple in mesh units; returns `[]` when no element is in `zone`.

Boundary edges of the zone are those used by exactly one of its elements
(inter-zone interfaces included), snapped on an adaptive grid like
[`mesh_boundary_rz`](@ref)'s fallback.
"""
function mesh_zone_boundary_rz(
        elems_plane::AbstractMatrix,
        zone_of_elem::AbstractVector, zone::Integer
    )
    cols = findall(==(zone), zone_of_elem)
    isempty(cols) && return NamedTuple[]
    sub = view(elems_plane, :, cols)
    xs = view(sub, 5, :);  zs = view(sub, 6, :)
    span = max(maximum(xs) - minimum(xs), maximum(zs) - minimum(zs), 1.0)
    for tolf in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
        loops = _all_boundary_loops(sub, span * tolf)
        isempty(loops) && continue
        all(l -> l.closed, loops) && return [(; l.R, l.Z) for l in loops]
    end
    loops = _all_boundary_loops(sub, span * 1.0e-3)
    return [(; l.R, l.Z) for l in loops]
end

# All closed boundary loops (not just the longest) of a set of elements, as
# vectors of ordered vertices with a `closed` flag.
function _all_boundary_loops(elems_plane::AbstractMatrix, tol::Float64)
    key(p) = (round(Int64, p[1] / tol), round(Int64, p[2] / tol))
    edges = Dict{NTuple{2, NTuple{2, Int64}}, Int}()
    coords = Dict{NTuple{2, Int64}, NTuple{2, Float64}}()
    for k in axes(elems_plane, 2)
        v1, v2, v3 = _tri_verts(elems_plane, k)
        for (p, q) in ((v1, v2), (v2, v3), (v3, v1))
            kp = key(p);  kq = key(q)
            coords[kp] = p;  coords[kq] = q
            ek = kp <= kq ? (kp, kq) : (kq, kp)
            edges[ek] = get(edges, ek, 0) + 1
        end
    end
    adj = Dict{NTuple{2, Int64}, Vector{NTuple{2, Int64}}}()
    nb = 0
    for (ek, n) in edges
        n == 1 || continue
        nb += 1
        push!(get!(() -> NTuple{2, Int64}[], adj, ek[1]), ek[2])
        push!(get!(() -> NTuple{2, Int64}[], adj, ek[2]), ek[1])
    end
    isempty(adj) && return NamedTuple[]
    all(v -> length(v) == 2, values(adj)) || return NamedTuple[]   # non-manifold: give up
    visited = Set{NTuple{2, Int64}}()
    out = NamedTuple[]
    for start in keys(adj)
        start in visited && continue
        loop = [start];  push!(visited, start)
        prev = start;  cur = first(adj[start]);  closed = false
        for _ in 1:nb
            if cur == start
                closed = true
                break
            end
            push!(loop, cur);  push!(visited, cur)
            nexts = adj[cur]
            nxt = nexts[1] == prev ? (length(nexts) > 1 ? nexts[2] : start) : nexts[1]
            prev, cur = cur, nxt
        end
        length(loop) >= 3 &&
            push!(
            out, (;
                R = Float64[coords[k][1] for k in loop],
                Z = Float64[coords[k][2] for k in loop], closed,
            )
        )
    end
    return out
end

"""
    elem_zones(mesh_zone_field) -> Vector{Int}

Per-element M3D-C1 region id from the `mesh_zone` auxiliary field (its constant
Hermite coefficient, rounded): `1`=plasma, `2`=conductor (resistive wall),
`3`=vacuum. All `1` for a single-region (`imulti_region=0`) run.
"""
elem_zones(mesh_zone_field::AbstractMatrix) =
    Int[round(Int, Float64(mesh_zone_field[1, k])) for k in axes(mesh_zone_field, 2)]

"""
    eval_psi_and_derivs(coef, ξ, η) -> (ψ, p_ξ, p_η, p_ξξ, p_ξη, p_ηη)

Analytic evaluation of ψ and its first/second derivatives at the local
coordinate (ξ, η) using the **first 20 coefficients** of the reduced quintic
Hermite basis (= the ζ=0 slice of an 80-coef element column). Accepts an
80-element or 20-element vector. All derivatives are exact polynomial
expressions of the FEM solution — no finite-difference error.
"""
@inline function eval_psi_and_derivs(coef::AbstractVector, ξ::Real, η::Real)
    lpξ = (1.0, ξ, ξ^2, ξ^3, ξ^4, ξ^5)
    lpη = (1.0, η, η^2, η^3, η^4, η^5)
    ψ = 0.0;   p_ξ = 0.0;  p_η = 0.0
    p_ξξ = 0.0;  p_ξη = 0.0;  p_ηη = 0.0
    @inbounds for k in 1:20
        m = MI[k];  n = NI[k];  c_k = Float64(coef[k])
        ξm = lpξ[m + 1];  ηn = lpη[n + 1]
        ψ += c_k * ξm * ηn
        m ≥ 1 && (p_ξ += c_k * m * lpξ[m] * ηn)
        n ≥ 1 && (p_η += c_k * ξm * n * lpη[n])
        m ≥ 2 && (p_ξξ += c_k * m * (m - 1) * lpξ[m - 1] * ηn)
        (m ≥ 1 && n ≥ 1) && (p_ξη += c_k * m * n * lpξ[m] * lpη[n])
        n ≥ 2 && (p_ηη += c_k * ξm * n * (n - 1) * lpη[n - 1])
    end
    return (ψ, p_ξ, p_η, p_ξξ, p_ξη, p_ηη)
end
