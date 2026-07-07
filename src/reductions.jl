# reductions.jl — toroidal averaging (n=0 mode), 2D rectilinear evaluation,
# and 1D adjoint reduction. The 1D adjoint step is code-agnostic; the 2D
# evaluation calls into the element layer (elements.jl).

# ============================================================================
# 1. Toroidal average  (3D 80-coef → 2D axisymmetric 80-coef)
# ============================================================================
"""
    average_toroidal_axisymmetric(field_coef, nplanes)

Compute the toroidal average of an M3D-C1 80-coefficient field across
`nplanes` identical (R,Z) toroidal planes. The mesh is plane-stacked, so the
average is taken element-wise across the plane axis.

# Arguments
- `field_coef::AbstractMatrix`: shape `(ncoef, nplanes*npp)` (typically
  `ncoef = 80`). Read directly from HDF5; do not transpose.
- `nplanes::Integer`: number of toroidal planes (e.g. 16 for m3d_smoke).

# Returns
- `Matrix{Float64}` of shape `(ncoef, npp)` — the axisymmetric (n=0 Fourier-
  mode trapezoidal approximation) coefficients on a single plane.

The mean is taken in raw coefficient space; this is mathematically equivalent
to averaging the function value at any (ξ, η, ζ=0) because Hermite-polynomial
evaluation is linear in the coefficients and the mesh is plane-identical.

# Example
```julia
psi_coef_3D = read(h5["time_022/fields/psi"])     # (80, 176528)
psi_coef_ax = average_toroidal_axisymmetric(psi_coef_3D, 16)   # (80, 11033)
```
"""
function average_toroidal_axisymmetric(field_coef::AbstractMatrix, nplanes::Integer)
    ncoef, nelms_total = size(field_coef)
    npp = nelms_total ÷ nplanes
    return average_toroidal_axisymmetric!(Matrix{Float64}(undef, ncoef, npp), field_coef, nplanes)
end

"""
    average_toroidal_axisymmetric!(out, field_coef, nplanes) -> out

In-place [`average_toroidal_axisymmetric`](@ref): write the n=0 average into the
caller-provided `(ncoef, npp)` matrix `out` (`npp = size(field_coef,2) ÷ nplanes`)
instead of allocating. Lets a per-slice caller draw the coefficient buffer from a
pool (the averaged coefficients are a slice-local temporary — consumed by the 2D
interpolation / FSA and never returned), avoiding one ~`ncoef·npp` allocation per
field per slice.
"""
function average_toroidal_axisymmetric!(
        out::AbstractMatrix{Float64},
        field_coef::AbstractMatrix, nplanes::Integer
    )
    ncoef, nelms_total = size(field_coef)
    nelms_total % nplanes == 0 ||
        error("nelms_total ($nelms_total) is not divisible by nplanes ($nplanes)")
    npp = nelms_total ÷ nplanes
    size(out) == (ncoef, npp) ||
        error("out has size $(size(out)), expected ($ncoef, $npp)")
    fill!(out, 0.0)
    @inbounds for p in 1:nplanes
        off = (p - 1) * npp
        for k in 1:npp, c in 1:ncoef
            out[c, k] += Float64(field_coef[c, off + k])
        end
    end
    out ./= nplanes
    return out
end

"""
    interpolate_at_zeta_to_grid(coef, elems_plane, R_grid, Z_grid, ζ;
                                  id_map=nothing)

Evaluate a single-plane 80-coef field at given element-local toroidal
coordinate `ζ` on a rectilinear R-Z grid. Use for non-axisymmetric 3D
visualization (each `time_NNN/fields/*` plane slice with `ζ` ∈ [0, Δφ]).

# Arguments
- `coef::AbstractMatrix`: shape `(80, npp)`. ONE plane's coefficients
  (e.g. `time_NNN/fields/psi[:, (plane_idx-1)*npp+1 : plane_idx*npp]`).
- `elems_plane`, `R_grid`, `Z_grid`, `id_map`: as in `interpolate_axisym_to_grid`.
- `ζ::Real`: element-local toroidal offset, normally in `[0, period/nplanes]`.

# Returns
Same shape/semantics as `interpolate_axisym_to_grid`.
"""
function interpolate_at_zeta_to_grid(
        coef::AbstractMatrix,
        elems_plane::AbstractMatrix,
        R_grid::AbstractVector,
        Z_grid::AbstractVector,
        ζ::Real;
        id_map::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
    )
    nR, nZ = length(R_grid), length(Z_grid)
    map_ = id_map === nothing ? build_grid_to_element_map(R_grid, Z_grid, elems_plane) : id_map
    b = view(elems_plane, 2, :);  x = view(elems_plane, 5, :)
    z = view(elems_plane, 6, :);  θ = view(elems_plane, 4, :)
    out = fill(NaN, nR, nZ)
    R_arr = collect(R_grid);  Z_arr = collect(Z_grid)
    @inbounds for j in 1:nZ, i in 1:nR
        k = map_[i, j]
        k == 0 && continue
        ξ, η = global_to_local(R_arr[i], Z_arr[j], x[k], z[k], b[k], θ[k])
        out[i, j] = eval_at_local(view(coef, :, k), ξ, η, ζ)
    end
    return out
end

"""
    interpolate_axisym_to_grid(coef_axisym, elems_plane, R_grid, Z_grid;
                                id_map=nothing)

Evaluate an axisymmetric M3D-C1 Hermite-coefficient field on a rectilinear
R-Z grid. Uses single-plane mesh element search + element-local polynomial
evaluation (the same algorithm as `eval_field_on_RectGrid_vectorized` in the
matlab visM3D `c_M3D_Grid_Converter`).

# Arguments
- `coef_axisym::AbstractMatrix`: shape `(≥20, npp)`. Typically the output of
  `average_toroidal_axisymmetric`. Only the first 20 rows are used (ζ⁰ layer).
- `elems_plane::AbstractMatrix`: `(10, npp)`.
- `R_grid::AbstractVector`, `Z_grid::AbstractVector`: rectilinear 1D grids.
- `id_map`: optional precomputed element id map from `build_grid_to_element_map`.
  If `nothing`, it is built internally (one-time cost).

# Returns
- `Matrix{Float64}` of shape `(length(R_grid), length(Z_grid))`. Points
  outside the mesh are `NaN`.

# Reuse
The same `id_map` can be reused across multiple fields and time slices as
long as the mesh is unchanged, which avoids re-running the element-search step.
"""
function interpolate_axisym_to_grid(
        coef_axisym::AbstractMatrix,
        elems_plane::AbstractMatrix,
        R_grid::AbstractVector,
        Z_grid::AbstractVector;
        id_map::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
    )
    nR, nZ = length(R_grid), length(Z_grid)
    map_ = id_map === nothing ? build_grid_to_element_map(R_grid, Z_grid, elems_plane) : id_map

    b = view(elems_plane, 2, :);  x = view(elems_plane, 5, :)
    z = view(elems_plane, 6, :);  θ = view(elems_plane, 4, :)

    out = fill(NaN, nR, nZ)
    R_arr = collect(R_grid);  Z_arr = collect(Z_grid)
    @inbounds for j in 1:nZ, i in 1:nR
        k = map_[i, j]
        k == 0 && continue
        ξ, η = global_to_local(R_arr[i], Z_arr[j], x[k], z[k], b[k], θ[k])
        out[i, j] = eval_axisym_at_local(view(coef_axisym, :, k), ξ, η)
    end
    return out
end

"""
    interpolate_axisym_gradient_to_grid(coef_axisym, elems_plane, R_grid, Z_grid;
                                        id_map=nothing) -> (; val, dR, dZ)

Like [`interpolate_axisym_to_grid`](@ref), but also return the exact first
derivatives of the FEM polynomial, rotated from each element's local `(ξ,η)`
frame into global `(R,Z)` (the same rotation as `critical_point_system`).
Off-mesh points are `NaN` in all three `(nR, nZ)` matrices.

This is the ∇ψ primitive of the field assembler: with ψ per-radian,
`B_R = −(∂ψ/∂Z)/R`, `B_Z = (∂ψ/∂R)/R` (and `Bφ = I/R` from the `:I` field).
"""
function interpolate_axisym_gradient_to_grid(
        coef_axisym::AbstractMatrix,
        elems_plane::AbstractMatrix,
        R_grid::AbstractVector,
        Z_grid::AbstractVector;
        id_map::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
    )
    nR, nZ = length(R_grid), length(Z_grid)
    map_ = id_map === nothing ? build_grid_to_element_map(R_grid, Z_grid, elems_plane) : id_map

    b = view(elems_plane, 2, :);  x = view(elems_plane, 5, :)
    z = view(elems_plane, 6, :);  θ = view(elems_plane, 4, :)

    val = fill(NaN, nR, nZ);  dR = fill(NaN, nR, nZ);  dZ = fill(NaN, nR, nZ)
    R_arr = collect(R_grid);  Z_arr = collect(Z_grid)
    @inbounds for j in 1:nZ, i in 1:nR
        k = map_[i, j]
        k == 0 && continue
        θk = Float64(θ[k]);  co = cos(θk);  sn = sin(θk)
        ξ, η = global_to_local(R_arr[i], Z_arr[j], x[k], z[k], b[k], θk)
        ψ, pξ, pη, _, _, _ = eval_psi_and_derivs(view(coef_axisym, :, k), ξ, η)
        val[i, j] = ψ
        dR[i, j] = pξ * co - pη * sn
        dZ[i, j] = pξ * sn + pη * co
    end
    return (; val, dR, dZ)
end

"""
    deltab_over_b_rz(psi3d, I3d, f3d, nplanes, elems_plane, R_grid, Z_grid;
                     id_map=nothing, fprime=true) -> (; db, br0, bz0, bphi0)

Magnetic-fluctuation map δB/B on the (R,Z) grid from the full 3D coefficient
fields, plane-sampled: for each toroidal plane, assemble the exact-FEM
`B = ∇ψ×∇φ + F∇φ − ∇⊥(∂f/∂φ)` (per-radian ψ; `B_R = −ψ_Z/R − ∂f′/∂R`,
`B_Z = +ψ_R/R − ∂f′/∂Z`, `Bφ = I/R` — fusion-io m3dc1_field.cpp:603), then

    δB   = √(2 · Σ_comp Var_φ(B_comp))      (NIMROD's Σ_{n≥1}(reₙ²+imₙ²) form)
    δB/B = δB / max(|⟨B⟩_φ|, 10⁻⁸·max|⟨B⟩_φ|)

Plane sampling is an exact toroidal variance for modes n < nplanes/2; the
√2 turns the RMS into the DB's mode-amplitude normalization. See
docs/deltab_over_b.md.

`f3d` may be `nothing` (ψ/F only). `fprime` says whether `f3d` stores
f′ = ∂f/∂φ (the `fp` dataset of modern files —
[`M3DC1Reader._fprime_stored`](@ref)) or the potential f itself (legacy `f`) —
in the latter case the ∂/∂φ at each plane is the second 20-coefficient
Hermite layer. Off-mesh points are `NaN`; `nplanes == 1` gives `db ≡ 0`.
Returned `br0/bz0/bphi0` are the plane-mean (n=0) components in normalized
units — they reproduce the axisymmetric assembler's maps up to the
machine-small discrete mean of the f′ term.
"""
function deltab_over_b_rz(
        psi3d::AbstractMatrix, I3d::AbstractMatrix,
        f3d::Union{Nothing, AbstractMatrix}, nplanes::Integer,
        elems_plane::AbstractMatrix,
        R_grid::AbstractVector, Z_grid::AbstractVector;
        id_map::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
        fprime::Bool = true
    )
    nR, nZ = length(R_grid), length(Z_grid)
    ntot = size(psi3d, 2)
    ntot % nplanes == 0 ||
        error("nelms_total ($ntot) is not divisible by nplanes ($nplanes)")
    size(I3d, 2) == ntot || error("psi and I coefficient tables disagree in size")
    npp = ntot ÷ nplanes
    map_ = id_map === nothing ? build_grid_to_element_map(R_grid, Z_grid, elems_plane) : id_map
    R_arr = collect(Float64, R_grid)

    sR = zeros(nR, nZ);  sZ = zeros(nR, nZ);  sP = zeros(nR, nZ)
    s2R = zeros(nR, nZ);  s2Z = zeros(nR, nZ);  s2P = zeros(nR, nZ)
    for p in 1:nplanes
        cols = ((p - 1) * npp + 1):(p * npp)
        pg = interpolate_axisym_gradient_to_grid(
            view(psi3d, :, cols),
            elems_plane, R_grid, Z_grid; id_map = map_
        )
        Iv = interpolate_axisym_to_grid(
            view(I3d, :, cols),
            elems_plane, R_grid, Z_grid; id_map = map_
        )
        fg = f3d === nothing ? nothing :
            interpolate_axisym_gradient_to_grid(
                fprime ? view(f3d, :, cols) : view(f3d, 21:40, cols),
                elems_plane, R_grid, Z_grid; id_map = map_
            )
        @inbounds for j in 1:nZ, i in 1:nR
            br = -pg.dZ[i, j] / R_arr[i]
            bz = pg.dR[i, j] / R_arr[i]
            bp = Iv[i, j] / R_arr[i]
            if fg !== nothing
                br -= fg.dR[i, j]
                bz -= fg.dZ[i, j]
            end
            sR[i, j] += br;      sZ[i, j] += bz;      sP[i, j] += bp
            s2R[i, j] += br * br; s2Z[i, j] += bz * bz; s2P[i, j] += bp * bp
        end
    end

    br0 = sR ./ nplanes;  bz0 = sZ ./ nplanes;  bphi0 = sP ./ nplanes
    b0 = sqrt.(br0 .^ 2 .+ bz0 .^ 2 .+ bphi0 .^ 2)
    b0max = maximum(x -> isfinite(x) ? x : -Inf, b0)
    floor_ = isfinite(b0max) && b0max > 0 ? 1.0e-8 * b0max : 0.0
    db = fill(NaN, nR, nZ)
    @inbounds for j in 1:nZ, i in 1:nR
        isfinite(b0[i, j]) || continue
        var = (s2R[i, j] / nplanes - br0[i, j]^2) +
            (s2Z[i, j] / nplanes - bz0[i, j]^2) +
            (s2P[i, j] / nplanes - bphi0[i, j]^2)
        db[i, j] = sqrt(2 * max(var, 0.0)) / max(b0[i, j], floor_)
    end
    return (; db, br0, bz0, bphi0)
end

# ============================================================================
# 3. 1D reduction (ψ vs func point cloud)
# ============================================================================
"""
    reduce_1d_psi_func(psi_pts, func_pts;
                        psi_grid=nothing, n_bins=60, psi_range=nothing,
                        adj=:linear, weights=nothing,
                        sigma_cells=1.5, ntrunc=2.0)

Reduce a (ψ, function-value) point cloud to a 1D profile: a kernel-weighted
per-bin average `Σᵢ K(ψᵢ) wᵢ fᵢ / Σᵢ K(ψᵢ) wᵢ`.

Accepts either:
- a true unstructured scatter (e.g. element vertex-1 values), or
- a flattened 2D rectilinear grid (`vec(ψ_2d), vec(func_2d)`).

# Arguments
- `psi_pts`, `func_pts`: 1D arrays of equal length. NaN entries are dropped.
- `psi_grid`: optional explicit grid. If `nothing`, built from `psi_range`
  or data extrema with `n_bins` points.
- `n_bins::Integer = 60`.
- `psi_range::Union{Nothing, Tuple{Real, Real}} = nothing`: explicit (lo, hi).
- `weights`: optional per-point quadrature weights (same length; NaN-weight
  points are dropped). For a physical flux-surface average pass the volume
  measure — `cell_area · R` for (R,Z) samples (the 1/|∇ψ| factor is implicit
  in uniform-density sampling). `nothing` (default) = equal weights.
- `adj::Symbol = :linear` (default): kernel selection
    `:linear`   — `linear_adjoint` (hat-function distribution)         **default**
    `:constant` — `constant_adjoint` (= NIMROD `np.bincount` simple bin mean)
    `:gauss`    — truncated-Gaussian binning: σ = `sigma_cells` grid cells,
                   support cut at `ntrunc`·σ. Non-negative kernel with built-in
                   smoothing (blurs sharp features accordingly). Assumes a
                   uniform `psi_grid`.
    `:cubic`    — `cubic_adjoint`. ⚠ its kernel has negative side-lobes, so
                   the result is not a convex average (can over/undershoot);
                   kept for comparison only.
    `:pchip`    — `pchip_adjoint` (monotonic Fritsch–Carlson Hermite).
                   Uses `constant_adjoint` result as the y operating point
                   so PCHIP's monotonic slope branching is well-defined.

# Returns
NamedTuple `(psi_grid, func_bin, den)`. `den[k] = Σᵢ Kₖ(ψᵢ)wᵢ` is the
kernel-weighted sample mass per node — with `weights = cell_area·R` it is the
flux-shell volume density: at interior nodes of a uniform grid,
`dV/dx ≈ 2π·den/h` (`h` = grid spacing). Boundary nodes carry only half a
kernel support (plus clamped out-of-range samples), so their `den` is biased.
"""
@with_pool pool function reduce_1d_psi_func(
        psi_pts::AbstractVector, func_pts::AbstractVector;
        psi_grid::Union{Nothing, AbstractVector} = nothing,
        n_bins::Integer = 60,
        psi_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        adj::Symbol = :linear,
        weights::Union{Nothing, AbstractVector} = nothing,
        sigma_cells::Real = 1.5, ntrunc::Real = 2.0
    )
    n_all = length(psi_pts)
    n_all == length(func_pts) ||
        error("psi_pts and func_pts must have equal length")
    weights === nothing || length(weights) == n_all ||
        error("weights must match psi_pts length")
    # Single-pass finite gather into pooled buffers: ψ/f/w and f·w are per-call
    # temporaries (they never escape — only the num/den/func_bin they feed do),
    # so acquire! them from the pool to avoid re-allocating on every call.
    _fin(i) = isfinite(psi_pts[i]) & isfinite(func_pts[i]) &
        (weights === nothing || isfinite(weights[i]))
    ncnt = 0
    @inbounds for i in 1:n_all
        _fin(i) && (ncnt += 1)
    end
    ncnt >= 2 || error("not enough finite points (got $ncnt)")
    ψ = acquire!(pool, Float64, ncnt)
    f = acquire!(pool, Float64, ncnt)
    w = acquire!(pool, Float64, ncnt)
    k = 0
    @inbounds for i in 1:n_all
        _fin(i) || continue
        k += 1
        ψ[k] = psi_pts[i]
        f[k] = func_pts[i]
        w[k] = weights === nothing ? 1.0 : weights[i]
    end

    grid = if psi_grid !== nothing
        Float64.(collect(psi_grid))
    else
        lo, hi = psi_range === nothing ? extrema(ψ) : psi_range
        collect(range(Float64(lo), Float64(hi), length = n_bins))
    end

    fw = acquire!(pool, Float64, ncnt)
    @inbounds @. fw = f * w
    if adj === :constant
        A = constant_adjoint((grid,), (ψ,); extrap = ClampExtrap())
        num = A(fw);  den = A(w)
    elseif adj === :linear
        A = linear_adjoint((grid,), (ψ,); extrap = ClampExtrap())
        num = A(fw);  den = A(w)
    elseif adj === :gauss
        num, den = _gauss_bin(grid, ψ, f, w, Float64(sigma_cells), Float64(ntrunc))
    elseif adj === :cubic
        @warn "reduce_1d_psi_func adj=:cubic — the cubic adjoint kernel has negative " *
            "side-lobes (not a convex average); prefer :linear or :gauss" maxlog = 1
        A = cubic_adjoint((grid,), (ψ,); extrap = ClampExtrap())
        num = A(fw);  den = A(w)
    elseif adj === :pchip
        # PCHIP adjoint requires a y operating point for monotonic slope branching.
        # Use a constant_adjoint pass as y_init, then apply PCHIP adjoint.
        A0 = constant_adjoint((grid,), (ψ,); extrap = ClampExtrap())
        num0 = A0(fw);  den0 = A0(w)
        y_init = [d > 0 ? n / d : 0.0 for (n, d) in zip(num0, den0)]
        Ap = pchip_adjoint(grid, y_init, ψ; extrap = ClampExtrap())
        num = Ap(fw);  den = Ap(w)
    else
        error("unknown adj=$(adj); choose :constant, :linear, :gauss, :cubic, or :pchip")
    end

    func_bin = [d > 0 ? n / d : NaN for (n, d) in zip(num, den)]
    return (; psi_grid = grid, func_bin, den = collect(Float64, den))
end

# Truncated-Gaussian binning: each sample deposits exp(-(x-xk)²/2σ²)·w onto the
# grid nodes within ±ntrunc·σ; σ = sigma_cells grid cells (uniform grid assumed).
function _gauss_bin(
        grid::Vector{Float64}, ψ::Vector{Float64}, f::Vector{Float64},
        w::Vector{Float64}, sigma_cells::Float64, ntrunc::Float64
    )
    nb = length(grid)
    nb ≥ 2 || error("adj=:gauss needs a grid with ≥ 2 points")
    σ = sigma_cells * (grid[2] - grid[1])
    iv = 1.0 / (2σ^2);  rad = ntrunc * σ
    num = zeros(nb);  den = zeros(nb)
    @inbounds for i in eachindex(ψ)
        x = ψ[i]
        klo = searchsortedfirst(grid, x - rad)
        khi = searchsortedlast(grid, x + rad)
        for k in klo:khi
            g = exp(-(grid[k] - x)^2 * iv) * w[i]
            num[k] += g * f[i]
            den[k] += g
        end
    end
    return num, den
end

# ============================================================================
# Convenience: end-to-end pipeline
# ============================================================================
"""
    pipeline_3d_to_1d(field_coef_3D, elems_3D, nplanes,
                       R_grid, Z_grid;
                       psi_axis=nothing, psi_lcfs=nothing,
                       n_bins=60, smooth=true, id_map=nothing)

End-to-end convenience wrapper:
  1. average_toroidal_axisymmetric
  2. interpolate_axisym_to_grid for both `field_coef_3D` AND a separate
     psi coefficient array passed via `psi_coef_3D` (… see signature note)

Most users will prefer calling the three primitive functions directly so they
can reuse `id_map` and axisymmetric ψ across multiple fields.
"""
function pipeline_3d_to_1d(
        field_coef_3D::AbstractMatrix,
        psi_coef_3D::AbstractMatrix,
        elems_3D::AbstractMatrix,
        nplanes::Integer,
        R_grid::AbstractVector,
        Z_grid::AbstractVector;
        psi_axis::Union{Nothing, Real} = nothing,
        psi_lcfs::Union{Nothing, Real} = nothing,
        n_bins::Integer = 60,
        smooth::Bool = true,
        id_map::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
    )
    nelms_total = size(elems_3D, 2)
    npp = nelms_total ÷ nplanes
    elems_plane = elems_3D[:, 1:npp]

    field_ax = average_toroidal_axisymmetric(field_coef_3D, nplanes)
    psi_axisym_coef = average_toroidal_axisymmetric(psi_coef_3D, nplanes)

    if id_map === nothing
        id_map = build_grid_to_element_map(R_grid, Z_grid, elems_plane)
    end
    field_2d = interpolate_axisym_to_grid(
        field_ax, elems_plane,
        R_grid, Z_grid; id_map = id_map
    )
    psi_rz = interpolate_axisym_to_grid(
        psi_axisym_coef, elems_plane,
        R_grid, Z_grid; id_map = id_map
    )

    # normalize ψ → ψ_norm if axis/lcfs provided
    if psi_axis !== nothing && psi_lcfs !== nothing
        Δψ = psi_lcfs - psi_axis
        psi_rz = (psi_rz .- psi_axis) ./ Δψ
    end

    res = reduce_1d_psi_func(vec(psi_rz), vec(field_2d); n_bins = n_bins)

    return (; field_2d, psi_rz, id_map, res...)
end
