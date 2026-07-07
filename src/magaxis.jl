# magaxis.jl — magnetic axis / O-point finding via Newton iteration on the
# FEM polynomial (mirrors M3D-C1 diagnostics.f90:magaxis, imethod=0).

"""
    find_axis_newton(coef_plane, elems_plane, R_guess, Z_guess;
                      max_iter=50, rel_tol=1e-3, step_factor=0.1)
        -> (R, Z, ψ, grad_norm, n_iter, converged, elem_idx)

Find a local minimum of ψ via Newton iteration on the C¹ reduced quintic
Hermite FEM polynomial, following M3D-C1 `diagnostics.f90:magaxis`
(`imethod=0`). Operates on the ζ=0 slice of `coef_plane` — i.e. on the
`(20, npp)` slab of polynomial coefficients at the chosen toroidal plane
(or the toroidally-averaged set).

Algorithm:
1. Locate the element containing the current point (R, Z).
2. Analytically evaluate ψ, ∇ψ, ∇²ψ on the element's polynomial.
3. Newton step in (ξ, η):  Δ = -H⁻¹ ∇ψ.
4. Rotate Δ(ξ, η) → Δ(R, Z); cap step at `step_factor · h` where `h =
   √((a+b)·c)` is the element's characteristic size (geometric mean of
   base × height, matching M3D-C1 `diagnostics.f90:1416`).
5. Converged when ‖Δ(R, Z)‖ < `rel_tol · h`.

# Returns
NamedTuple with the converged position `(R, Z)`, ψ value, ‖∇ψ‖ at that point,
iteration count, convergence flag, and the final containing element index.
A `converged=false` result with `elem_idx=0` indicates a Newton step left the
mesh — try a closer initial guess.
"""
function find_axis_newton(
        coef_plane::AbstractMatrix,
        elems_plane::AbstractMatrix,
        R_guess::Real, Z_guess::Real;
        max_iter::Integer = 50,
        rel_tol::Real = 1.0e-3,
        step_factor::Real = 0.1,
        record_trajectory::Bool = false
    )
    size(coef_plane, 1) ≥ 20 ||
        error("coef_plane needs ≥ 20 rows (got $(size(coef_plane, 1)))")
    size(coef_plane, 2) == size(elems_plane, 2) ||
        error("coef_plane / elems_plane element-count mismatch")

    R = Float64(R_guess);  Z = Float64(Z_guess)
    elem_idx = 0
    ψ_val = NaN;  grad_norm = NaN
    converged = false;  n_iter = 0
    traj = record_trajectory ?
        (
            R = Float64[], Z = Float64[], ψ = Float64[], grad = Float64[],
            elem = Int[], step = Float64[],
        ) : nothing

    a_col = view(elems_plane, 1, :);  b_col = view(elems_plane, 2, :)
    c_col = view(elems_plane, 3, :);  θ_col = view(elems_plane, 4, :)
    x_col = view(elems_plane, 5, :);  z_col = view(elems_plane, 6, :)

    for iter in 1:max_iter
        n_iter = iter
        elem_idx = locate_element(R, Z, elems_plane)
        elem_idx == 0 && break    # walked off the mesh

        a = Float64(a_col[elem_idx]);  b = Float64(b_col[elem_idx])
        c = Float64(c_col[elem_idx]);  θ = Float64(θ_col[elem_idx])
        x_e = Float64(x_col[elem_idx]);  z_e = Float64(z_col[elem_idx])
        h = sqrt((a + b) * c)    # matches M3D-C1 diagnostics.f90:1416

        ξ, η = global_to_local(R, Z, x_e, z_e, b, θ)
        ψ_val, p_ξ, p_η, p_ξξ, p_ξη, p_ηη =
            eval_psi_and_derivs(view(coef_plane, :, elem_idx), ξ, η)
        grad_norm = hypot(p_ξ, p_η)

        denom = p_ξξ * p_ηη - p_ξη * p_ξη
        abs(denom) < 1.0e-30 && break

        Δξ = -(p_ηη * p_ξ - p_ξη * p_η) / denom
        Δη = -(-p_ξη * p_ξ + p_ξξ * p_η) / denom

        co, sn = cos(θ), sin(θ)
        ΔR = Δξ * co - Δη * sn
        ΔZ = Δξ * sn + Δη * co

        rdiff = hypot(ΔR, ΔZ)
        if rdiff > step_factor * h
            scale = step_factor * h / rdiff
            ΔR *= scale;  ΔZ *= scale
            rdiff = step_factor * h
        end

        if traj !== nothing
            push!(traj.R, R);   push!(traj.Z, Z);   push!(traj.ψ, ψ_val)
            push!(traj.grad, grad_norm);  push!(traj.elem, elem_idx)
            push!(traj.step, rdiff)
        end

        R += ΔR;  Z += ΔZ

        if rdiff < rel_tol * h
            # final evaluation at converged point
            elem_idx = locate_element(R, Z, elems_plane)
            if elem_idx > 0
                ξf, ηf = global_to_local(
                    R, Z,
                    Float64(x_col[elem_idx]),
                    Float64(z_col[elem_idx]),
                    Float64(b_col[elem_idx]),
                    Float64(θ_col[elem_idx])
                )
                ψ_val, p_ξ, p_η, _, _, _ =
                    eval_psi_and_derivs(view(coef_plane, :, elem_idx), ξf, ηf)
                grad_norm = hypot(p_ξ, p_η)
            end
            converged = true
            break
        end
    end

    return (;
        R, Z, ψ = ψ_val, grad_norm, n_iter, converged, elem_idx,
        trajectory = traj,
    )
end
