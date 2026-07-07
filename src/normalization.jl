# normalization.jl — flux-coordinate helpers (ψ_N, ρ_pol).
#
# Physical unit conversion (Te→eV, fields→SI, …) lives in units.jl.

"""
    psi_to_psi_norm(psi, psi_axis, psi_lcfs) -> ψ_N

Normalized poloidal flux ψ_N = (ψ − ψ_axis) / (ψ_lcfs − ψ_axis).
"""
@inline psi_to_psi_norm(psi::Real, psi_axis::Real, psi_lcfs::Real) =
    (psi - psi_axis) / (psi_lcfs - psi_axis)

"""
    psi_n_to_rho_pol(ψn) -> Float64

Normalized poloidal flux ψ_N → ρ_pol = √(max(ψ_N, 0)). The clamp guards
floating-point round-off near the magnetic axis, where ψ_N can dip a hair
below zero. This matches the `rho_pol_norm` radial coordinate used by the
NIMROD IMAS path in `disruption_database_tools`.
"""
@inline psi_n_to_rho_pol(ψn::Real) = sqrt(max(Float64(ψn), 0.0))
