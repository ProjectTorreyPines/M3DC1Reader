# units.jl — M3D-C1 unit normalization, selectable per quantity and per system.
#
# M3D-C1 stores fields in its own dimensionless ("M3D") normalization. Every
# physical quantity has a scale factor built from the run's normalization
# constants (b0_norm, n0_norm, l0_norm, ion_mass). This module turns those into
# a small data-driven registry: pick a quantity and a target system
# (`:m3d`, `:cgs`, `:si`) and get the multiplicative factor (or convert data).
#
# Ports the non-visualization unit logic of PPPL visM3D
# (`c_M3D_Grid_Converter.m:Units_cgs_mks`), using the physically-grounded
# cgs/mks table (so e.g. current is statA→A = ×10/c, not visM3D's `-3e9`).

# physical constants (CGS)
const _MP_CGS = 1.6726219e-24   # proton mass [g]
const _C_CGS = 2.9979e10       # speed of light [cm/s]
const _EV_ERG = 1.6022e-12      # [erg/eV]

"""
    M3DNormalization(; b0, n0, l0, ion_mass)

M3D-C1 normalization constants (C1.h5 root attributes):
`b0`=`b0_norm` [Gauss], `n0`=`n0_norm` [cm⁻³], `l0`=`l0_norm` [cm],
`ion_mass`=mᵢ/mₚ. The derived Alfvén velocity `v0` and time `t0` are cached:

    v0 = b0 / √(4π · ion_mass·mₚ · n0)   [cm/s]
    t0 = l0 / v0                          [s]
"""
struct M3DNormalization
    b0::Float64
    n0::Float64
    l0::Float64
    ion_mass::Float64
    v0::Float64
    t0::Float64
end

function M3DNormalization(; b0::Real, n0::Real, l0::Real, ion_mass::Real)
    mi_g = ion_mass * _MP_CGS
    v0 = b0 / sqrt(4π * mi_g * n0)
    t0 = l0 / v0
    return M3DNormalization(b0, n0, l0, ion_mass, v0, t0)
end

"""
    UnitSpec(cgs, cgs_to_si, cgs_unit, si_unit)

Recipe for one physical quantity: `cgs(norm)` gives the M3D→cgs scale factor,
`cgs_to_si` multiplies it to reach SI, and the two strings are display labels.
"""
struct UnitSpec
    cgs::Function       # M3DNormalization -> Float64   (M3D → cgs factor)
    cgs_to_si::Float64  # cgs → SI multiplier (constant)
    cgs_unit::String
    si_unit::String
end

# Registry. Add a quantity = add a row. Factors follow visM3D Units_cgs_mks.
const UNITS = Dict{Symbol, UnitSpec}(
    :density => UnitSpec(u -> u.n0, 1.0e6, "cm^-3", "m^-3"),
    :velocity => UnitSpec(u -> u.v0, 1.0e-2, "cm/s", "m/s"),
    :time => UnitSpec(u -> u.t0, 1.0, "s", "s"),
    :length => UnitSpec(u -> u.l0, 1.0e-2, "cm", "m"),
    :magnetic_field => UnitSpec(u -> u.b0, 1.0e-4, "G", "T"),
    :magnetic_flux => UnitSpec(u -> u.b0 * u.l0^2, 1.0e-8, "Mx", "Wb"),
    :temperature => UnitSpec(u -> u.b0^2 / (4π * u.n0 * _EV_ERG), 1.0, "eV", "eV"),
    :pressure => UnitSpec(u -> u.b0^2 / (4π), 0.1, "erg/cm^3", "Pa"),
    :power_density => UnitSpec(u -> u.b0^2 / (4π * u.t0), 0.1, "erg/cm^3/s", "W/m^3"),
    :energy => UnitSpec(u -> u.b0^2 * u.l0^3 / (4π), 1.0e-7, "erg", "J"),
    :power => UnitSpec(u -> u.b0^2 * u.l0^3 / (4π * u.t0), 1.0e-7, "erg/s", "W"),
    :force => UnitSpec(u -> u.b0^2 * u.l0^2 / (4π), 1.0e-7, "dyn", "N"),
    :current => UnitSpec(u -> _C_CGS * u.b0 * u.l0 / (4π), 10 / _C_CGS, "statA", "A"),
    :current_density => UnitSpec(u -> _C_CGS * u.b0 / (4π * u.l0), 1.0e5 / _C_CGS, "statA/cm^2", "A/m^2"),
    :electric_field => UnitSpec(u -> u.v0 * u.b0 / _C_CGS, _C_CGS / 1.0e6, "statV/cm", "V/m"),
    :voltage => UnitSpec(u -> u.v0^2 * u.b0 * u.t0 / _C_CGS, _C_CGS / 1.0e8, "statV", "V"),
    :diffusion => UnitSpec(u -> u.l0^2 / u.t0, 1.0e-4, "cm^2/s", "m^2/s"),
    :resistivity => UnitSpec(u -> 4π * u.t0 * u.v0^2 / _C_CGS^2, _C_CGS^2 / 1.0e11, "s", "Ohm*m"),
    :thermal_conductivity => UnitSpec(u -> u.n0 * u.l0^2 / u.t0, 100.0, "1/cm-s", "1/m-s"),
    :viscosity => UnitSpec(u -> u.ion_mass * _MP_CGS * u.n0 * u.l0^2 / u.t0, 0.1, "P", "Pa*s"),
)

"""
    available_quantities() -> Vector{Symbol}

Sorted list of quantities known to [`unit_factor`](@ref).
"""
available_quantities() = sort!(collect(keys(UNITS)))

@inline _normsystem(s::Symbol) = s === :mks ? :si : s   # :mks is an alias for :si

"""
    unit_factor(norm, quantity; system=:si) -> Float64

Multiplicative factor converting `quantity` from M3D-C1 normalized units to
`system` (`:m3d` ⇒ 1, `:cgs`, or `:si`/`:mks`). Multiply M3D-normalized data
by this to obtain physical units.

```julia
norm = M3DNormalization(b0=1e4, n0=1e14, l0=100, ion_mass=2)
unit_factor(norm, :temperature)            # eV per M3D unit
unit_factor(norm, :magnetic_field)         # 1.0  (1e4 G ⇒ 1 T)
```
"""
function unit_factor(norm::M3DNormalization, quantity::Symbol; system::Symbol = :si)
    sys = _normsystem(system)
    sys === :m3d && return 1.0
    spec = get(UNITS, quantity) do
        error("unknown quantity :$quantity; available: $(available_quantities())")
    end
    f = spec.cgs(norm)
    sys === :cgs && return f
    sys === :si  && return f * spec.cgs_to_si
    error("unknown unit system :$system (use :m3d, :cgs, :si/:mks)")
end

"""
    to_units(norm, data, quantity; system=:si)

Convert `data` from M3D-C1 normalized units to `system`, returning
`data .* unit_factor(norm, quantity; system)`. Works on scalars or arrays.
"""
to_units(norm::M3DNormalization, data, quantity::Symbol; system::Symbol = :si) =
    data .* unit_factor(norm, quantity; system)

"""
    unit_label(quantity; system=:si) -> String

Display label for `quantity` in `system` (`"normalized"` for `:m3d`).
"""
function unit_label(quantity::Symbol; system::Symbol = :si)
    sys = _normsystem(system)
    sys === :m3d && return "normalized"
    spec = get(UNITS, quantity) do
        error("unknown quantity :$quantity; available: $(available_quantities())")
    end
    sys === :cgs && return spec.cgs_unit
    sys === :si  && return spec.si_unit
    error("unknown unit system :$system (use :m3d, :cgs, :si/:mks)")
end
