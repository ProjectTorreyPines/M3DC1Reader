# imas_writer.jl — generic OMAS/IMAS HDF5 serializer.
#
# Writes a flat Dict of dotted IMAS paths into the OMAS nested-group HDF5
# layout (0-based integer subgroups, no attributes, fixed strings, Float64
# arrays, N-D arrays transposed for numpy/h5py row-major). Physics-agnostic.

"""
    write_omas_h5(path, ir::AbstractDict) -> path

Serialize `ir` (keys = dotted IMAS paths, e.g.
`"core_profiles.profiles_1d.0.electrons.temperature"`) into an OMAS-compatible
HDF5 file at `path`. Overwrites `path`.
"""
function write_omas_h5(path::AbstractString, ir::AbstractDict)
    h5open(String(path), "w") do f
        for (key, val) in ir
            _write_omas_leaf!(f, String(key), val)
        end
    end
    return String(path)
end

function _write_omas_leaf!(f::Union{HDF5.File, HDF5.Group}, key::AbstractString, val)
    segs = split(key, '.')
    parent = f
    @inbounds for s in @view segs[1:(end - 1)]
        sn = String(s)
        if haskey(parent, sn)
            parent = parent[sn]
            parent isa HDF5.Group ||
                error("write_omas_h5: path collision at '$sn' in key '$key' — a leaf was already written there")
        else
            parent = create_group(parent, sn)
        end
    end
    leaf = String(segs[end])
    (haskey(parent, leaf) && parent[leaf] isa HDF5.Group) &&
        error("write_omas_h5: path collision at '$leaf' in key '$key' — a group already exists there")
    return _write_omas_value!(parent, leaf, val)
end

# N-D arrays are stored transposed so the on-disk C/row-major order matches
# numpy's interpretation (a Julia (nR,nZ) array reads back as (nR,nZ) in h5py).
_omas_layout(A::AbstractArray) =
    ndims(A) ≥ 2 ? permutedims(A, reverse(ntuple(identity, ndims(A)))) : A

function _write_omas_value!(parent, name::AbstractString, val)
    if val isa AbstractString
        parent[name] = String(val)
    elseif val isa Integer
        parent[name] = Int64(val)
    elseif val isa Real
        parent[name] = Float64(val)
    elseif val isa AbstractArray{<:Real}
        parent[name] = _omas_layout(Array{Float64}(val))
    else
        error("write_omas_h5: unsupported value type $(typeof(val)) at leaf '$name'")
    end
    return nothing
end
