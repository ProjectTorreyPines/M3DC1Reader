module M3DC1ReaderEFITExt

using M3DC1Reader
using EFIT
using MXHEquilibrium

"""
    to_geqdsk(eq::M3DAxisymField; npsi=length(eq.psi1d)) -> EFIT.GEQDSKFile

Adapt a `M3DAxisymField` into an in-memory `EFIT.GEQDSKFile` (gEQDSK format).
gEQDSK's ψ is per-radian (Weber/rad) like the native COCOS-1 field — no 2π
transform, unlike the IMAS (COCOS 11) path. The q sign follows `eq.cocos`
(COCOS 1 = EFIT-like signed q, sign(q)=sign(Ip·F); a cocos=5-labelled input
yields the σ_ρθφ-flipped q). Reuses `to_mxh` (MXHEquilibrium) as the analysis
engine for the q profile (FSA), plasma current, LCFS outline, and normalized
toroidal flux; `F`/`p` are resampled onto a uniform ψ grid (gEQDSK requires it).

No file I/O — the returned struct is in-memory; call `EFIT.writeg(g, path)`
separately only when an on-disk `.geqdsk` file is wanted. Requires closed nested
flux surfaces (equilibrium-like slices); on a non-nested slice the MXHEquilibrium
boundary/FSA step fails (see `to_mxh`).
"""
function M3DC1Reader.to_geqdsk(
        eq::M3DC1Reader.M3DAxisymField;
        npsi::Integer = length(eq.psi1d)
    )
    M3DC1Reader._assert_native(eq)
    M = M3DC1Reader.to_mxh(eq)              # MXHEquilibrium.EFITEquilibrium
    shift = eq.psi_boundary                 # to_mxh shifts ψ by -psi_boundary internally

    # uniform R/Z grids + computational box
    r = range(first(eq.R), last(eq.R); length = length(eq.R))
    z = range(first(eq.Z), last(eq.Z); length = length(eq.Z))
    nw, nh = length(r), length(z)
    rdim = Float64(last(r) - first(r)); zdim = Float64(last(z) - first(z))
    rleft = Float64(first(r)); zmid = Float64((first(z) + last(z)) / 2)

    # uniform ψ grid (physical, axis→boundary); resample F/p onto it
    psi = range(eq.psi_axis, eq.psi_boundary; length = npsi)
    F = M3DC1Reader._resample(eq.psi1d, eq.F1d, psi)
    p = eq.p1d === nothing ? zeros(npsi) : M3DC1Reader._resample(eq.psi1d, eq.p1d, psi)

    # q(ψ) via the GENERIC FSA method (EFITEquilibrium overrides safety_factor to
    # the stored profile, which to_mxh sets to zeros). M's ψ is shifted, so evaluate
    # at ψ-shift; guard the boundary endpoint (separatrix tracing can fail).
    q_fsa(ψs) = try
        invoke(
            MXHEquilibrium.safety_factor,
            Tuple{MXHEquilibrium.AbstractEquilibrium, Any}, M, ψs
        )
    catch
        NaN
    end
    ψv = collect(Float64, psi)
    qpsi = Float64[q_fsa(ψ - shift) for ψ in psi]
    # fill any non-finite q (edge/separatrix tracing hiccups) by interpolation
    if any(!isfinite, qpsi)
        fin = isfinite.(qpsi)
        count(fin) ≥ 2 && (qpsi = M3DC1Reader._resample(ψv[fin], qpsi[fin], ψv))
    end

    # ffprim = F·dF/dψ, pprime = dp/dψ (finite differences on the uniform ψ grid)
    ffprim = F .* _deriv(ψv, F)
    pprime = _deriv(ψv, p)

    # LCFS outline + normalized toroidal flux rhovn = √(Φ/Φ_boundary)
    bdry = MXHEquilibrium.plasma_boundary(M)
    rbbbs = collect(Float64, bdry.r); zbbbs = collect(Float64, bdry.z)
    Φ = Float64[
        (
                try
                    MXHEquilibrium.toroidal_flux(M, ψ - shift)
            catch
                    NaN
            end
            ) for ψ in psi
    ]
    Φb = Φ[end]
    rhovn = (isfinite(Φb) && Φb != 0) ? sqrt.(abs.(Φ ./ Φb)) :
        collect(range(0.0, 1.0; length = npsi))

    # limiter: computational-box rectangle (M3D-C1 carries no separate limiter)
    rlim = [rleft, rleft + rdim, rleft + rdim, rleft, rleft]
    zlim = [zmid - zdim / 2, zmid - zdim / 2, zmid + zdim / 2, zmid + zdim / 2, zmid - zdim / 2]

    Ip = MXHEquilibrium.plasma_current(M)

    # positional order matches EFIT.GEQDSKFile (EFIT/src/io.jl readg construction)
    return EFIT.GEQDSKFile(
        "M3DC1Reader", eq.time, nw, nh, r, z, rdim, zdim, rleft, zmid,
        length(rbbbs), rbbbs, zbbbs, length(rlim), rlim, zlim,
        eq.r0, eq.b0, eq.axis[1], eq.axis[2], eq.psi_axis, eq.psi_boundary,
        psi, Ip, F, p, ffprim, pprime, qpsi, eq.psi_rz, rhovn
    )
end

# central finite-difference dy/dx (one-sided at the ends)
function _deriv(x::Vector{Float64}, y::Vector{Float64})
    n = length(x); d = similar(y)
    d[1] = (y[2] - y[1]) / (x[2] - x[1])
    d[n] = (y[n] - y[n - 1]) / (x[n] - x[n - 1])
    @inbounds for i in 2:(n - 1)
        d[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    end
    return d
end

end # module
