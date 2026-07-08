#!/usr/bin/env julia
# Quick equilibrium viewer for the IMAS HDF5 written by export_imas / export_run.
# For each `equilibrium.time_slice`, it draws the poloidal ψ map with the LCFS
# picked out in bold, plus the magnetic axis (O-point), the recomputed X-point(s)
# and the wall/limiter outline. With more than one slice it also stitches the
# frames into an MP4 (default; GIF optional) so the time evolution plays back at
# a glance.
#
# It colours by NORMALIZED flux ψ_N = (ψ − ψ_axis)/(ψ_bnd − ψ_axis): the LCFS is
# then simply the ψ_N = 1 contour, independent of the COCOS convention the file
# was written in (ψ, ψ_axis and ψ_bnd all come from the same slice, so any
# rescaling cancels).
#
# Usage (runs in your DEFAULT Julia env, which must have Plots + HDF5 — NOT the
# package project, which has no plotting deps):
#
#   julia scripts/plot_equilibrium.jl <imas.h5 | folder> [--key=value ...]
#   scripts/plot_equilibrium         <imas.h5 | folder> [--key=value ...]   # wrapper
#
# The only positional argument is the input: the M3DC1Reader IMAS HDF5 file, or a
# folder containing `M3DC1Reader_imas.h5`. Options are all --key=value:
#
#   --outdir=<path>   where to write the PNGs/video (default: <input folder>/equilibrium_plots)
#   --video=true      also write a time-evolution movie when >1 slice (default: true)
#   --format=mp4      movie container: mp4 (default, compact + smooth) | gif
#   --fps=8           movie frame rate (default: 8)
#   --slices=0,2,5    equilibrium time_slice indices to plot (0-based; default: all)
#   --field=psi_norm  color field: psi_norm (default) | psi (raw, per-slice ψ)
#
# Examples:
#   julia scripts/plot_equilibrium.jl /scratch/out
#   julia scripts/plot_equilibrium.jl /scratch/out/M3DC1Reader_imas.h5 --fps=12
#   julia scripts/plot_equilibrium.jl /scratch/out --slices=0,12,24 --video=false
#   julia scripts/plot_equilibrium.jl /scratch/out --format=gif

using HDF5
using Printf: @sprintf
using Plots

# ── reading ────────────────────────────────────────────────────────────────
# The OMAS writer stores dotted IMAS paths as nested groups and transposes N-D
# arrays for row-major h5py; read back through HDF5.jl (column-major) a Julia
# (nR,nZ) map returns as (nZ,nR) — exactly what heatmap(R, Z, ψ) wants. Scalars
# are 0-dim datasets, so `read(...)[]`-style access via `_scalar` handles both.
_scalar(g, path) = (v = read(g[path]); v isa AbstractArray ? first(v) : v)

"Wall/limiter outline (R,Z), closed into a loop; `nothing` if the file has none."
function read_wall(f)
    haskey(f, "wall") || return nothing
    p = "wall/description_2d/0/limiter/unit/0/outline"
    haskey(f, "$p/r") || return nothing
    r = read(f["$p/r"]);  z = read(f["$p/z"])
    (isempty(r) || length(r) != length(z)) && return nothing
    return (R = vcat(r, r[1]), Z = vcat(z, z[1]))     # close the polygon
end

"Everything needed to draw one equilibrium slice."
function read_slice(f, key)
    g = f["equilibrium/time_slice/$key"]
    R = read(g["profiles_2d/0/grid/dim1"])
    Z = read(g["profiles_2d/0/grid/dim2"])
    psi = read(g["profiles_2d/0/psi"])
    # Expect (nZ, nR); guard the (rare) non-square transposed case.
    if size(psi) == (length(R), length(Z)) && length(R) != length(Z)
        psi = permutedims(psi)
    end
    ψa = _scalar(g, "global_quantities/psi_axis")
    ψb = _scalar(g, "global_quantities/psi_boundary")
    axis = (R = _scalar(g, "global_quantities/magnetic_axis/r"),
        Z = _scalar(g, "global_quantities/magnetic_axis/z"))
    xpts = NTuple{2, Float64}[]
    if haskey(g, "boundary/x_point")
        xg = g["boundary/x_point"]
        for k in sort(collect(keys(xg)); by = s -> parse(Int, s))
            push!(xpts, (_scalar(xg, "$k/r"), _scalar(xg, "$k/z")))
        end
    end
    return (R = R, Z = Z, psi = psi, psi_axis = ψa, psi_boundary = ψb,
        axis = axis, xpoints = xpts, time = _scalar(g, "time"))
end

# ── plotting ─────────────────────────────────────────────────────────────────
function plot_slice(s, wall; idx, field, xlims, ylims, clims, ctitle)
    if field == :psi
        C = s.psi
        lcfs = s.psi_boundary                        # LCFS at raw ψ_bnd
        lines = range(s.psi_axis, s.psi_boundary, length = 10)[2:(end - 1)]
    else
        span = s.psi_boundary - s.psi_axis
        C = (s.psi .- s.psi_axis) ./ span            # ψ_N: 0 at axis, 1 at LCFS
        lcfs = 1.0
        lines = 0.1:0.1:0.9
    end

    plt = heatmap(s.R, s.Z, C;
        c = :viridis, clims = clims, colorbar_title = ctitle,
        aspect_ratio = :equal, xlims = xlims, ylims = ylims,
        xlabel = "R [m]", ylabel = "Z [m]",
        title = @sprintf("slice %d    t = %.3f ms", idx, s.time * 1.0e3),
        titlefontsize = 10, size = (620, 720), framestyle = :box,
        background_color = :white)

    # nested flux surfaces (thin, faint) then the LCFS in bold red. The LCFS gets
    # a black casing under the red line so it stays legible over both the dark
    # core and the bright scrape-off region.
    contour!(plt, s.R, s.Z, C; levels = collect(lines),
        c = :white, alpha = 0.35, lw = 0.6, colorbar_entry = false)
    contour!(plt, s.R, s.Z, C; levels = [lcfs],
        c = :black, lw = 4.0, colorbar_entry = false)
    contour!(plt, s.R, s.Z, C; levels = [lcfs],
        c = :red, lw = 2.0, colorbar_entry = false)

    # wall / limiter
    wall === nothing || plot!(plt, wall.R, wall.Z;
        c = :black, lw = 2, label = "wall")

    # O-point (magnetic axis) and X-point(s)
    scatter!(plt, [s.axis.R], [s.axis.Z]; marker = :circle, ms = 6,
        c = :red, markerstrokecolor = :white, markerstrokewidth = 1.5,
        label = "O-point")
    if !isempty(s.xpoints)
        scatter!(plt, first.(s.xpoints), last.(s.xpoints);
            marker = :xcross, ms = 8, c = :cyan, markerstrokewidth = 3,
            label = "X-point")
    end
    plot!(plt; legend = :topleft, legendfontsize = 7,
        foreground_color_legend = nothing, background_color_legend = RGBA(1, 1, 1, 0.6))
    return plt
end

# ── CLI ──────────────────────────────────────────────────────────────────────
function main(args)
    input = ""
    opts = Dict{String, String}()
    for a in args
        if startswith(a, "--")
            kv = a[3:end]
            if occursin('=', kv)
                k, v = split(kv, '='; limit = 2)
                opts[String(k)] = String(v)
            else
                opts[kv] = "true"
            end
        elseif isempty(input)
            input = a
        else
            error("unexpected positional '$a' — options use --key=value")
        end
    end
    isempty(input) && error(
        "usage: plot_equilibrium <imas.h5 | folder> " *
            "[--outdir= --gif=true|false --fps= --slices= --field=psi_norm|psi]"
    )

    h5 = isdir(input) ? joinpath(input, "M3DC1Reader_imas.h5") : input
    isfile(h5) || error("no IMAS HDF5 found at: $h5")
    outdir = get(opts, "outdir", joinpath(dirname(abspath(h5)), "equilibrium_plots"))
    mkpath(outdir)
    make_video = get(opts, "video", "true") in ("true", "on", "yes", "1")
    fmt = get(opts, "format", "mp4")
    fmt in ("mp4", "gif") || error("--format must be mp4 | gif, got $fmt")
    fps = parse(Int, get(opts, "fps", "8"))
    field = get(opts, "field", "psi_norm") == "psi" ? :psi : :psi_norm

    keys_all, wall, slices = h5open(h5, "r") do f
        haskey(f, "equilibrium/time_slice") ||
            error("no equilibrium.time_slice in $h5 — was this written by export_imas?")
        ks = sort(collect(keys(f["equilibrium/time_slice"])); by = s -> parse(Int, s))
        (ks, read_wall(f), [read_slice(f, k) for k in ks])
    end

    want = haskey(opts, "slices") ? Set(parse.(Int, split(opts["slices"], ','))) : nothing
    sel = [i for (i, k) in enumerate(keys_all) if want === nothing || parse(Int, k) in want]
    isempty(sel) && error("no matching slices (have indices $(join(keys_all, ',')))")

    # Common frame: union of grid extents + a fixed color scale across time.
    xlims = (minimum(s -> s.R[1], slices), maximum(s -> s.R[end], slices))
    ylims = (minimum(s -> s.Z[1], slices), maximum(s -> s.Z[end], slices))
    if field == :psi
        allψ = filter(isfinite, reduce(vcat, [vec(slices[i].psi) for i in sel]))
        clims = (minimum(allψ), maximum(allψ));  ctitle = "ψ"
    else
        # ψ_N ∈ [0,1] in the core; a bit of headroom keeps the scrape-off region a
        # visible gradient (not a flat saturated band) so the LCFS line stands out.
        clims = (0.0, 1.2);  ctitle = "ψ_N"
    end

    @info "plot_equilibrium" file = h5 slices = length(sel) outdir = outdir field = field
    pngs = String[]
    for i in sel
        s = slices[i]
        plt = plot_slice(s, wall; idx = parse(Int, keys_all[i]),
            field = field, xlims = xlims, ylims = ylims, clims = clims, ctitle = ctitle)
        png = joinpath(outdir, @sprintf("equilibrium_%03d.png", parse(Int, keys_all[i])))
        savefig(plt, png);  push!(pngs, png)
        @info "  wrote" png = basename(png) t_ms = round(s.time * 1.0e3, digits = 3)
    end

    if make_video && length(sel) > 1
        anim = @animate for i in sel
            s = slices[i]
            plot_slice(s, wall; idx = parse(Int, keys_all[i]),
                field = field, xlims = xlims, ylims = ylims, clims = clims, ctitle = ctitle)
        end
        vpath = joinpath(outdir, "equilibrium_evolution.$fmt")
        (fmt == "mp4" ? mp4 : gif)(anim, vpath; fps = fps)
        @info "  wrote movie" movie = basename(vpath) fps = fps
    end
    @info "done" pngs = length(pngs) dir = outdir
    return outdir
end

main(ARGS)
