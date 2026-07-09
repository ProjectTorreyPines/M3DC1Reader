#!/usr/bin/env julia
# Quick equilibrium + profiles viewer for the IMAS HDF5 written by export_imas /
# export_run. Each frame packs the 2D picture and the key 1D profiles side by side:
#
#   • LEFT  — poloidal ψ map (coloured by normalized flux ψ_N) with the LCFS in
#     bold, the magnetic axis (O-point), the recomputed X-point(s), the wall, and
#     — when the export carries them — the pellet / SPI-fragment (R,Z) locations.
#   • RIGHT — the main core_profiles 1D traces vs ρ_pol: nₑ, Tₑ/Tᵢ, q, and the
#     total impurity density (or δB/B when there is no impurity). A faint dashed
#     line shows the first frame as a reference so the evolution is obvious.
#
# With more than one slice it also stitches the frames into an MP4 (default; GIF
# optional) so the time evolution plays back at a glance.
#
# ψ_N = (ψ − ψ_axis)/(ψ_bnd − ψ_axis): the LCFS is then simply the ψ_N = 1
# contour, independent of the COCOS convention the file was written in (ψ, ψ_axis
# and ψ_bnd all come from the same slice, so any rescaling cancels).
#
# Usage (runs in your DEFAULT Julia env, which must have Plots + HDF5 — NOT the
# package project, which has no plotting deps):
#
#   julia scripts/plot_equilibrium.jl <imas.h5 | folder> [--key=value ...]
#   scripts/plot_equilibrium         <imas.h5 | folder> [--key=value ...]   # wrapper
#
# The only positional argument is the input: the M3DC1Reader OMAS HDF5 file, or a
# folder containing `M3DC1_axisym.h5` (the old `M3DC1Reader_imas.h5` name still
# works). Options are all --key=value:
#
#   --outdir=<path>   where to write the PNGs/video (default: <input folder>/axisym_fields_plots)
#   --video=true      also write a time-evolution movie when >1 slice (default: true)
#   --format=mp4      movie container: mp4 (default, compact + smooth) | gif
#   --fps=8           movie frame rate (default: 8)
#   --slices=0,2,5    equilibrium time_slice indices to plot (0-based; default: all)
#   --field=psi_norm  2D color field: psi_norm (default) | psi (raw, per-slice ψ)
#
# Examples:
#   julia scripts/plot_equilibrium.jl /scratch/out
#   julia scripts/plot_equilibrium.jl /scratch/out/M3DC1_axisym.h5 --fps=12
#   julia scripts/plot_equilibrium.jl /scratch/out --slices=0,12,24 --video=false

# Headless rendering + HPC env safety. These MUST be set before the `using`
# lines below, because the native libraries read them when they load:
#   • GKSwstype=100 → GR writes PNG/MP4 files but never opens an on-screen
#     window, so this runs fine over SSH / in the background / in a batch job.
#   • OMP_NUM_THREADS → libgomp (pulled in by the plotting/HDF5 native libs)
#     aborts if this is empty or non-numeric, which some HPC module environments
#     leave it as; force a valid value, but keep a valid one the user already set.
get!(ENV, "GKSwstype", "100")
occursin(r"^[1-9][0-9]*$", get(ENV, "OMP_NUM_THREADS", "")) || (ENV["OMP_NUM_THREADS"] = "1")

using HDF5
using Printf: @sprintf
using Plots

# ── reading ────────────────────────────────────────────────────────────────
# The OMAS writer stores dotted IMAS paths as nested groups and transposes N-D
# arrays for row-major h5py; read back through HDF5.jl (column-major) a Julia
# (nR,nZ) map returns as (nZ,nR) — exactly what heatmap(R, Z, ψ) wants. Scalars
# are 0-dim datasets, so `_scalar` handles both the 0-dim and 1-element cases.
_scalar(g, path) = (v = read(g[path]); v isa AbstractArray ? first(v) : v)
_rd(g, path) = haskey(g, path) ? read(g[path]) : nothing   # vector or nothing

"Wall/limiter outline (R,Z), closed into a loop; `nothing` if the file has none."
function read_wall(f)
    haskey(f, "wall") || return nothing
    p = "wall/description_2d/0/limiter/unit/0/outline"
    haskey(f, "$p/r") || return nothing
    r = read(f["$p/r"]);  z = read(f["$p/z"])
    (isempty(r) || length(r) != length(z)) && return nothing
    return (R = vcat(r, r[1]), Z = vcat(z, z[1]))     # close the polygon
end

"""
Pellet / SPI-fragment (R,Z) positions for slice `key` — one point per pellet from
`pellets/time_slice/<key>/pellet/<ip>/path_profiles/position`. Returns an empty
vector when the file predates this field (only added to exports on 2026-07-08) or
has no pellet model, so old files just plot without markers.
"""
function read_pellet_positions(f, key)
    base = "pellets/time_slice/$key/pellet"
    haskey(f, base) || return NTuple{2, Float64}[]
    pg = f[base]
    pts = NTuple{2, Float64}[]
    for k in sort(collect(keys(pg)); by = s -> parse(Int, s))
        p = "$k/path_profiles/position"
        (haskey(pg, "$p/r") && haskey(pg, "$p/z")) || continue
        r = read(pg["$p/r"]);  z = read(pg["$p/z"])
        (isempty(r) || isempty(z)) && continue
        r1 = Float64(first(r));  z1 = Float64(first(z))
        (isfinite(r1) && isfinite(z1)) || continue
        push!(pts, (r1, z1))
    end
    return pts
end

"The 1D core_profiles traces for slice `key`, vs ρ_pol; `nothing` if absent."
function read_profiles(f, key)
    haskey(f, "core_profiles/profiles_1d/$key") || return nothing
    g = f["core_profiles/profiles_1d/$key"]
    rho = _rd(g, "grid/rho_pol_norm")
    rho === nothing && return nothing
    # total impurity density: neutral state 0 + ion charge states 1..Z
    nimp = nothing
    if haskey(g, "ion") || haskey(g, "neutral/0/density")
        acc = zeros(length(rho));  any_imp = false
        if haskey(g, "ion")
            for k in keys(g["ion"])
                k == "0" && continue                 # 0 is the main (fuel) ion
                d = _rd(g["ion"], "$k/density")
                d === nothing && continue
                acc .+= ifelse.(isfinite.(d), d, 0.0);  any_imp = true
            end
        end
        d0 = _rd(g, "neutral/0/density")
        d0 === nothing || (acc .+= ifelse.(isfinite.(d0), d0, 0.0); any_imp = true)
        any_imp && (nimp = acc)
    end
    return (
        rho = rho, ne = _rd(g, "electrons/density"),
        te = _rd(g, "electrons/temperature"), ti = _rd(g, "ion/0/temperature"),
        q = _rd(g, "q"), dbb = _rd(g, "custom/deltab_over_b"), nimp = nimp,
    )
end

"Everything needed to draw one slice: the 2D equilibrium + the 1D profiles."
function read_slice(f, key)
    g = f["equilibrium/time_slice/$key"]
    R = read(g["profiles_2d/0/grid/dim1"])
    Z = read(g["profiles_2d/0/grid/dim2"])
    psi = read(g["profiles_2d/0/psi"])
    # Expect (nZ, nR); guard the (rare) non-square transposed case.
    if size(psi) == (length(R), length(Z)) && length(R) != length(Z)
        psi = permutedims(psi)
    end
    xpts = NTuple{2, Float64}[]
    if haskey(g, "boundary/x_point")
        xg = g["boundary/x_point"]
        for k in sort(collect(keys(xg)); by = s -> parse(Int, s))
            push!(xpts, (_scalar(xg, "$k/r"), _scalar(xg, "$k/z")))
        end
    end
    return (
        R = R, Z = Z, psi = psi,
        psi_axis = _scalar(g, "global_quantities/psi_axis"),
        psi_boundary = _scalar(g, "global_quantities/psi_boundary"),
        axis = (
            R = _scalar(g, "global_quantities/magnetic_axis/r"),
            Z = _scalar(g, "global_quantities/magnetic_axis/z"),
        ),
        xpoints = xpts, time = _scalar(g, "time"),
        pellets = read_pellet_positions(f, key),
        prof = read_profiles(f, key),
    )
end

# ── 1D profile panels ────────────────────────────────────────────────────────
# Candidate panels, in priority order; the first ≤4 with data are shown. Each
# series is (field-in-prof, legend-label, color, scale). Densities/temps are
# rescaled to friendly units; ρ_pol is the shared x-axis.
const PANEL_SPECS = (
    (id = :ne, ylabel = "nₑ  [10¹⁹ m⁻³]", series = ((:ne, "nₑ", :dodgerblue, 1.0e-19),)),
    (
        id = :temp, ylabel = "T  [keV]",
        series = ((:te, "Tₑ", :red, 1.0e-3), (:ti, "Tᵢ", :darkorange, 1.0e-3)),
    ),
    (id = :q, ylabel = "q", series = ((:q, "q", :seagreen, 1.0),)),
    (id = :nimp, ylabel = "n_imp  [10¹⁹ m⁻³]", series = ((:nimp, "n_imp", :purple, 1.0e-19),)),
    (id = :dbb, ylabel = "δB/B", series = ((:dbb, "δB/B", :teal, 1.0),)),
)

_series_y(prof, acc) = prof === nothing ? nothing : getproperty(prof, acc)
has_data(prof, spec) =
    any(s -> (y = _series_y(prof, s[1]); y !== nothing && any(isfinite, y)), spec.series)

"Fixed y-limits for a panel across all frames (anchored at 0 for densities/temps)."
function panel_ylims(spec, profs)
    vals = Float64[]
    for pr in profs, (acc, _, _, sc) in spec.series
        y = _series_y(pr, acc)
        y === nothing || append!(vals, filter(isfinite, y .* sc))
    end
    isempty(vals) && return nothing
    lo, hi = extrema(vals);  pad = 0.05 * (hi - lo + eps())
    lo = spec.id in (:ne, :temp, :nimp) ? min(0.0, lo) : lo - pad
    return (lo, hi + pad)
end

function panel_1d(spec, prof, ref, ylims)
    multi = length(spec.series) > 1
    p = plot(;
        xlabel = "ρ_pol", ylabel = spec.ylabel, ylims = ylims, xlims = (0, 1),
        legend = multi ? :best : false, legendfontsize = 12, titlefontsize = 12,
        labelfontsize = 12, tickfontsize = 10, framestyle = :box
    )
    for (acc, sub, col, sc) in spec.series
        yr = _series_y(ref, acc)
        yr === nothing || plot!(
            p, ref.rho, yr .* sc;         # faint first-frame ref
            c = col, alpha = 0.5, ls = :dash, lw = 1, label = ""
        )
        y = _series_y(prof, acc)
        y === nothing || plot!(
            p, prof.rho, y .* sc;          # current frame
            c = col, lw = 2, label = multi ? sub : ""
        )
    end
    return p
end

# ── 2D equilibrium panel ──────────────────────────────────────────────────────
# `trail` is the pellet (R,Z) positions from earlier slices (oldest first); drawn
# as a fading comet-tail under the current markers so the motion is visible.
function panel_2d(s, wall; field, xlims, ylims, clims, ctitle, title = "", trail = Vector{NTuple{2, Float64}}[])
    if field == :psi
        C = s.psi
        lcfs = s.psi_boundary
        lines = range(s.psi_axis, s.psi_boundary, length = 10)[2:(end - 1)]
    else
        C = (s.psi .- s.psi_axis) ./ (s.psi_boundary - s.psi_axis)
        lcfs = 1.0
        lines = 0.1:0.1:0.9
    end

    plt = heatmap(
        s.R, s.Z, C;
        c = :viridis, clims = clims, colorbar_title = ctitle,
        aspect_ratio = :equal, xlims = xlims, ylims = ylims,
        xlabel = "R [m]", ylabel = "Z [m]", title = title, titlefontsize = 12,
        labelfontsize = 12, tickfontsize = 10, framestyle = :box
    )

    # nested flux surfaces (thin, faint), then the LCFS in bold red over a black
    # casing so it reads over both the dark core and the bright scrape-off region.
    contour!(
        plt, s.R, s.Z, C; levels = collect(lines),
        c = :white, alpha = 0.35, lw = 0.6, colorbar_entry = false
    )
    contour!(plt, s.R, s.Z, C; levels = [lcfs], c = :black, lw = 4.0, colorbar_entry = false)
    contour!(plt, s.R, s.Z, C; levels = [lcfs], c = :red, lw = 2.0, colorbar_entry = false)

    wall === nothing || plot!(plt, wall.R, wall.Z; c = :black, lw = 2, label = "wall")
    scatter!(
        plt, [s.axis.R], [s.axis.Z]; marker = :circle, ms = 6, c = :red,
        markerstrokecolor = :white, markerstrokewidth = 1.5, label = "O-point"
    )
    isempty(s.xpoints) || scatter!(
        plt, first.(s.xpoints), last.(s.xpoints);
        marker = :xcross, ms = 8, c = :cyan, markerstrokewidth = 3, label = "X-point"
    )
    # fading comet-tail: pellet positions from earlier slices, older = fainter, so
    # the (sampled) trajectory reads at a glance under the current markers. Split the
    # alpha budget across the fragment count so a dense SPI cloud (many overlapping
    # points) doesn't stack into a saturated blob — a lone pellet keeps the full
    # ~0.4, and it's floored so very large clouds don't vanish entirely.
    nt = length(trail)
    npel = maximum(length, trail; init = 1)            # SPI fragments per slice (≈ constant)
    amax = clamp(0.4 / sqrt(npel), 0.1, 0.4)
    for (j, past) in enumerate(trail)
        isempty(past) && continue
        a = amax * (0.3 + 0.7 * (j / max(nt, 1)))      # ramp toward the present
        scatter!(
            plt, first.(past), last.(past);
            marker = :circle, ms = 2.2, c = :magenta, alpha = a,
            markerstrokewidth = 0, label = "", colorbar_entry = false
        )
    end
    # pellet / SPI-fragment locations at this slice (magenta stands out on viridis);
    # many fragments for SPI, so small markers with a thin dark edge for definition.
    isempty(s.pellets) || scatter!(
        plt, first.(s.pellets), last.(s.pellets);
        marker = :diamond, ms = 3.5, c = :magenta,
        markerstrokecolor = :black, markerstrokewidth = 0.4,
        label = length(s.pellets) == 1 ? "pellet" : "SPI ($(length(s.pellets)))"
    )
    plot!(
        plt; legend = :topleft, legendfontsize = 6,
        foreground_color_legend = nothing, background_color_legend = RGBA(1, 1, 1, 0.6)
    )
    return plt
end

# ── one composed frame (2D + 1D panels) ───────────────────────────────────────
function plot_frame(s, ref, wall, specs, pylims; idx, field, xlims, ylims, clims, ctitle, reftime = nothing, trail = Vector{NTuple{2, Float64}}[])
    base = @sprintf("slice %d      t = %.3f ms", idx, s.time * 1.0e3)
    isempty(specs) &&                                  # no core_profiles → 2D only
        return panel_2d(s, wall; field, xlims, ylims, clims, ctitle, title = base, trail = trail)

    # The dashed curve in every 1D panel is the first frame — note it in the title
    # (only present when there is more than one slice).
    suptitle = reftime === nothing ? base :
        base * @sprintf("        (dashed = initial, t = %.3f ms)", reftime * 1.0e3)
    p2d = panel_2d(s, wall; field, xlims, ylims, clims, ctitle, trail = trail)
    p1d = [panel_1d(specs[k], s.prof, ref, pylims[k]) for k in eachindex(specs)]
    n = length(specs)
    # The 2D map is aspect-locked (tall + narrow), so it only needs a slim column;
    # a small left fraction hands the extra width to the 1D profiles' x-axes.
    l = n == 4 ? (@layout [a{0.36w} grid(2, 2)]) :
        n == 3 ? (@layout [a{0.34w} grid(3, 1)]) :
        n == 2 ? (@layout [a{0.42w} grid(2, 1)]) :
        (@layout [a{0.5w} b])
    return plot(
        p2d, p1d...; layout = l, size = (1500, 780),
        plot_title = suptitle, plot_titlefontsize = 11,
        left_margin = 3Plots.mm, bottom_margin = 3Plots.mm
    )
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
            "[--outdir= --video=true|false --format=mp4|gif --fps= --slices= --field=psi_norm|psi --trail=true|false]"
    )

    # A folder resolves to the default export name (fall back to the old name so
    # files exported before the rename still plot).
    h5 = if !isdir(input)
        input
    else
        cand = joinpath(input, "M3DC1_axisym.h5")
        isfile(cand) ? cand : joinpath(input, "M3DC1Reader_imas.h5")
    end
    isfile(h5) || error("no OMAS/IMAS HDF5 found in: $input")
    outdir = get(opts, "outdir", joinpath(dirname(abspath(h5)), "axisym_fields_plots"))
    mkpath(outdir)
    make_video = get(opts, "video", "true") in ("true", "on", "yes", "1")
    fmt = get(opts, "format", "mp4")
    fmt in ("mp4", "gif") || error("--format must be mp4 | gif, got $fmt")
    fps = parse(Int, get(opts, "fps", "8"))
    field = get(opts, "field", "psi_norm") == "psi" ? :psi : :psi_norm
    # draw each pellet's earlier-slice positions as a fading comet-tail (on by default)
    show_trail = get(opts, "trail", "true") in ("true", "on", "yes", "1")

    keys_all, wall, slices = h5open(h5, "r") do f
        haskey(f, "equilibrium/time_slice") ||
            error("no equilibrium.time_slice in $h5 — was this written by export_imas?")
        ks = sort(collect(keys(f["equilibrium/time_slice"])); by = s -> parse(Int, s))
        (ks, read_wall(f), [read_slice(f, k) for k in ks])
    end

    want = haskey(opts, "slices") ? Set(parse.(Int, split(opts["slices"], ','))) : nothing
    sel = [i for (i, k) in enumerate(keys_all) if want === nothing || parse(Int, k) in want]
    isempty(sel) && error("no matching slices (have indices $(join(keys_all, ',')))")

    # Common frame across time: union of grid extents + a fixed 2D color scale.
    xlims = (minimum(s -> s.R[1], slices), maximum(s -> s.R[end], slices))
    ylims = (minimum(s -> s.Z[1], slices), maximum(s -> s.Z[end], slices))
    if field == :psi
        allψ = filter(isfinite, reduce(vcat, [vec(slices[i].psi) for i in sel]))
        clims = (minimum(allψ), maximum(allψ));  ctitle = "ψ"
    else
        # ψ_N ∈ [0,1] in the core; headroom keeps the scrape-off region a visible
        # gradient (not a flat saturated band) so the LCFS line stands out.
        clims = (0.0, 1.2);  ctitle = "ψ_N"
    end

    # Pick the 1D panels (first ≤4 with data) and their fixed y-limits; the faint
    # reference curve is the first selected slice.
    profs = [slices[i].prof for i in sel]
    specs = [sp for sp in PANEL_SPECS if has_data(first(profs), sp)]
    length(specs) > 4 && (specs = specs[1:4])
    pylims = [panel_ylims(sp, profs) for sp in specs]
    ref = length(sel) > 1 ? first(profs) : nothing

    @info "plot_equilibrium: $(length(sel)) slices, field=$field, panels=[$(join([sp.id for sp in specs], ", "))] → $outdir"
    reftime = ref === nothing ? nothing : slices[first(sel)].time
    # frame k renders selected slice sel[k], trailed by the pellet positions of all
    # earlier selected slices (the fading comet-tail) when --trail is on.
    function frame(k)
        i = sel[k]
        trail = show_trail ? [slices[sel[m]].pellets for m in 1:(k - 1)] :
            Vector{NTuple{2, Float64}}[]
        return plot_frame(
            slices[i], ref, wall, specs, pylims;
            idx = parse(Int, keys_all[i]), field = field,
            xlims = xlims, ylims = ylims, clims = clims, ctitle = ctitle,
            reftime = reftime, trail = trail
        )
    end

    pngs = String[]
    for (k, i) in enumerate(sel)
        png = joinpath(outdir, @sprintf("axisym_fields_%03d.png", parse(Int, keys_all[i])))
        savefig(frame(k), png);  push!(pngs, png)
        # one compact line per slice (message-only @info stays single-line)
        @info @sprintf("  [%d/%d] %s  t=%.3f ms", k, length(sel), basename(png), slices[i].time * 1.0e3)
    end

    if make_video && length(sel) > 1
        anim = @animate for k in eachindex(sel)
            frame(k)
        end
        vpath = joinpath(outdir, "axisym_fields_evolution.$fmt")
        (fmt == "mp4" ? mp4 : gif)(anim, vpath; fps = fps)   # Plots logs "Saved animation to …"
    end
    @info @sprintf(
        "done: %d PNG%s%s → %s", length(pngs), length(pngs) == 1 ? "" : "s",
        (make_video && length(sel) > 1) ? " + movie" : "", outdir
    )
    return outdir
end

main(ARGS)
