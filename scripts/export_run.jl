#!/usr/bin/env julia
# Batch runner: point at an M3D-C1 run folder (or a C1.h5 directly) and write an
# OMAS/IMAS HDF5 next to it. The multi-GB `time_NNN.h5` slice files are read
# transparently via C1.h5's external links (resolved relative to C1.h5), so only
# the folder needs to be on disk.
#
# Usage:
#   julia --project=. scripts/export_run.jl <run_dir | C1.h5> [out.h5] [key=val ...]
#
# Positional:
#   <run_dir | C1.h5>   folder containing C1.h5, or the C1.h5 path itself
#   [out.h5]            output path (default: <run_dir>/M3DC1Reader_imas.h5)
#
# key=val options (all optional):
#   nbins=128           FSA radial bins
#   ngrid=200           R-Z evaluation grid (ngrid × ngrid)
#   cocos=11            11 (IMAS) | mhdsimdb (NIMROD-DB drop-in) | raw (per-radian)
#   fsa=cumulative      ratio-profile FSA estimator: cumulative (smooth, default) | bin (raw)
#   fsa_window=4        cumulative regression window (smaller = follows pedestal tighter)
#   pulse=<int>         dataset_description.data_entry.pulse
#   slices=0,12,24      comma-separated timeslice indices (default: all)
#   ascot5=<ts>         also write an ASCOT5 input for slice <ts> into the folder
#
# NOTE: this runner defaults to fsa=cumulative (de-noised ne/Te/dB-over-B profiles).
# The library `export_imas` default is fsa_method=:bin for backward compatibility;
# pass `fsa=bin` here to reproduce the raw per-bin estimator.
#
# Examples:
#   julia --project=. scripts/export_run.jl /scratch/run_042
#   julia --project=. scripts/export_run.jl /scratch/run_042 out.h5 nbins=256 pulse=200123
#   julia --project=. scripts/export_run.jl /scratch/run_042 slices=0,24 ascot5=24
#   julia --project=. scripts/export_run.jl /scratch/run_042 fsa=bin       # raw profiles

using M3DC1Reader

function main(args)
    isempty(args) && error(
        "usage: export_run.jl <run_dir | C1.h5> [out.h5] [nbins= ngrid= cocos= pulse= slices= ascot5=]"
    )

    # Split positionals from key=val options.
    positional = String[]
    opts = Dict{String, String}()
    for a in args
        if occursin('=', a)
            k, v = split(a, '='; limit = 2)
            opts[k] = v
        else
            push!(positional, a)
        end
    end

    input = positional[1]
    # Resolve the C1.h5 path: accept either a folder or the file itself.
    c1 = isdir(input) ? joinpath(input, "C1.h5") : input
    isfile(c1) || error("no C1.h5 found at: $c1")
    rundir = dirname(abspath(c1))
    out = length(positional) ≥ 2 ? positional[2] : joinpath(rundir, "M3DC1Reader_imas.h5")

    # Parse options with defaults matching export_imas.
    nbins = parse(Int, get(opts, "nbins", "128"))
    ngrid = parse(Int, get(opts, "ngrid", "200"))
    pulse = haskey(opts, "pulse") ? parse(Int, opts["pulse"]) : nothing
    cocos = let c = get(opts, "cocos", "11")
        c == "11" ? 11 : c == "mhdsimdb" ? :mhdsimdb :
            (c == "raw" || c == "nothing") ? nothing :
            error("cocos must be 11 | mhdsimdb | raw, got $c")
    end
    # This runner defaults to the smooth cumulative estimator (the library default
    # is :bin); pass fsa=bin to opt back into the raw per-bin profiles.
    fsa_method = let m = get(opts, "fsa", "cumulative")
        m == "cumulative" ? :cumulative : m == "bin" ? :bin :
            error("fsa must be cumulative | bin, got $m")
    end
    fsa_window = parse(Float64, get(opts, "fsa_window", "4"))

    file = M3DC1File(c1)
    all_slices = list_timeslices(file)
    slices = haskey(opts, "slices") ?
        parse.(Int, split(opts["slices"], ',')) : all_slices

    @info "M3DC1Reader export" run = rundir c1 = c1 out = out slices = length(slices) nbins ngrid cocos pulse fsa_method fsa_window

    export_imas(
        file, out;
        slices = slices, nbins = nbins, ngrid = ngrid,
        cocos = cocos, pulse = pulse,
        fsa_method = fsa_method, fsa_window = fsa_window, verbose = true
    )
    @info "IMAS export written" out

    # Optional: one ASCOT5 input file for a chosen slice (auto-named into the folder).
    if haskey(opts, "ascot5")
        ts = parse(Int, opts["ascot5"])
        a5 = write_ascot5(file, ts)
        @info "ASCOT5 input written" out = a5
    end

    return out
end

main(ARGS)
