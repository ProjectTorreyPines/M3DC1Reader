# Main test entry point — TestItemRunner statically discovers every @testitem /
# @testsnippet under the package tree and runs each item in its own isolated
# module (`using M3DC1Reader` + `using Test` auto-injected). Shared fixtures live
# in test/setup.jl as @testsnippets, opted into via `setup = [Name]`.
#
# ARGS-based filter: pass a testitem NAME or FILENAME substring, or a `re:` regex.
# Examples (via cc-julia-test-runner):
#   cc-julia-test-runner . deltab                 # items whose name/file matches "deltab"
#   cc-julia-test-runner . test_export_ir         # by filename
#   cc-julia-test-runner . "reduce_axisym_slice"  # by testitem name
#   cc-julia-test-runner . "re:^assemble_ir"      # regex on name or filename
#
# Real-data (C1.h5) items self-skip unless M3DC1_TEST_FILE (or the default path)
# exists; IMAS/IMASdd items self-skip unless those weakdeps resolve in this env.
using TestItemRunner

@run_package_tests verbose = true filter = ti -> begin
    isempty(ARGS) && return true
    return any(ARGS) do arg
        p = startswith(arg, "re:") ? Regex(chopprefix(arg, "re:")) : arg
        return occursin(p, ti.name) || occursin(p, string(ti.filename))
    end
end
