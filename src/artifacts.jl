using Pkg.Artifacts

const ARTIFACTS_TOML = joinpath(@__DIR__, "Artifacts.toml")

function get_registered_file(name)
    global ARTIFACTS_TOML
    hash = artifact_hash(name, ARTIFACTS_TOML)
    isnothing(hash) && error("$name not registered.")
    artifact_path(hash)
end

"""

Example
register_custom_file('custom', 'xyz.txt','./folder/folder/')
"""
function register_custom_file(artifact_name, file_name, path)
    file_path = joinpath(path, file_name)
    isfile(file_path) || error("Can't register to $artifact_name: file $file_path not found.")
    filegetter = function (dest)
        cp(file_path, dest)
    end
   
    file_hash = create_artifact() do artifact_dir
        filegetter(joinpath(artifact_dir, file_name))
    end

    # register if no in artifact.toml
    if isnothing(artifact_hash(artifact_name, ARTIFACTS_TOML))
        bind_artifact!(ARTIFACTS_TOML, artifact_name, file_hash)
    end
end

