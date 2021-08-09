using Pkg.GitTools

isurl(path::AbstractString) = startswith(path, "http://") || startswith(path, "https://")

"""
    get_registered_file(name)

Fetch registered file path from Artifacts.toml, based on the artifact `name`.  

"""
function get_registered_file(name)
    global ARTIFACTS_TOML
    hash = artifact_hash(name, ARTIFACTS_TOML)
    isnothing(hash) && error("$name not registered.")
    if !artifact_exists(hash)
        println("Artifact doesn't exist. Downloading....")
        get_artifact(name)
        hash = artifact_hash(name, ARTIFACTS_TOML)
    end
    artifact_path(hash)
end

"""
    get_artifact(name)

Utility function to download/install the artifact in case not already installed.
"""
function get_artifact(name)
    meta = Artifacts.artifact_meta(name, ARTIFACTS_TOML)
    file_path = meta["download"][1]["url"]
    if isurl(file_path)
        ensure_artifact_installed(name, ARTIFACTS_TOML)
    else
        file_name = split(file_path, "/")[end]
        path = joinpath(@__DIR__, file_path[1:end-length(file_name)])
        register_custom_file(name, file_name, path)
    end
end

"""
    register_custom_file(artifact_name, file_name, path)

Function to register custom file under `artifact_name` in Artifacts.toml. `path` expects path of the directory where the file `file_name` is stored. Stores the complete path to the file as Artifact URL.

# Example

```
register_custom_file('custom', 'xyz.txt','./folder/folder/')

```
Note: In case this gives permission denied error, change the Artifacts.toml file permissions using 
`chmod(path_to_file_in_julia_installation , 0o764)`or similar.
"""
function register_custom_file(artifact_name, file_name, path)
    file_path = joinpath(path, file_name)
    isfile(file_path) || error("Can't register to $artifact_name: file $file_path not found.")
    filegetter = function (dest)
        cp(file_path, dest)
    end
   
    file_hash = create_artifact() do artifact_dir
        tarball = filegetter(joinpath(artifact_dir, file_name))
        global tarball_hash = bytes2hex(GitTools.blob_hash(tarball))
    end

    # register if not in artifact.toml
    if isnothing(artifact_hash(artifact_name, ARTIFACTS_TOML))
        bind_artifact!(ARTIFACTS_TOML, artifact_name, file_hash; download_info=[(file_path, tarball_hash)])
    end
end
