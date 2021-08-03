module PPLM

using Flux
using Zygote: @adjoint
using Flux: onehotbatch
using Flux.Data: DataLoader
using Flux.Optimise: update!
using DataFrames: DataFrame
using Transformers.HuggingFace
using DataDeps
using BSON
using CUDA
using Parameters: @with_kw
using MLDataUtils
using ProgressMeter
using Pkg.Artifacts

const ARTIFACTS_TOML = joinpath(@__DIR__, "Artifacts.toml")

const SMALL_CONST = Float32(1.0e-15)

abstract type PretrainedTokenizer end

include("artifacts.jl")
include("bow.jl")
include("discriminator.jl")
include("data_preprocess.jl")
include("discrim_train.jl")
include("tokenizer_datadeps.jl")
include("tokenizer.jl")
include("generate.jl")
include("utils.jl")
include("pplm.jl")

export load_pretrained_tokenizer, GPT2, encode, decode, detokenize, tokenize, load_data, load_cached_data, data_preprocess, ClassifierHead, DisciminatorV1, DiscriminatorV2

"""
    get_gpt2()

Function to load gpt2 lmheadmodel along with the `tokenizer`.
"""
function get_gpt2()
    model = hgf"gpt2:lmheadmodel"
    tokenizer = load_pretrained_tokenizer(GPT2)
    return tokenizer, model
end

"""
    get_gpt2_medium()

Function to load gpt2-medium lmhead model along with the `tokenizer`.

**Note**: In case this function gives error of permission denied, try changing the file permissions for the Artifacts.toml file of Transformers.jl package (as it is read only by default) under the `src/huggingface` folder. 
"""
function get_gpt2_medium()
    model = hgf"gpt2-medium:lmheadmodel"
    tokenizer = load_pretrained_tokenizer(GPT2)
    return tokenizer, model
end

device_id=-1
device=Flux.cpu

"""
    set_device(d_id=0)

Function to set cuda device if available and also to disallow scalar operations
"""
function set_device(d_id=0)
    global device_id, device
    if CUDA.has_cuda()       # Check if CUDA is available
        device = Flux.gpu
        device_id = d_id
        CUDA.device!(d_id)
        @info "CUDA is on"
        CUDA.allowscalar(false)
    end
end

function __init__()
    init_tokenizer_datadeps()
    set_device()
end

end