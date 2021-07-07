module PPLM

using Flux
using Zygote: @adjoint
using Flux: onehotbatch
using Flux.Data: DataLoader
using DataFrames: DataFrame
using DataDeps
using BSON
using CUDA
using Pkg.Artifacts

const ARTIFACTS_TOML = joinpath(@__DIR__, "Artifacts.toml")

const SMALL_CONST = Float32(1.0e-15)

abstract type PretrainedTokenizer end

include("artifacts.jl")
include("bow.jl")
include("discriminator.jl")
include("data_preprocess.jl")
#include("discrim_train.jl")
include("tokenizer_datadeps.jl")
include("tokenizer.jl")
#include("generate.jl")
include("utils.jl")
#include("pplm.jl")

export load_pretrained_tokenizer, GPT2, encode, decode, detokenize, tokenize, load_data, load_cached_data, data_preprocess, ClassifierHead, DisciminatorV1, DiscriminatorV2

function __init__()
    init_tokenizer_datadeps()
end

function get_gpt2()
    model = hgf"gpt2:lmheadmodel"
    tokenizer = load_pretrained_tokenizer(GPT2)
    return tokenizer, model
end

device_id=-1
device=Flux.cpu

function set_device(d_id)
    if CUDA.has_cuda()
        device = Flux.gpu
        device_id = d_id
        CUDA.device!(d_id)
    end
end

end