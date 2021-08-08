using CUDA
const pretrained_folder = "./pretrained_discriminators"

# Dictionary for pretrained discriminator paths
pretrained = Dict(
            "clickbait" => "clickbait_classifier_head_1024.bson",
            "toxicity"  => "toxicity_classifier_head_768.bson",
            "sentiment" => "sentiment_classifier_head_1024.bson"
        )

"""
struct ClassifierHead
    linear_layer::Dense
    embed_size::Int
    class_size::Int
end

Struct for ClassifiedHead, defined with a single linear layer and two paramters: embed_size-> Size of Embedding, class_size->Number of classes.

"""
struct ClassifierHead
    linear_layer::Dense
    embed_size::Int
    class_size::Int
end

"""
    ClassifierHead(class_size::Int=1, embed_size::Int=768; load_from_pretrained=false, discrim=nothing, file_name=nothing, path=nothing)

Function to initiate ClassifierHead layer, with defined class_size and embed_size. If `load_from_pretrained` is set to be true, load model from pretrained models (based on `discrim` name) or from specified `path`. 
"""
function ClassifierHead(class_size::Int=1, embed_size::Int=768; load_from_pretrained=false, discrim=nothing, file_name=nothing, path=nothing)
    global pretrained_folder
    if load_from_pretrained==true
        if discrim==nothing
            error("Discriminator not provided")
        else
            if discrim in keys(pretrained)
                path_f = joinpath(@__DIR__, pretrained_folder, pretrained[discrim])
            else 
                print("Discriminator $discrim not in pretrained discriminators, checking if `path` provided")
                if isnothing(path)
                    error("`path` not provided for custom discriminator bson file. Try `ClassifierHead(...; path=...)` or `getDiscriminator(...; ... , path=... )`")
                else
                    path_f = path
                end
            end
            BSON.@load path_f config_metadata weights
            layer = Dense(weights["weight"], weights["bias"])
            class_size = length(weights["bias"])
        end
    else
        layer = Dense(embed_size, class_size)
        config_metadata=nothing
    end
    ClassifierHead(layer, embed_size, class_size), config_metadata
end

Flux.@functor ClassifierHead (linear_layer,)
 
"""
    (m::ClassifierHead)(hidden_state)

Method for passing hidden state through Linear layer of ClassifierHead `m`, return the final logits.

"""
function (m::ClassifierHead)(hidden_state)
    lm_logits = m.linear_layer(hidden_state)
    lm_logits
end

"""
struct DiscriminatorV1
    cl_head::ClassifierHead
    model
    embed_size::Int
end

Struct to contain the model and ClassifierHead of PPLM model, to be used in masked training.

"""
struct DiscriminatorV1
    cl_head::ClassifierHead
    model
    embed_size::Int
end

Flux.@functor DiscriminatorV1 (cl_head,) 

"""
    avg_representation(m::DiscriminatorV1, input_ids; args=nothing)

Function to create average representation (without mask) of Input tokens `input_ids`, using the DiscriminatorV1 `m`.

"""
function avg_representation(m::DiscriminatorV1, input_ids; args=nothing)
    if args!=nothing && args.cached == true
        return input_ids
    else
        outputs = m.model(input_ids; output_attentions=false,
                    output_hidden_states=true, use_cache=false)
        hidden = output.hidden_states
        hidden = sum(hidden, dims = 2)
        hidden_ = dropdims(hidden, dims=2)
        return hidden_
    end
end

# DiscriminatorV1 -> without mask on hidden_state
function (m::DiscriminatorV1)(x; args=nothing)
    hidden_state = avg_representation(m, x; args)
    logits = m.cl_head(hidden_state) 
    logits, dims=1
end

"""
struct DiscriminatorV2
    cl_head::ClassifierHead
    model
    embed_size::Int
end

Struct to contain the model and ClassifierHead of PPLM model, to be used in masked training.

"""
struct DiscriminatorV2
    cl_head::ClassifierHead
    model
    embed_size::Int
end

Flux.@functor DiscriminatorV2 (cl_head,)

"""
    avg_representation(m::DiscriminatorV2, input_ids, mask; args=nothing)

Function to create masked average representation of Input tokens `input_ids`, using the DiscriminatorV2 `m`.

"""
function avg_representation(m::DiscriminatorV2, input_ids, mask; args=nothing)
    if args!=nothing && args.cached == true
        return input_ids
    else
        outputs = m.model(input_ids; output_attentions=false,
                    output_hidden_states=true, use_cache=false)
        hidden = outputs.hidden_states[end]
        hidden.*= mask
        hidden = sum(hidden, dims = 2) ./ sum(mask, dims=2)
        hidden_ = dropdims(hidden, dims=2)
        return hidden_ 
    end
end

function (m::DiscriminatorV2)(x, mask; args=nothing)
    hidden_state = avg_representation(m, x, mask; args)
    logits = m.cl_head(hidden_state) 
    logits
end

"""
    get_discriminator(model; load_from_pretrained=false, discrim=nothing, file_name=nothing, version=2, class_size::Int=1, embed_size::Int=768, path=nothing)

Function to create discriminator based on provided model. Incase, `load_from_pretrained` is set to be true, loads ClassifierHead layer from pretrained models or `path` provided.

"""
function get_discriminator(model; load_from_pretrained=false, discrim=nothing, file_name=nothing, version=2, class_size::Int=1, embed_size::Int=768, path=nothing)
    global device
    model = model |> device 
    cl_head, _ = ClassifierHead(class_size, embed_size; load_from_pretrained=load_from_pretrained, discrim=discrim, file_name=file_name, path=path) |> gpu
    if version == 1
        return DiscriminatorV1(cl_head, model, embed_size)
    else
        return DiscriminatorV2(cl_head, model, embed_size)
    end
end

"""
    save_classifier_head(cl_head; file_name=nothing, path=nothing, args=nothing, register_discrim=true, discrim_name="")

Function to save the ClassifiedHead as a BSON once the training is complete, based on the path provided. In case path is set as nothing, it saves the discriminators in `./pretrained_discriminators` folder relative to the package directory.
"""
function save_classifier_head(cl_head; file_name=nothing, path=nothing, args=nothing, register_discrim=true, discrim_name="")
    if path == nothing
        joinpath(@__DIR__,"./pretrained_discriminators")
    end
    if file_name == nothing
        file_name = string("custom_classifier_head_", cl_head.embed_size, "_.jl")
    end
    
    cl_head = cl_head |> cpu
    save_path = joinpath(path, file_name)
    config_metadata = args
    weights = Dict("weight"=> cl_head.linear_layer.W, "bias"=>cl_head.linear_layer.bias)
    if args==nothing
        @warn "Saving classifier without Hyperparameter information"
        config_metadata = Dict()
        config_metadata["embed_size"]=size(weights["weight"])[2]
        config_metadata["class_size"]=size(weights["weight"])[1]
        config_metadata["default_class"]=1
    else
        config_metadata = Dict()
        config_metadata=args
    end
    
    BSON.@save save_path config_metadata weights
end

"""
    save_discriminator(discrim, discrim_name="Custom"; file_name=nothing, path=nothing, args=nothing)

Function to save ClassifiedHead part of discriminator (by calling `save_classifier_head` function), which is the only trainable part of discriminator
"""
function save_discriminator(discrim, discrim_name="Custom"; file_name=nothing, path=nothing, args=nothing)
    if path == nothing
        path = joinpath(@__DIR__, "./pretrained_discriminators")
    end
    if file_name == nothing
        file_name = string("custom_classifier_head_", discrim.embed_size, "_.jl")
    end
    save_path = joinpath(path, file_name)
    println("Saving classifier head weights for the discriminator to $save_path")
    save_classifier_head(discrim.cl_head; file_name=file_name, path=path, args=args, discrim_name=discrim_name)
    print("Discriminator saved successfully")
end
