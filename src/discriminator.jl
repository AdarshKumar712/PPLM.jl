struct ClassifierHead
    linear_layer::Dense
    embed_size::Int
    class_size::Int
end

function ClassifierHead(class_size::Int=1, embed_size::Int=768; load_from_pretrained=false, discrim=nothing)
    if load_from_pretrained==true
        if discrim==nothing
            error("Discriminator not provided")
        else
            path = get_registered_file(discrim)
            BSON.@load path config_metadata weights
            layer = Dense(weights["weight"], weights["bias"])
        end
    else
        layer = Dense(embed_size, class_size)
    end
    ClassifierHead(layer, embed_size, class_size)
end

Flux.@functor ClassifierHead (linear_layer,)
 
function (m::ClassifierHead)(hidden_state)
    lm_logits = m.linear_layer(hidden_state)
    lm_logits
end

struct DiscriminatorV1
    cl_head::ClassifierHead
    model
    embed_size::Int
end

Flux.@functor DiscriminatorV1 (cl_head,) 

function avg_representation(m::DiscriminatorV1, input_ids; args=nothing)
    if args!=nothing && args.cached == true
        return data
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

struct DiscriminatorV2
    cl_head::ClassifierHead
    model
    embed_size::Int
end

Flux.@functor DiscriminatorV2 (cl_head,)

function avg_representation(m::DiscriminatorV2, input_ids, mask; args=nothing)
    if args!=nothing && args.cached == true
        return data
    else
        outputs = m.model(input_ids; output_attentions=false,
                    output_hidden_states=true, use_cache=false)
        hidden = output.hidden_states
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

function getDiscriminator(model; load_from_pretrained=false, discrim=nothing, version=2, class_size::Int=1, embed_size::Int=768)
    cl_head = ClassifierHead(class_size, embed_size; load_from_pretrained=load_from_pretrained, discrim=discrim)
    if version == 1
        return DiscriminatorV1(cl_head, model, embed_size)
    else
        return DiscriminatorV2(cl_head, model, embed_size)
    end
end

"""

"""
function save_classifier_head(cl_head; file_name="custom_classifier_head.bson", path="./pretrained_discriminators", args=nothing, register_discrim=true, discrim_name="")
    save_path = joinpath(path, file_name)
    config_metadata = args
    weights = Dict("weight"=> cl_head.linear_layer.W, "bias"=>cl_head.linear_layer.bias)
    if args==nothing
        @warn "Saving classifier without Hyperparameter information"
        config_metadata = Dict()
        config_metadata["embed_size"]=size(weights["weight"])[2]
        config_metadata["class_size"]=size(weights["weight"])[1]
        config_metadata["path"] = save_path
    end
    BSON.@save save_path config_metadata weights
    
    if register_discrim==true
        if discrim_name == ""
            discrim_name=file_name
        end
        register_custom_file(discrim_name, file_name, path)
    end
end


function save_discriminator(discrim; file_name="Custom_discrim.bson", path="./pretrained_discriminators", args=nothing, register_discrim=true, discrim_name="")
    save_path = joinpath(path, file_name)
    println("Saving classifier head weights for the discriminator to $save_path")
    save_classifier_head(discrim.cl_head; file_name=file_name, path=path, args=args, register_discrim=register_discrim, discrim_name=discrim_name)
    print("Discriminator saved successfully")
end
