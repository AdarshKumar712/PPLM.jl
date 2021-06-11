struct ClassifierHead
    linear_layer::Dense
    embed_size::Int
    class_size::Int
end

function ClassifierHead(class_size::Int=1, embed_size::Int=768)
    layer = Dense(embed_size, class_size)
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

function getDiscriminator(model; version=2, class_size::Int=1, embed_size::Int=768)
    cl_head = ClassifierHead(class_size, embed_size)
    if version == 1
        return DiscriminatorV1(cl_head, model, embed_size)
    else
        return DiscriminatorV2(cl_head, model, embed_size)
    end
end

## TODO Load_classifier_head, load_discriminator, save_discriminator, save_classifier_head

