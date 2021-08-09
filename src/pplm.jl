using Flux.Optimise:ADAM, update!
using Zygote

"""
@with_kw struct pplm
    method::String="BoW"
    perturb::String="hidden"     # hidden or past -> hidden support BoW only without gradient based change
    bow_list::Vector{String}=["military"]
    discrim::String="toxicity"
    embed_size::Int=768
    target_class_id=-1
    file_name::String=""
    path::String=""
    stepsize::Float32=0.01      
    max_length::Int=100
    num_iterations::Int=2        # more the number of iterations, more updates, more time to update
    top_k::Int=50
    top_p::Float32=0.8
    temperature::Float32=1.1
    fusion_gm_scale::Float32=0.9
    fusion_kl_scale::Float32=0.01
    cuda::Tuple{Bool, Int64}=(CUDA.has_cuda(), device_id)
    window_length::Int=0          # window length 0 corresponds to infinite length
    gamma::Float32=1.5
end

Struct to contain all the hyperparameters required for PPLM based generation.

Example:

```julia
# bow model
args = pplm(method="BoW", bow_list=["legal"], num_iterations=3)

# discriminator model
args = pplm(method="Discrim", perturb="past", stepsize=0.02)
```

"""
@with_kw struct pplm
    method::String="BoW"
    perturb::String="hidden"     # hidden or past -> hidden support BoW only without gradient based change
    bow_list::Vector{String}=["military"]
    discrim::String="toxicity"
    embed_size::Int=768
    target_class_id=-1
    file_name::String=""
    path::String=""
    stepsize::Float32=0.01      
    max_length::Int=100
    num_iterations::Int=2        # more the number of iterations, more updates, more time to update
    top_k::Int=50
    top_p::Float32=0.8
    temperature::Float32=1.1
    fusion_gm_scale::Float32=0.9
    fusion_kl_scale::Float32=0.01
    cuda::Tuple{Bool, Int64}=(CUDA.has_cuda(), device_id)
    window_length::Int=0          # window length 0 corresponds to infinite length
    gamma::Float32=1.5
end

# this is gradient based perturbation technique, supports only BoW
"""
    perturb_probs(probs, tokenizer, args)

Perturb probabilities `probs` based on provided Bag of Words list (as given with `args`). This function is supported only for BoW model. 
"""
function perturb_probs(probs, tokenizer, args)
    global device
    opt=ADAM(args.stepsize)
    bow_indices, _ = PPLM.get_bow_indices(args.bow_list, tokenizer)
    _, bow_ohe = PPLM.build_bow_ohe(bow_indices, tokenizer) 
    bow_all = bow_ohe[1]
    for i in 2:length(bow_ohe)
        bow_all = vcat(bow_all, bow_ohe)
    end
    bow_all = bow_all |> device
    
    function bow_loss(probs)
        loss_words = bow_all*probs
        loss = -log(sum(loss_words))
        loss
    end
    
    probs_ = deepcopy(probs)
    ps = params(probs_)
    kl_scale = args.fusion_kl_scale
    for i in 1:args.num_iterations
        _, back = Zygote.pullback(ps) do
               loss = bow_loss(probs_)
               loss_2 = kl_scale * Flux.kldivergence(probs_, probs)
               loss + loss_2
        end
        grads = back(1f0)
        update!(opt, ps, grads)
    end
    probs_ = max.(probs_, 0)
    
    return probs_
end

# need some abstration here
"""
    perturb_hidden_bow(hidden, model, tokenizer, args)

Perturb hidden states `hidden` based on provided Bag of Words list (as given with `args`). The perturbation is primarily based on the gradient calculated over losses evaluated over desired Bag of Words and KL Divergence from original token. 

Also checkout [`perturb_hidden_discrim`](@ref PPLM.perturb_hidden_discrim)
"""
function perturb_hidden_bow(hidden, model, tokenizer, args)
    global device
    bow_indices, _ = get_bow_indices(args.bow_list, tokenizer)
    _, bow_ohe = build_bow_ohe(bow_indices, tokenizer) 
    
    bow_all = bow_ohe[1]
    for i in 2:length(bow_ohe)
        bow_all = vcat(bow_all, bow_ohe)
    end
    bow_all = bow_all |> device
    
    kl_scale=args.fusion_kl_scale
    opt = ADAM(args.stepsize)
    p = temp_softmax(model.lm_head(hidden)[:, end, :]; t=args.temperature)
    
    function bow_loss(probs)
        loss_words = bow_all*probs
        loss = -log(sum(loss_words))
        loss
    end
    
    new_hidden = deepcopy(hidden)
    ps = params(new_hidden)
    for i in 1:args.num_iterations
        _, back = Zygote.pullback(ps) do
                logits = model.lm_head(new_hidden)[:, end, :]
                probs = temp_softmax(logits; t=args.temperature)
                loss_1 = bow_loss(probs)
                loss_2 = kl_scale * Flux.kldivergence(probs, p) 
                loss_1 + loss_2
            end
        grads = back(1f0)
        update!(opt, ps, grads)
    end
    return new_hidden 
end

"""
    perturb_past_bow(model, prev, past, original_probs, args)

Perturb past key values `prev` based on provided Bag of Words list (as given with `args`). The perturbation is primarily based on the gradient calculated over losses evaluated over desired Bag of Words and KL Divergence from original token. 

Also checkout [`perturb_past_discrim`](@ref PPLM.perturb_past_discrim)
"""
function perturb_past_bow(model, prev, past, original_probs, args)
    global device
    bow_indices, _ = PPLM.get_bow_indices(args.bow_list, tokenizer)
    _, bow_ohe = PPLM.build_bow_ohe(bow_indices, tokenizer) 
    bow_all = bow_ohe[1]
    for i in 2:length(bow_ohe)
        bow_all = vcat(bow_all, bow_ohe)
    end
    bow_all =bow_all|> device
    
    kl_scale=args.fusion_kl_scale
    opt = ADAM(args.stepsize)
    
    function bow_loss(probs)
        loss_words = bow_all*probs
        loss = -log(sum(loss_words))
        loss
    end
    
    pert_past = deepcopy(past) |> device
    ps = params(pert_past)
    for i in 1:args.num_iterations
        _, back = Zygote.pullback(ps) do
                output = model(prev; past_key_values=pert_past,
                                        output_attentions=false,
                                        output_hidden_states=true,
                                        use_cache=true);
                hidden = output.hidden_states[end]
                logits = model.lm_head(hidden)[:, end, :]
                probs = PPLM.temp_softmax(logits; t=args.temperature)
                loss_1 = bow_loss(probs)
                loss_2 = kl_scale * Flux.kldivergence(probs, original_probs)        # throwing error
                loss_1 + loss_2
            end
        grads = back(1f0)
        update!(opt, ps, grads)
    end
    return pert_past 
end


# need some abstration here
"""
    perturb_hidden_discrim(hidden, model, tokenizer, args)

Perturb hidden states `hidden` based on provided Discriminator (as given with `args`). The perturbation is primarily based on the gradient calculated over losses evaluated over desired Discriminator attribute and KL Divergence from original token. 

Also checkout [`perturb_hidden_bow`](@ref PPLM.perturb_hidden_bow)
"""
function perturb_hidden_discrim(hidden, model, tokenizer, args)
    global device
    classifier, config_metadata = ClassifierHead(;load_from_pretrained=true, discrim=args.discrim)
    classifier = classifier|> device
    if args.target_class_id ==-1
        y_label = config_metadata.default_class
    else
        y_label = args.target_class_id
    end
    y_one_hot = Flux.onehotbatch([y_label], 1:config_metadata["class_size"]) |> device
    
    kl_scale=args.fusion_kl_scale
    opt = ADAM(args.stepsize)
    p = temp_softmax(model.lm_head(hidden)[:, end, :]; t=args.temperature)
    
    new_hidden = deepcopy(hidden)
    ps = params(new_hidden)
    for i in 1:args.num_iterations
        _, back = Zygote.pullback(ps) do
                hidden_sum = sum(new_hidden; dims=2)
                logits_1 = classifier(hidden_sum)
                if length(classifier.linear_layer.bias)==1
                    y_label = y_label |> device
                    loss_1 = Flux.logitbinarycrossentropy(logits_1, y_label)
                else
                    loss_1 = Flux.logitcrossentropy(logits_1, y_one_hot)
                end
                logits = model.lm_head(new_hidden)[:, end, :]
                probs = temp_softmax(logits; t=args.temperature)
                loss_2 = kl_scale * Flux.kldivergence(probs, p) 
                
                loss_1 + loss_2
            end
        grads = back(1f0)
        update!(opt, ps, grads)
    end
    return new_hidden 
end

# TODO add code to support horizontal length, futuristic perturbation
"""
    perturb_past_discrim(model, prev, past, original_probs, args)

Perturb past key values `prev` based on provided Discriminator (as given with `args`). The perturbation is primarily based on the gradient calculated over losses evaluated over desired Discriminator attribute and KL Divergence from original token.

Also checkout [`perturb_past_bow`](@ref PPLM.perturb_past_bow)
"""
function perturb_past_discrim(model, prev, past, original_probs, args)
    global device
    classifier, config_metadata = ClassifierHead(;load_from_pretrained=true, discrim=args.discrim)
    classifier = classifier|> device
    if args.target_class_id ==-1
        y_label = config_metadata.default_class
    else
        y_label = args.target_class_id
    end
    y_one_hot = Flux.onehotbatch([y_label], 1:config_metadata["class_size"]) |> device
    
    kl_scale=args.fusion_kl_scale
    opt = ADAM(args.stepsize)
    
    function bow_loss(probs)
        loss_words = bow_all*probs
        loss = -log(sum(loss_words))
        loss
    end
    
    pert_past = deepcopy(past) |> device
    ps = params(pert_past)
    for i in 1:args.num_iterations
        _, back = Zygote.pullback(ps) do
                output = model(prev; past_key_values=pert_past,
                                        output_attentions=false,
                                        output_hidden_states=true,
                                        use_cache=true);
                hidden = output.hidden_states[end]
                hidden_sum = sum(hidden; dims=2)
                logits_1 = classifier(hidden_sum)
                if length(classifier.linear_layer.bias)==1
                    y_label = y_label |> device
                    loss_1 = Flux.logitbinarycrossentropy(logits_1, y_label)
                else
                    loss_1 = Flux.logitcrossentropy(logits_1, y_one_hot)
                end
                logits = model.lm_head(hidden)[:, end, :]
                probs = temp_softmax(logits; t=args.temperature)
                loss_1 = bow_loss(probs)
                loss_2 = kl_scale * Flux.kldivergence(probs, original_probs)        # throwing error
                loss_1 + loss_2
            end
        grads = back(1f0)
        update!(opt, ps, grads)
    end
    return pert_past 
end
