using Flux.Optimise:ADAM, update!
using Zygote

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
