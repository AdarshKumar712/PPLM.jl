# function for normal generation
# To ask: Which is better? Pass tokenizer and model to the function or call inside function or to create a function to get_tokenizer and model
# To ask: How to handle bpe tokenizing EOS Token while encoding
# TODO: define global verbosity throughout the package
function sample_normal(;primer="I hate the customs", tokenizer=nothing, model=nothing, max_length=100, method="top_k", k=50, t=1.2, p=0.5, add_eos_start=true)
    if tokenizer==nothing
        tokenizer = load_pretrained_tokenizer(GPT2)
        model = hgf"gpt2:lmheadmodel"
    end
    if add_eos_start==true
        input_ = [tokenizer.eos_token_id; tokenizer(primer)]
    else
        input_ = tokenizer(primer)
    end
    for i in 1:max_length
        input_ids = reshape(input_[:], :, 1)
        outputs = model(input_ids; output_attentions=false,
                        output_hidden_states=false,
                        use_cache=false)
        logits = outputs.logits[:, end, 1]
        probs = temp_softmax(logits; t=1.2)
        if method=="top_k"
            new_token = top_k_sample(probs; k=k)[1]
        elseif method == "nucleus"
            new_token = nucleus_sample(probs; p=p)[1]
        end
        push!(input_, new_token)
    end
    text = detokenize(tokenizer, input_)
    if add_eos_start==true
        text = text[length(tokenizer.eos_token)+1:end]
    end
    print("Unpurturbed generated text: ", text, "\n")
    return text
end

# function for perturbed generation

function sample_pplm(pplm_args;tokenizer=nothing, model=nothing, primer="I hate the customs", add_eos_start=true, gm_function=nothing)
    if tokenizer == nothing
        tokenizer = load_pretrained_tokenizer(GPT2)
        model = hgf"gpt2:lmheadmodel"
    end
    if pplm_args.cuda[1]==true
        global device
        device = gpu
        set_device(pplm_args.cuda[2])
    end
    if add_eos_start==true
        input_ = [tokenizer.eos_token_id; tokenizer(primer)]
    else
        input_ = tokenizer(primer)
    end
    model = model |> device
    for i in 1:max_length
        input_ids = reshape(input_[:], :, 1) |> device
        outputs = model(input_ids; output_attentions=false,
                        output_hidden_states=true,
                        use_cache=false)
        original_logits = outputs.logits[:, end, 1]
        original_probs = temp_softmax(original_logits; t=pplm_args.temperature)
        if pplm_args.method=="BoW"
            if pplm_args.perturb=="probs"
                pert_probs = perturb_probs(original_probs, tokenizer, pplm_args)
            elseif pplm_args.perturb == "hidden"
                hidden = outputs.hidden_states[end]
                modified_hidden = perturb_hidden_bow(hidden, model, tokenizer, pplm_args)
                pert_logits = model.lm_head(modified_hidden)[:, end, 1]
                pert_probs = temp_softmax(pert_logits; t=pplm_args.temperature)
            else
                error("Not Implemented Error")
            end
        else
            error("Not Implemented Error")
        end
        gm_scale=pplm_args.fusion_gm_scale
        pert_probs = Float32.((original_probs.^(1-gm_scale)).*(pert_probs.^gm_scale))
        if method=="top_k"
            new_token = top_k_sample(probs; k=args.top_k)[1] |> cpu
        elseif method == "nucleus"
            new_token = nucleus_sample(probs; p=args.top_p)[1] |> cpu
        end
        
        push!(input_, new_token)
    end

    text = detokenize(tokenizer, input_)
    if add_eos_start==true
        text = text[length(tokenizer.eos_token)+1:end]
    end
    print("PPLM (",pplm_args.method,"-",pplm_args.perturb,") generated text: ", text, "\n")
    return text
end
