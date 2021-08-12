# TODO: define global verbosity throughout the package

# function for normal generation

"""
    sample_normal(;prompt="I hate the customs", tokenizer=nothing, model=nothing, max_length=100, method="top_k", k=50, t=1.2, p=0.5, add_eos_start=true)

Function to generate normal Sentences with `model` and `tokenizer` provided. In case not provided, function itself create instance of GPT2-small tokenizer and LM Head Model. The sentences are started with the provided `prompt`, and generated till token length reaches `max_length`.

Two sampling methods of generation are provided with this function:
1) method='top_k'
2) method='nucleus'

Any of these methods can be used provided with either k or p.

"""
function sample_normal(;prompt="I hate the customs", tokenizer=nothing, model=nothing, max_length=100, method="top_k", k=50, t=1.2, p=0.5, add_eos_start=true)
    global device
    if tokenizer==nothing
        tokenizer = load_pretrained_tokenizer(GPT2)
        model = hgf"gpt2:lmheadmodel"
    end
    if add_eos_start==true
        input_ = [tokenizer.eos_token_id; tokenizer(prompt)]
    else
        input_ = tokenizer(prompt)
    end
    model = model |> device
    @showprogress "Generating..." for i in 1:max_length
        input_ids = reshape(input_[:], :, 1) |> device
        outputs = model(input_ids; output_attentions=false,
                        output_hidden_states=false,
                        use_cache=false)
        logits = outputs.logits[:, end, 1]
        probs = temp_softmax(logits; t=1.2)  |> cpu
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
"""
    function sample_pplm(pplm_args;tokenizer=nothing, model=nothing, prompt="I hate the customs", sample_method="top_k", add_eos_start=true)

Function for PPLM model based generation. Generate perturbed sentence using `pplm_args`, tokenizer and model (GPT2, in case not provided), starting with `prompt`. In this function the generation is based on the arguments/parameters provided in `pplm_args`, which is an instance of `pplm` struct.  

"""
function sample_pplm(pplm_args;tokenizer=nothing, model=nothing, prompt="I hate the customs", sample_method="top_k", add_eos_start=true)
    if tokenizer == nothing
        if pplm_args.embed_size==1024 || pplm_args.discrim in ["sentiment", "clickbait"]
            tokenizer, model = get_gpt2_medium()
        else
            tokenizer, model = get_gpt2()
        end
    end
        
    if pplm_args.cuda[1]==true
        global device
        device = gpu
        set_device(pplm_args.cuda[2])
    end
        
    if add_eos_start==true
        input_ = [tokenizer.eos_token_id; tokenizer(prompt)]
    else
        input_ = tokenizer(prompt)
    end
    args=pplm_args
    model = model |> device
    @showprogress "Generating..." for i in 1:args.max_length
        input_ids = reshape(input_[:], :, 1) |> device
        outputs = model(input_ids; output_attentions=false,
                            output_hidden_states=true,
                            use_cache=false);
        original_logits = outputs.logits[:, end, 1]
        original_probs = PPLM.temp_softmax(original_logits; t=args.temperature)
        if pplm_args.method=="BoW"
            if pplm_args.perturb=="probs"
                pert_probs = perturb_probs(original_probs, tokenizer, pplm_args)
            elseif pplm_args.perturb == "hidden"
                hidden = outputs.hidden_states[end]
                modified_hidden = perturb_hidden_bow(hidden, model, tokenizer, pplm_args)
                pert_logits = model.lm_head(modified_hidden)[:, end, 1]
                pert_probs = temp_softmax(pert_logits; t=pplm_args.temperature)
            else
                inp = input_ids[1:end-1, :]
                prev = input_ids[end:end, :]
                outputs = model(inp; output_attentions=false,
                                    output_hidden_states=true,
                                    use_cache=true);
                past = outputs.past_key_values;
                original_logits_ = outputs.logits[:, end, 1]
                original_probs_ = temp_softmax(original_logits_; t=pplm_args.temperature)
                past = outputs.past_key_values;
                new_past = perturb_past_bow(model, prev, past, original_probs_, tokenizer, pplm_args)
                output_new = model(prev; past_key_values=new_past,
                                                    output_attentions=false,
                                                    output_hidden_states=true,
                                                    use_cache=true);    
                pert_logits = output_new.logits[:, end, 1]
                pert_probs = temp_softmax(pert_logits; t=args.temperature)
            end
        else
            if pplm_args.perturb == "hidden"
                hidden = outputs.hidden_states[end]
                modified_hidden = perturb_hidden_discrim(hidden, model, tokenizer, pplm_args)
                pert_logits = model.lm_head(modified_hidden)[:, end, 1]
                pert_probs = temp_softmax(pert_logits; t=pplm_args.temperature)
            else
                inp = input_ids[1:end-1, :]
                prev = input_ids[end:end, :]
                outputs = model(inp; output_attentions=false,
                                    output_hidden_states=true,
                                    use_cache=true);
                past = outputs.past_key_values;
                original_logits_ = outputs.logits[:, end, 1]
                original_probs_ = temp_softmax(original_logits_; t=pplm_args.temperature)
                past = outputs.past_key_values;
                new_past = perturb_past_discrim(model, prev, past, original_probs_, pplm_args)
                output_new = model(prev; past_key_values=new_past,
                                                    output_attentions=false,
                                                    output_hidden_states=true,
                                                    use_cache=true);    
                pert_logits = output_new.logits[:, end, 1]
                pert_probs = temp_softmax(pert_logits; t=args.temperature)
            end
        end
        gm_scale=pplm_args.fusion_gm_scale
        pert_probs = Float32.((original_probs.^(1-gm_scale)).*(pert_probs.^gm_scale)) |> cpu
        if sample_method=="top_k"
            new_token = top_k_sample(pert_probs; k=args.top_k)[1] 
        elseif sample_method == "nucleus"
            new_token = nucleus_sample(pert_probs; p=args.top_p)[1]
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
