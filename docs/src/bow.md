# Bag Of Words Model

In the BoW model, we have a list of words that correspond to that particular attribute, usually used in sentences that can be considered to have that attribute. For example, consider the case of Topic-based attribute, let's say `Politics`, where we want to drive the topic of generation towards politics, then a typical BagOfWord for `Politics` will include words like government, politics, democracy, federation, etc, etc. These words will then be used for perturbation through PPLM.jl.

Let's take up an example to understand how it works.

Let's first initialize the package and model.
```julia
using PPLM

tokenizer, model = PPLM.get_gpt2();
model = model |> PPLM.gpu
```

> **Prompt**: To conclude


## Perturb Probability

This feature only supports for Bag of Words Model. Perturbation of probability can be done similar to the given example:

```julia
args= PPLM.pplm(perturb="probs", bow_list = ["politics"], stepsize=1.0, fusion_gm_scale=0.8f0, top_k=50)

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="To conclude")

```

Another more crude way of generation could be:

```julia
input_ = [tokenizer.eos_token_id; tokenizer("To conclude")]

args= PPLM.pplm(perturb="probs", bow_list = ["politics"], stepsize=1.0, fusion_gm_scale=0.8f0, top_k=50)

for i in 1:100
    input_ids = reshape(input_[:], :, 1)
    outputs = model(input_ids; output_attentions=false,
                        output_hidden_states=true,
                        use_cache=false);
    original_logits = outputs.logits[:, end, 1]
    original_probs = PPLM.temp_softmax(original_logits; t=args.temperature)
    pert_probs = PPLM.perturb_probs(original_probs, tokenizer, args)
    gm_scale = args.fusion_gm_scale
    pert_probs = Float32.((original_probs.^(1-gm_scale)).*(pert_probs.^(gm_scale)))
    new_token = PPLM.top_k_sample(pert_probs; k=args.top_k)[1]
    push!(input_, new_token)
end

text = detokenize(tokenizer, input_)

```

Sample generation:

```julia
"To conclude the last week about our current policy, they say we \"don't follow their dictates.\"[25][26] 
We've never followed that precept, so in order, what we have is a very limited, almost not always followed 
agenda. It took decades to implement and it has already occurred once.[27] These are the arguments that 
most conservative leaders used to put forth to show the world that our public spending has failed – as 
conservatives pointed out. What they don't seem to understand is that this ideology"
```

## Perturb Hidden States

Perturbation of hidden states can be done similar to the given example

```julia
args= PPLM.pplm(perturb="hidden", bow_list = ["politics"], stepsize=0.02, fusion_gm_scale=0.8f0, top_k=50)

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="To conclude")

```
Another more crude way of generation could be:

```julia

input_ = [tokenizer.eos_token_id; tokenizer("To conclude")]

args= PPLM.pplm(perturb="hidden", bow_list = ["politics"], stepsize=0.02, fusion_gm_scale=0.8f0, top_k=50)

for i in 1:100
    input_ids = reshape(input_[:], :, 1) |> PPLM.gpu
    outputs = model(input_ids; output_attentions=false,
                        output_hidden_states=true,
                        use_cache=false);
    original_logits = outputs.logits[:, end, 1]
    original_probs = PPLM.temp_softmax(original_logits; t=args.temperature)
    
    hidden = outputs.hidden_states[end]
    
    modified_hidden = PPLM.perturb_hidden_bow(model, hidden, args)
    pert_logits = model.lm_head(modified_hidden)[:, end, 1]
    pert_probs = PPLM.temp_softmax(pert_logits; t=args.temperature)
    
    gm_scale = args.fusion_gm_scale
    pert_probs = Float32.((original_probs.^(1-gm_scale)).*(pert_probs.^(gm_scale))) |> cpu
    new_token = PPLM.top_k_sample(pert_probs; k=args.top_k)[1]
    push!(input_, new_token)
    #print(".")
end

text = detokenize(tokenizer, input_)

```

Sample generation:

```julia
"To conclude this brief essay, I have briefly discussed one of my own writing's main points: How a great 
many poor working people who were forced by the government to sell goods to high-end supermarkets to make 
ends meet were put off purchasing goods at a time they wouldn't be able afford. That point of distinction 
arises in every social democracy I identify as libertarian.\n\nA large number of people in this group 
simply did not follow basic political norms, and in order not to lose faith that politics was in"
```

### Perturb Past


```julia
args= PPLM.pplm(perturb="past", bow_list = ["politics"], stepsize=0.005, fusion_gm_scale=0.8f0, top_k=50)

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="To conclude")

```
Another more crude way of generation could be:

```julia
input_ = [tokenizer.eos_token_id; tokenizer("To conclude")]

args= PPLM.pplm(perturb="past", bow_list = ["politics"], stepsize=0.005, fusion_gm_scale=0.8f0, top_k=50)

for i in 1:100
    input_ids = reshape(input_[:], :, 1) |> PPLM.gpu
    inp = input_ids[1:end-1, :]
    prev = input_ids[end:end, :]
    outputs = model(inp; output_attentions=false,
                        output_hidden_states=true,
                        use_cache=true);
    past = outputs.past_key_values;
    original_logits = outputs.logits[:, end, 1]
    original_probs = PPLM.temp_softmax(original_logits; t=args.temperature)
    
    new_past = PPLM.perturb_past_bow(model, prev, past, original_probs, args)
    output_new = model(prev; past_key_values=new_past,
                                        output_attentions=false,
                                        output_hidden_states=true,
                                        use_cache=true);    
    pert_logits = output_new.logits[:, end, 1]
    pert_probs = PPLM.temp_softmax(pert_logits; t=args.temperature)
    
    gm_scale = args.fusion_gm_scale
    pert_probs = Float32.((original_probs.^(1-gm_scale)).*(pert_probs.^(gm_scale))) |> cpu
    new_token = PPLM.top_k_sample(pert_probs; k=args.top_k)[1]
    push!(input_, new_token)
    #print(".")
end

text = detokenize(tokenizer, input_)

```
Sample generation:

```julia
"To conclude, it's important for governments, from the government of Canada, who decide matters of 
international importance and culture and language issues to the authorities the responsible party for 
immigration enforcement, when that person's an international terrorist, as these are important and cultural 
communities, rather and international business people, like the Canadian government, should take seriously 
when they say these, to the authorities, and then have the Canadian people deal with, and to them be more 
involved in the process itself and their work ethics should really be to"
```

**Note**: For different topics, you may need to tune some hyperparameters like `stepsize`, `fusion_gm_scale` etc. to get some really interesting results. Will add more details on it later. Also note that in first iteration, it usually takes more time to evaluate the gradients but becomes fast in consecutive passes.