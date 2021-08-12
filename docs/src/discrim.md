# Discriminator Model

In the BoW Model, we first train a Linear Discriminator over the Large Language Model called as ClassifierHead, to classify wanted vs unwanted class. This ClassifierHead is then used to calculate gradients against the crossentropy loss for p(a/x) where a is the desired attribute. 

In PPLM.jl, the ClassifierHead is defined as a struct:
```julia
struct ClassifierHead
    linear_layer::Dense
    embed_size::Int
    class_size::Int
end
```
You can load a ClassifierHead with any of the following Methods:

```
#Method 1: load a pretrained model
classifier, config_metadata = PPLM.ClassifierHead(;load_from_pretrained=true, discrim="toxicity")    

#Method 2: load a custom trained mode
classifier, config_metadata = PPLM.ClassifierHead(;load_from_pretrained=true, path="./pretrained/custom_model.bson") 

#Method 3: Intiate a random Classifier Layer
classifier, _ = PPLM.ClassifierHead(;load_from_pretrained=true, discrim="toxicity") 

```
Let's delve into an example of PPLM based generation with Discriminator Model.

First, let's load the package and model:
```
using PPLM

tokenizer, model = PPLM.get_gpt2();
```

> **Prompt**: Do I look like I give a

## Perturb Hidden State

Perturbation of hidden states can be done similar to the given example

```julia
args = PPLM.pplm(method="Discrim", perturb="hidden", discrim="toxicity", target_class_id=1, stepsize=0.008, fusion_kl_scale=0.05);

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="Do I look like I give a")

```
Another more crude way of generation could be:

```julia

input_ = [tokenizer.eos_token_id; tokenizer("Do I look like I give a")]

args = PPLM.pplm(method="Discrim", perturb="hidden", discrim="toxicity", target_class_id=1, stepsize=0.008, fusion_kl_scale=0.05);

for i in 1:100
    input_ids = reshape(input_[:], :, 1) |> PPLM.gpu
    outputs = model(input_ids; output_attentions=false,
                        output_hidden_states=true,
                        use_cache=false);
    original_logits = outputs.logits[:, end, 1]
    original_probs = PPLM.temp_softmax(original_logits; t=args.temperature)
    
    hidden = outputs.hidden_states[end]
    modified_hidden = PPLM.perturb_hidden_discrim(hidden, model, tokenizer, args)
    pert_logits = model.lm_head(modified_hidden)[:, end, 1]
    pert_probs = PPLM.temp_softmax(pert_logits; t=args.temperature)
    
    gm_scale = args.fusion_gm_scale
    pert_probs = Float32.((original_probs.^(1-gm_scale)).*(pert_probs.^(gm_scale))) |> cpu
    new_token = PPLM.top_k_sample(pert_probs; k=args.top_k)[1]
    push!(input_, new_token)
end

text = detokenize(tokenizer, input_)

```

Sample generation:

```julia
"Do I look like I give a damn? I want to be a nice person who treats my colleagues and even friends 
like people.\n\nFor one thing, it takes time for me and others to really consider and think about 
your value. In the past, I often felt uncomfortable working with people who thought my interests, 
opinions and interests were different, and didn't have the emotional and spiritual value to interact 
with them. I didn't feel like they wanted me to speak to their views. So I started getting involved 
on many other topics"
```

## Perturb Past Key Values


Perturbation of hidden states can be done similar to the given example

```julia
args = PPLM.pplm(method="Discrim", perturb="past", discrim="toxicity", target_class_id=1, stepsize=0.004, fusion_kl_scale=0.05);

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="Do I look like I give a")

```
Another more crude way of generation could be:

```julia

input_ = [tokenizer.eos_token_id; tokenizer("Do I look like I give a")]

args = PPLM.pplm(method="Discrim", perturb="past", discrim="toxicity", target_class_id=1, stepsize=0.008, fusion_kl_scale=0.05);

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
    
    new_past = PPLM.perturb_past_discrim(model, prev, past, original_probs, args)
    output_new = model(prev; past_key_values=new_past,
                                        output_attentions=false,
                                        output_hidden_states=true,
                                        use_cache=true);    
    pert_logits = output_new.logits[:, end, 1]
    pert_probs = PPLM.temp_softmax(pert_logits; t=args.temperature)
    #print(sum(pert_probs.-original_probs))
    
    gm_scale = args.fusion_gm_scale
    pert_probs = Float32.((original_probs.^(1-gm_scale)).*(pert_probs.^(gm_scale))) |> cpu
    new_token = PPLM.top_k_sample(pert_probs; k=args.top_k)[1]
    push!(input_, new_token)
end

text = detokenize(tokenizer, input_)

```

Sample generation:

```julia
"Do I look like I give a proper treatment to these people? We're seeing real examples in all the 
things that they have done as well. There is going to be a discussion on there with the state of 
what steps we should be taking to address all cases of people in the community, and then what we 
are going to do going forward that has not a national interest interest. Is your experience with 
similar issues from different different sides affected your work/responsibility of not doing that 
things you find seem quite simple, at first glance?"
```

## Load Custom Model

You can use your own custom train Model (suppose saved at path=`path`) using the following:

```julia
args = PPLM.pplm(method="Discrim", discrim="custom", path=path, target_class_id=1, stepsize=0.008, fusion_kl_scale=0.05);

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="Do I look like I give a")

```

**Note**: For different Discriminator, you may need to tune some hyperparameters like `stepsize`, `fusion_gm_scale` etc. to get some really interesting results. Will add more details on it later. Also note that in first iteration, it usually takes more time to evaluate the gradients but becomes fast in consecutive passes.
