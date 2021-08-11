# PPLM.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adarshkumar712.github.io/PPLM.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://adarshkumar712.github.io/PPLM.jl/dev)
[![Build Status](https://github.com/adarshkumar712/PPLM.jl/workflows/CI/badge.svg)](https://github.com/adarshkumar712/PPLM.jl/actions)

PPLM.jl is a Julia Package for Controllable Text Generation based on Plug and Play Language Models. The implementation is primarily based on 
Transformers.jl GPT2 and allows user to steer the Text generation task based on some Attribute Models. While being centered around the idea of <b>Gradient based Perturbation</b> from PPLM paper, PPLM.jl supports attribute controlled changes in `Hidden States` and `past key values` of GPT2 model. 

- #### [Original Implementation of PPLM: *uber-research/PPLM*](https://github.com/uber-research/PPLM)

## Plug and Play Language Models: a Simple Approach to Controlled Text Generation

Authors: Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu 

While large pretrained language models can generate coherent text, it's hard to control the context are actually generating. 
Plug and Play Language Models or PPLM allows a user to flexibly plug in one or more tiny attribute models representing the desired steering objective into a large, unconditional language model (LM). The main feature of PPLM is that no fine-tuning of Large LMs is required, thus enabling user to leverage LMs without any hardwares needed to train them.

Arxiv: https://arxiv.org/abs/1912.02164

## What's in PPLM.jl?

PPLM.jl is a `Transformers.jl` based PPLM implementation in Julia to facilitate <b>attribute controlled text generation</b>. While the main feature of this package is to help with Controlled Text generation, it also facilitates the following through simple API functions: 

1) GPT2 pretrained Tokenizer
2) Normal Text generation with GPT2 using few lines of code.
3) Pretrained Discriminators from Huggingface loaded as BSON file. 
4) Some predefined BagofWords.
6) Discriminator Training -  Linear layer classifier on GPT2 model
7) Some more options for Controlled generation of Text, beyond PPLM.

## Status

WIP (Not yet registered)

## RoadMap / Checkpoints

- [x] GPT2 Tokenizer
- [x] Discriminator structure
- [x] Data Preprocessing
- [x] Normal Text Generation
- [x] Controlled Text Generation: Perturb Probabilities
- [x] Controlled Text Generation: Perturb hidden states
- [x] Controlled Text Generation: Perturb Past key values
- [x] Support BagOfWords Models
- [x] Add Docstrings
- [x] Add Documentation
- [x] Add Jupyter Notebook for example on Discriminator Training

For more details on the progress, checkout the [Project:PPLM](https://github.com/AdarshKumar712/PPLM.jl/projects/1)

## Example

First let's load the package and the model:
```julia
using PPLM

tokenizer, model = PPLM.get_gpt2();
```

Example for **BoW Model**:

> **Prompt**: To conclude

```
args= PPLM.pplm(perturb="past", bow_list = ["politics"], stepsize=0.1, fusion_gm_scale=0.8f0, top_k=50)

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="To conclude")
```

Sample generated:

Nomral Generation:

```julia
"To conclude, it is only fair to ask, on the other hand, what we think about one particular type of religious denomination that has an unusual relation to American history (other than the ones associated with Catholicism)?\n\nI could imagine it is just because American social studies scholars aren't as committed to explaining the causes of the American revival. Nor would I imagine Protestant professors who write for the Nation, not least because they might fear an attack by critics on their writings that might bring a backlash against their conclusions,"
```

With BoW model (bow_list = `["politics"]`)

```julia
"To conclude, it's important for governments, from the government of Canada, who decide matters of 
international importance and culture and language issues to the authorities the responsible party for 
immigration enforcement, when that person's an international terrorist, as these are important and cultural 
communities, rather and international business people, like the Canadian government, should take seriously 
when they say these, to the authorities, and then have the Canadian people deal with, and to them be more 
involved in the process itself and their work ethics should really be to"
```

Example for **Discriminator**

> **Prompt**: "You should just kill"

```julia
args = PPLM.pplm(method="Discrim", discrim="toxicity", target_class_id=1, stepsize=0.008, fusion_kl_scale=0.05);

PPLM.sample_pplm(args; tokenizer=tokenizer, model=model, prompt="You should just kill")
```

Sample generated:

Nomral Generation:

```julia
"You should just kill him.\n\n-The Chosen\n\nYou never heard, no matter how stupid people said it. I don't understand how she could think she was safe when she had nothing but contempt for me for four very fucking years I just walked along and played the part of a woman who had the power to do what any woman can do if a woman's life isn't as she told herself it would be so. What's on the page? I mean, there's my dad who was just running"
```

With Disciminator model (discrim = "toxicity")

```julia
"You should just kill yourself before you realize how much harm it can do. If you have never spent a penny you can always try to quit sometime. Some men want a break and some don't: If money helps you, it'll help yourself too, but try to make up for it if that helps or if something good starts to come out.\n\nYou can do even more harm to yourself by being a \"good man,\" that is, by not being selfish.\n\n5 What Would Stake Out\n"
```

For more, checkout the documentation.

## Tutorials / Blogs

1) [PPLM Part 1](https://nextjournal.com/Adarshkumar712/gsoc-2021-pplm.jl)
2) More Blogs Comming soon... (In the meantime checkout the documentation for some examples on usage)

**Note**: There might be some differences here from original implementation of PPLM, but the overall idea of Plug and Play Language Models remains the same as the paper. In case some feature from original repository of PPLM is missing here, I will try to accomodate that in future. 

# References 

1) [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/abs/1912.02164)
2) [https://github.com/uber-research/PPLM](https://github.com/uber-research/PPLM)
3) [https://eng.uber.com/pplm/](https://eng.uber.com/pplm/)
