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
- [ ] Add Docstrings
- [ ] Add Documentation
- [ ] Add Jupyter Notebook for examples

For more details on the progress, checkout the [Project:PPLM](https://github.com/AdarshKumar712/PPLM.jl/projects/1)

## Example

Coming soon...

## Tutorials / Blogs

Coming soon...

**Note**: There might be some differences here from original implementation of PPLM, but the overall idea of Plug and Play Language Models remains the same as the paper. In case some feature from original repository of PPLM is missing here, I will try to accomodate that in future. 

# References 

1) [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/abs/1912.02164)
2) [https://github.com/uber-research/PPLM](https://github.com/uber-research/PPLM)
3) [https://eng.uber.com/pplm/](https://eng.uber.com/pplm/)
