```@meta
CurrentModule = PPLM
```

# PPLM.jl

PPLM.jl is a julia based implementation of [Plug and Play Language Models](https://github.com/uber-research/PPLM). The implementation is primarily based on Transformers.jl GPT2 and allows user to steer the Text generation task based on some Attribute Models.


# Why PPLM.jl?

While large pretrained language models can generate coherent text, it's hard to control the context are actually generating. 
Plug and Play Language Models or PPLM allows a user to flexibly plug in one or more tiny attribute models representing the desired steering objective into a large, unconditional language model (LM). While the main feature of this package is to help with Controlled Text generation, it also facilitates the following through simple API functions: 

1) GPT2 pretrained Tokenizer
2) Normal Text generation with GPT2 using few lines of code.
3) Pretrained Discriminators from Huggingface loaded as BSON file. 
4) Some predefined BagofWords.
6) Discriminator Training -  Linear layer classifier on GPT2 model
7) Some more options for Controlled generation of Text, as an extension to PPLM.


# Installation

Will be updated once registered...

```
Not yet regsitered
```