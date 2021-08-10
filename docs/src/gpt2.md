# GPT2: Tokenization and Generation

PPLM.jl comes along with support of GPT2 tokenizer and Generation. Here is an example of how you can use this feature of PPLM.jl package for tokenization and normal generation

## Tokenization

PPLM.jl allows users to load pre-trained GPT2 tokenizers based on BytePairEncoding.jl and Transformers.jl, which can then be used to tokenize/encode English text with a single line of code. The tokenizer is implemented with the following structure:

```julia
abstract type GPT2 <: PretrainedTokenizer end

struct GPT2Tokenizer <: GPT2
    encoder::Vocabulary{String}
    bpe_encode::GenericBPE
    bpe_decode::UnMap
    vocab::Dict{String, Any}
    unk_token::String
    unk_id::Int   
    eos_token::String
    eos_token_id::Int
    pad_token::String
    pad_token_id::Int
end
```

#### Example of Tokenization

Let's see how you can tokenize text with PPLM.

```julia
# Load Tokenizer
using PPLM
tokenizer = PPLM.load_pretrained_tokenizer(GPT2)

sentence = "This is an example of Tokenization"
```
Once, you have loaded your tokenizer, one can use either of the methods:

```julia
tokens = tokenizer(sentence)
# or 
tokens = encode(tokenizer, sentence)
```
It will return the following output:
```
7-element Vector{Int64}:
  1213
   319
   282
  1673
   287
 29131
  1635
```

Now you have your list of tokens. Suppose you want to get back your sentence. This can be done in two ways:

```
# Firsth Method:
sentence = detokenize(tokenizer, tokens)

# Second Method:
decoded_tokens_list = decode(tokenizer, tokens)	
# returns vector: ["This", "Ġis", "Ġan", "Ġexample", "Ġof", "ĠToken", "ization"]
sentence = detokenizer(tokenizer, decoded_tokens_list) 
```
You will get back your original sentence `This is an example of Tokenization`


## Generation : Normal Text

PPLM.jl can be used to generate normal (unperturbed) text with the GPT2 model, with any of the two sampling methods `top_k` and `nucleus`:

To generate text, you can use the following code:


Here is a Sample text generated with GPT2 using the above code:

> With **Top_k sampling**, k=50, prompt = "Fruits are"
```julia
"Fruits are the key ingredient in our diet; their vitamins, and proteins are essential to build our immune system. What makes a good fruit one of them is simply as simple as your diet is. Fruit is one simple nutrient that is used effectively as a defense against sickness and stress (which can be very life changing indeed). When the body has just consumed enough fat for at least 40-50 days, the body also releases hormones known as the hormone estrogen in order to prevent infection. A good diet makes life easier"
```

> With **Nucleus sampling**, p=0.6, prompt = "Fruits are"
```julia
"Fruits are packed with the goodness of ancient Greek life, plants that protect and revive us from death. At every stone in the garden, your fruit may reflect on the people who once carried you from town to town, those who would still give you food to live, and the perfect pair of hand-gloved fingers you may wear in your golden bedroll."
```