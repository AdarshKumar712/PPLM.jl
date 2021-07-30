using JSON
using BytePairEncoding
using BytePairEncoding: UnMap
using Transformers.Basic

abstract type GPT2 <: PretrainedTokenizer end

# wrapper for GPT2 Tokenizer with required functionalities
"""
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

Structure to hold all essential information / functions for GPT2 tokenizer

"""
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

"""
    load_pretrained_tokenizer(ty::Type{T}; unk_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>") where T<:PretrainedTokenizer

Load GPT2 tokenizer using Datadeps for pretrained bpe and vocab. Returns tokenizer as `GPT2Tokenizer` structure.

"""
function load_pretrained_tokenizer(ty::Type{T}; unk_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>") where T<:PretrainedTokenizer
    path_bpe = joinpath(datadep"BPE", readdir(datadep"BPE")[1])
    path_vocab = joinpath(datadep"Vocab", readdir(datadep"Vocab")[1])
    load_pretrained_tokenizer(path_bpe, path_vocab, unk_token, eos_token, pad_token)
end

"""
    load_pretrained_tokenizer(path_bpe, path_vocab, unk_token, eos_token, pad_token)

Load pretrained tokenizer for GPT2 from provided bpe and vocab file path. Initialises `unk_token`, `eos_token`, `pad_token` as provided with the function. Returns tokenizer as `GPT2Tokenizer` structure.

"""
function load_pretrained_tokenizer(path_bpe, path_vocab, unk_token, eos_token, pad_token)
    vocab = JSON.parsefile(path_vocab)
    labels = map(x->x[1], sort!(collect(pairs(vocab)), by=x->x[2]))
    encoder = Vocabulary(labels, unk_token)
    bpe_encode = ByteLevelBPE(path_bpe)
    bpe_decode = BytePairEncoding.UnMap(bpe_encode.codemap)
    GPT2Tokenizer(encoder, bpe_encode, bpe_decode, vocab, unk_token, encoder(unk_token),
            eos_token, encoder(eos_token), pad_token, encoder(eos_token))
end

"""
    tokenize(t::GPT2Tokenizer, text::AbstractString)

Function to tokenize given `text` with tokenizer bpe encoder (`t.bpe_encode`). Returns a string vector of tokens.
"""
function tokenize(t::GPT2Tokenizer, text::AbstractString)
    t.bpe_encode(text)
end

"""
    encode(t::GPT2Tokenizer, text::AbstractString; add_prefix_space=false)

Returns the encoded vector of tokens (mapping from vocab of Tokenizer) for `text`. If `add_prefix_space`=true, add space at the start of 'text' before tokenization. 

# Example 

For single text:

```julia
encode(tokenizer, text)
```

For vector of text:

```julia
map(x->encode(tokenizer, x), text_vector) 
```
"""
function encode(t::GPT2Tokenizer, text::AbstractString; add_prefix_space=false)
    if add_prefix_space==true
        text = string(" ", text)
    end
    tokens = tokenize(t, text)
    t.encoder(tokens)
end

"""
    encode(t::GPT2Tokenizer, tokens::Vector{String})

Function to encode tokens vectors to their integer mapping from vocab of tokenizer.
"""
function encode(t::GPT2Tokenizer, tokens::Vector{String})
    t.encoder(tokens)
end

"""
    (t::GPT2Tokenizer)(text::AbstractString; add_prefix_space=false)

Encode the text with tokenizer and returns the encoded vector. If `add_prefix_space`=true, add space at the start of 'text' before tokenization. 

# Examples: 

For a single text:
```julia
tokenizer(text; add_prefix_space=true)
```

For vector of texts, use:

```julia
map(x->encode(tokenizer, x), text_vector) 

# or

tokenizer.(text_vector)
```

Also checkout [`encode`](@ref PPLM.encode)
"""
function (t::GPT2Tokenizer)(text::AbstractString; add_prefix_space=false)
    encode(t, text; add_prefix_space=add_prefix_space)
end


decode(vocab::Vocabulary{T}, i::Int) where T = 0 <= i <= length(vocab) ? vocab.list[i] : vocab.unk

"""
    decode(vocab::Vocabulary{T}, is::Vector{Int}) where T

Return decoded vector of `string` tokens from the indices vector `is`, using the vocab.
"""
function decode(vocab::Vocabulary{T}, is::Vector{Int}) where T
    tokens = Vector{String}(undef, length(is))
    for (idx, i) ∈ enumerate(is)
        token = decode(vocab, i)
        tokens[idx] = token
    end
    tokens
end


"""
    decode(t::GPT2Tokenizer, tokens_ids::Vector{Int})

Return decoded vector of `string` tokens from the indices vector `tokens_ids`, using the tokenizer `t` encoder .
"""
function decode(t::GPT2Tokenizer, tokens_ids::Vector{Int})
    decode(t.encoder, tokens_ids)
end

"""
    detokenize(t::GPT2Tokenizer, tokens::Vector{String})

BPE Decode the vector of strings, using the tokenizer `t`.
"""
function detokenize(t::GPT2Tokenizer, tokens::Vector{String})
    t.bpe_decode(join(tokens))
end

# Example for detokenizing multiple examples: map(token_ids->detokenize(tokenizer, token_ids), tokens)
"""
    detokenize(t::GPT2Tokenizer, tokens_ids::Vector{Int})

Decode and Detokenize the vector of indices `token_ids`. Returns the final sentence after detokenization.

# Example 

For single vector of token_ids:

```julia
detokenize(tokenizer, token_ids)
```

For vector of vector of `token_ids`, use:

``` julia
map(x->decode(tokenizer, x), tokens_id_vector_of_vector)
```

"""
function detokenize(t::GPT2Tokenizer, tokens_ids::Vector{Int})
    tokens_list = decode(t, tokens_ids)
    detokenize(t, tokens_list)
end
