using JSON
using BytePairEncoding
using BytePairEncoding: UnMap
using Transformers.Basic

abstract type GPT2 <: PretrainedTokenizer end

# wrapper for GPT2 Tokenizer with required functionalities
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
   
function load_pretrained_tokenizer(ty::Type{T}; unk_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>") where T<:PretrainedTokenizer
    path_bpe = joinpath(datadep"BPE", readdir(datadep"BPE")[1])
    path_vocab = joinpath(datadep"Vocab", readdir(datadep"Vocab")[1])
    load_pretrained_tokenizer(path_bpe, path_vocab, unk_token, eos_token, pad_token)
end

function load_pretrained_tokenizer(path_bpe, path_vocab, unk_token, eos_token, pad_token)
    vocab = JSON.parsefile(path_vocab)
    labels = map(x->x[1], sort!(collect(pairs(vocab)), by=x->x[2]))
    encoder = Vocabulary(labels, unk_token)
    bpe_encode = ByteLevelBPE(path_bpe)
    bpe_decode = BytePairEncoding.UnMap(bpe_encode.codemap)
    GPT2Tokenizer(encoder, bpe_encode, bpe_decode, vocab, unk_token, encoder(unk_token),
            eos_token, encoder(eos_token), pad_token, encoder(eos_token))
end

function tokenize(t::GPT2Tokenizer, text::AbstractString)
    t.bpe_encode(text)
end

function encode(t::GPT2Tokenizer, text::AbstractString; add_prefix_space=false)
    if add_prefix_space==true
        text = string(" ", text)
    end
    tokens = tokenize(t, text)
    t.encoder(tokens)
end

function encode(t::GPT2Tokenizer, tokens::Vector{String})
    t.encoder(tokens)
end

"""


Example: for vector of texts -> map(x->encode(tokenizer, x), text_vector) or tokenizer.(text_vector)
"""
function (t::GPT2Tokenizer)(text::AbstractString; add_prefix_space=false)
    encode(t, text; add_prefix_space=add_prefix_space)
end

decode(vocab::Vocabulary{T}, i::Int) where T = 0 <= i <= length(vocab) ? vocab.list[i] : vocab.unk

function decode(vocab::Vocabulary{T}, is::Vector{Int}) where T
    tokens = Vector{String}(undef, length(is))
    for (idx, i) ∈ enumerate(is)
        token = decode(vocab, i)
        tokens[idx] = token
    end
    tokens
end

function decode(t::GPT2Tokenizer, tokens_ids::Vector{Int})
    decode(t.encoder, tokens_ids)
end

function detokenize(t::GPT2Tokenizer, tokens::Vector{String})
    t.bpe_decode(join(tokens))
end

# Example for detokenizing multiple examples: map(token_ids->detokenize(tokenizer, token_ids), tokens)

function detokenize(t::GPT2Tokenizer, tokens_ids::Vector{Int})
    tokens_list = decode(t, tokens_ids)
    detokenize(t, tokens_list)
end

