using HTTP

"""
    get_bow_indices(bow_key_or_path_list::Vector{String}, tokenizer)

Returns a list of `list of indices` of words from each Bag of word in the `bow_key_or_path_list`, after tokenization. The functions looks for provided BoW key in the registered artifacts `Artifacts.toml` file. In case not present there, function expects that bow_key is provided as the complete path to the file the URL to download .txt file.

# Example
``` julia
get_bow_indices(["legal", "military"])
```

"""
function get_bow_indices(bow_key_or_path_list::Vector{String}, tokenizer)
    bow_indices = []
    bow_words = []
    for bow_key_or_path in bow_key_or_path_list
        if !isnothing(artifact_hash(bow_key_or_path, ARTIFACTS_TOML))
            file_path = joinpath(get_registered_file(bow_key_or_path), string(bow_key_or_path, ".txt"))
        else
            if isfile(bow_key_or_path)
                file_path = bow_key_or_path
            elseif isurl(bow_key_or_path)
                isdir("BoW")||mkdir("BoW")
                HTTP.download(bow_key_or_path, joinpath("BoW", split(bow_key_or_path, "/")[end]))
            else
                error("$bow_key_or_path not a valid entry for BoW list source")
            end
        end
        f = open(file_path, "r") 
        words = split(strip(read(f, String)), "\n")
        encoded = map(x -> tokenizer(x; add_prefix_space=true), words)
        push!(bow_indices, encoded)
        push!(bow_words, words)
        close(f)
    end
    return bow_indices, bow_words
end

"""
    build_bow_ohe(bow_indices, tokenizer)

Build and return a list of `one_hot_matrix` for each Bag Of Words list from indices. Each item of the list is of dimension `(num_of_BoW_words, tokenizer.vocab_size)`. 

Note: While building the OHE of word indices, it only keeps those words, which have length `1` after tokenization and discard the rest.

"""
function build_bow_ohe(bow_indices, tokenizer)
    global device
    if bow_indices==nothing
        return
    end
    onehot =[]
    onehot_single = zeros(length(tokenizer.vocab))
    for bow in bow_indices
        bow = filter(x->(length(x)<=1),bow)
        x = zeros(length(bow), length(tokenizer.vocab))
        for (i, word_id) in enumerate(bow)
            x[i, word_id[1]] = 1
            onehot_single[word_id[1]]=1
        end 
        x = x|> device
        push!(onehot, x)
    end
    onehot_single = onehot_single |> device
    return onehot_single, onehot
end