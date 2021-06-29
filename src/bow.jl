using HTTP

function get_bow_indices(bow_key_or_path_list::Vector{String}, tokenizer)
    bow_indices = []
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
        close(f)
    end
    return bow_indices
end

function build_bow_ohe(bow_indices, tokenizer)
    global device
    if bow_indices==nothing
        return
    end
    onehot =[]
    onehot_single = zeros(length(tokenizer.vocab))
    for bow in bow_indices
        bow = filter(x->(length(x)<=1),bow)
        x = zeros(length(tokenizer.vocab), length(bow))
        for (i, word_id) in enumerate(bow)
            x[word_id[0],i] = 1
            onehot_single[word_id[0]]=1
        end 
        x = x|> device
        push!(onehot, x)
    end
    onehot_single = onehot_single |> gpu 
    return onehot_single, onehot[1]
end


