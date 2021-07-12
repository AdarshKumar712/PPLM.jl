# Pre-process the dataset before use in discriminator training
using CSV

function pad_seq(batch::AbstractVector{T}, pad_token::Integer=0) where T	# send a list of sequence
    length_vec = map(length, batch)
    max_len = maximum(length_vec)
    padded_seq = zeros(max_len, length(batch))
    for (index,tokens) in enumerate(batch)
        len = length_vec[index]
        padded_seq[1:len, index] = tokens
        padded_seq[len+1:end, index] .= pad_token
    end 
    return padded_seq
end

function get_mask(seq::AbstractMatrix{T}, pad_token::Integer=0, embed_size::Integer=768) where T
    seq_len, batch_size = size(seq)
    mask = reshape(repeat(seq.!=pad_token, embed_size),  seq_len, embed_size, batch_size)
    permutedims(mask, (2,1,3))	# shape (embed_size, seq_length, batch_size)
end

# expect a tuple of (X, Y) where each is a list
function data_preprocess(data_x, data_y, classification_type::String="Binary", num_classes::Integer=2; args=nothing)
    if classification_type!="Binary"
        data_y = Float32.(onehotbatch(data_y, 1:num_classes))
    else
        data_y = Float32.(reshape(data_y, 1, length(data_y)))
    end
    if args==nothing
        pad_token = 0
        embed_size = 768
    else
        pad_token = args.pad_token
        embed_size = args.embed_size
    end
    data_x = pad_seq(data_x, pad_token)
    mask_src = get_mask(data_x, pad_token, embed_size)
    return Int.(data_x), data_y, Float32.(mask_src) 
end

function truncate_(x, max_length::Integer)
    x[1:min(max_length, length(x))]
end

function load_data(data_x, data_y, tokenizer::PretrainedTokenizer; batchsize::Integer=8, truncate::Bool=false, max_length::Integer=256, shuffle::Bool=false, drop_last::Bool=false, add_eos_start::Bool=true)
    data_x = map(tokenizer, data_x)
    if add_eos_start==true
        data_x = map(pushfirst!, data_x, repeat([tokenizer.eos_token_id], length(data_x)))
    end
    if truncate==true
        data_x = truncate_.(data_x, max_length)
    end
    loader = DataLoader((data_x, data_y), batchsize=batchsize, partial=drop_last, shuffle=shuffle)
    return loader
end

function load_cached_data(discrim::Union{DiscriminatorV1, DiscriminatorV2}, data_x, data_y, tokenizer::PretrainedTokenizer; truncate::Bool=false, max_length::Integer=256, shuffle::Bool=false, batchsize::Int=4, drop_last::Bool=false, classification_type="Binary", num_classes=2, args=nothing)
    global device
    # if label zero indexed, increase by 1
    if 0 in data_y
        data_y.+=1
    end
    loader = load_data(data_x, data_y, tokenizer, batchsize=1,
                truncate=truncate, max_length=max_length, shuffle=false, drop_last=drop_last)
    
    xs = Array{Float32}(undef, discrim.embed_size, 0)
    
    if classification_type == "Binary"
        ys = Array{Int}(undef, 1, 0)
    else
        ys = Array{Float32}(undef, num_classes, 0)
    end
    
    for (x, y) in loader
        x_, y_, mask =  (data_preprocess(x, y, classification_type, num_classes; args=args))|> device
        if typeof(discrim)==DiscriminatorV1
            data_x_ = avg_representation(discrim, x_)             
        else
            data_x_ = avg_representation(discrim, x_, mask)
        end
        data_x_ = Float32.(data_x_) |> cpu
        data_y_ = Float32.(y_) |> cpu
        xs = cat(xs, data_x_, dims=2)
        ys = cat(ys, data_y_, dims=2)
    end
    if args!=nothing
        batchsize=args.batchsize
    end
    loader = DataLoader((xs, ys), batchsize=batchsize, partial=drop_last, shuffle=shuffle)
    return loader
end

function load_data_from_csv(path_to_csv; text_col="text", label_col="label", delim=',', header=1)
    df = CSV.File(path_to_csv; delim=delim) |> DataFrame
    return df[!, text_col], df[!, label_col]
end

