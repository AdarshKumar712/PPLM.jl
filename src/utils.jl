using Zygote:@adjoint
using StatsBase

@adjoint function sort(x::AbstractArray; by=identity, rev = true)
    p = sortperm(x, by=by, rev = rev)
    return x[p], x̄ -> (x̄[invperm(p)],)
end

"""
    top_k_logits(logits::AbstractArray, k; prob = false)
    
Masks everything but the k top entries as -infinity (1e10). Incase of `probs=true`, everthing except top-k probabilities are masked to 0.0. `logits` is expected to be a vector.

"""
function top_k_logits(logits::AbstractArray, k; prob = false)
    if k==0
    	return logits
    else
        top_kth = sort(logits;rev=true)[k]
        mask = logits.>=top_kth
        if prob==false
            return logits_ = map(*, logits, mask) + map(~, mask).*-1e-10 
        else 
            return logits_ = map(*, logits, mask)
        end
    end
end

# Onecold
function onecold(y)
    onecold_arr = zeros(size(y,2))
    argmax_vec = argmax(y, dims = 1)
    for i in 1:size(y,2)
        onecold_arr[i] = argmax_vec[i].I[1]
    end
    return onecold_arr
end

function temp_softmax(logits; t=1.2)
  return softmax(logits ./ t)
end

function top_k_sample(probs; k=10)
  probs ./=sum(probs)
  sorted = sort(probs, rev = true)
  indexes = partialsortperm(probs, 1:k, rev=true)
  sorted_k = sorted[1:k] ./ sum(sorted[1:k])
  index = sample(indexes, ProbabilityWeights(sorted_k), 1)
  return index[1]
end

function nucleus_sample(probs; p=0.9)
    probs ./=sum(probs)
    sorted = sort(probs, rev = true)
    indexes = sortperm(probs, rev=true)
    cusum = cumsum(sorted)
    upto_threshold = cusum .<= p
    sorted .*= upto_threshold
    sorted ./= sum(sorted)
    index = sample(indexes, ProbabilityWeights(sorted),1)
    return index[1]
end

function binary_accuracy(y_pred, y_true; threshold=0.5)
    @assert size(y_pred) == size(y_true)
    return sum((y_pred .>= threshold) .== y_true) / size(y_true, 1)
end

function categorical_accuracy(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    return sum(onecold(y_pred) .== onecold(y_true)) / size(y_true, 2)
end

# TODO Implement Kldivergence loss, f1_score
