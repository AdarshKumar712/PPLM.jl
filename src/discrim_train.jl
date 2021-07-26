using Parameters: @with_kw
using MLDataUtils
using Flux
using Flux: OneHotArray
using Zygote
using CUDA
import Flux.Optimise: update!, ADAM
using Flux: logitcrossentropy, logitbinarycrossentropy
using Transformers.HuggingFace

"""
@with_kw struct HyperParams
    batchsize::Int=8
    eos_token::Int=50527
    pad_token::Int=0
    embed_size::Int=768
    classification_type::String="Binary"
    target_class=1
    num_classes::Int=2
    cached::Bool=true
    lr::Float64=1e-6
    epochs::Int=20
    log_interval::Int=100
end

Structure for `HyperParams` to select hyperparamters for training the discriminator model. 

"""
@with_kw struct HyperParams
    batchsize::Int=8
    eos_token::Int=50527
    pad_token::Int=0
    embed_size::Int=768
    classification_type::String="Binary"
    target_class=1
    num_classes::Int=2
    cached::Bool=true
    lr::Float64=1e-6
    epochs::Int=20
    log_interval::Int=100
end

"""
    train!(discrim, data_loader; args=args)

Train the discriminator using the provided `data_loader` training data and arguments `args` provided.
"""
function train!(discrim, data_loader; args=nothing)
    if args==nothing
        @warn "No args provided, using the default arguments from HyperParams structure."
        args = HyperParams()
    end
    
    global device
    opt = ADAM(args.lr)
    ps = params(discrim)
    
    function loss_fn(x, y)
        if args.classification_type=="Binary"
            return logitbinarycrossentropy(x, y)
        else
            return logitcrossentropy(x, y)
        end
    end
    
    print("Training starting.........\n\n")
    for epoch in 1:args.epochs
        avg_loss = 0.0
        step =0
        for (data_x, data_y) in data_loader
            if args.cached == true
                data_x, data_y = (data_x, data_y) |> device
                mask = nothing
            else 
               data_x, data_y, mask = data_preprocess(data_x, data_y, args.classification_type, args.num_classes; args=args) |> device
            end
            
            loss, back = Zygote.pullback(ps) do
                        logits = discrim(data_x, mask; args = args)
                        loss_fn(logits, data_y)
                      end
            grads = back(1f0)
            update!(opt, ps, grads)
            avg_loss += loss
            step+=1
            
            if step%args.log_interval==0
                print("Epoch #", epoch, " : Step #", step, " : NLL Loss: ", avg_loss/step, "\n")
            end
        end
        print("Loss Epoch #", epoch, " : ", step, " NLL Loss: ", avg_loss/step, "\n")
        print("\n")
    end
end

"""
    test!(discrim, data_loader; args=nothing)

Test the discriminator on test data provided using `data_loader`, based on Accuracy and NLL Loss. 

"""
function test!(discrim, data_loader; args=nothing)
    global device
    function loss_fn(x, y)
        if args.classification_type=="Binary"
            return logitbinarycrossentropy(x, y)
        else
            return logitcrossentropy(x, y)
        end
    end
    
    avg_loss = 0
    if args.classification_type=="Binary"
        preds = Array{Float32}(undef, 1, 0)
        ys = Array{Float32}(undef, 1, 0)
    else
        preds = Array{Float32}(undef, args.num_classes, 0)
        ys = Array{Float32}(undef, args.num_classes, 0)
    end
    steps=0
    print("Testing.......\n\n")
    for (data_x, data_y) in data_loader
        if args.cached == true
            data_x, data_y = (data_x, data_y) |> device
            mask = nothing
        else 
           data_x, data_y, mask = data_preprocess(data_x, data_y, args.classification_type, args.num_classes; args=args) |> device
        end
    
        logits = discrim(data_x, mask; args = args)
        loss = loss_fn(logits, data_y)
        if args.classification_type !="Binary"
            logits = softmax(logits)
        else
            logits = sigmoid.(logits)
        end
        logits, data_y = (logits, data_y) |> cpu
        preds= cat(preds, logits, dims=2)
        ys = cat(ys, data_y, dims=2)
        avg_loss += loss
        steps+=1
    end
    loss = avg_loss/steps
    if args.classification_type=="Binary"
        acc = binary_accuracy(preds, ys)
    else
        acc = categorical_accuracy(preds, ys)
    end
    print("Test Results: NLL Loss: ", loss, ", Accuracy: ", acc)
end

"""
    train_discriminator(text, labels, batchsize::Int=8, classification_type::String="Binary", num_classes::Int=2; model="gpt2", cached::Bool=true, discrim=nothing, tokenizer=nothing, truncate=true, max_length=256, train_size::Float64=0.9, lr::Float64=1e-5, epochs::Int=10, args=nothing)

Function to train discriminator for provided `text` and target `labels`, based on set of function paramters provided. Returns `discrim` discriminator after training.

Here the `cached=true` allows cacheing of contexualized embeddings (forward pass) in GPT2 model, as the model itself is non-trainable. This reduces the time of training effectively as the forward pass through GPT2 model is to be done only once.

# Example

Consider a Multiclass classification problem with class size of 5, it can trained on `text` and `labels` vectors using:

```julia
train_discriminator(text, labels, 16, "Multiclass", 5)
```

"""
function train_discriminator(text, labels, batchsize::Int=8, classification_type::String="Binary", num_classes::Int=2; model="gpt2", cached::Bool=true, discrim=nothing, tokenizer=nothing, truncate=true, max_length=256, train_size::Float64=0.9, lr::Float64=1e-5, epochs::Int=10, args=nothing)
    global device
    # Right now only support gpt2
    if tokenizer == nothing && model=="gpt2"
        model = hgf"gpt2:lmheadmodel" |> device
        config = load_config("gpt2")
        tokenizer = load_pretrained_tokenizer(GPT2)
    end
    
    if args == nothing
        args = HyperParams(batchsize=batchsize, embed_size=config.n_embd, classification_type=classification_type, num_classes=num_classes, cached=cached, lr=lr, epochs=epochs)
    end
    
    if discrim == nothing
        if args.classification_type=="Binary"
            class_size=1
        else
            class_size=num_classes
        end
        discrim = get_discriminator(model, class_size=class_size, embed_size=args.embed_size)
    end
    if cached == true
        print("Creating Cache of data and Data Loader...")
        text, labels = shuffleobs((text, labels))
        (train_x, train_y), (test_x, test_y) = splitobs((text, labels); at=train_size)
        
        train_loader = load_cached_data(discrim, train_x, train_y, tokenizer; truncate=truncate, max_length=max_length, shuffle=true, 
                    drop_last=true, classification_type=classification_type, num_classes=num_classes, args=args)
        
        test_loader = load_cached_data(discrim, test_x, test_y, tokenizer; truncate=truncate, max_length=max_length, shuffle=false,
                    drop_last=true, classification_type=classification_type, num_classes=num_classes, args=args)
        
        train!(discrim, train_loader; args=args)
        test!(discrim, test_loader; args=args)
    else
        print("Creating Data Loader...")
        text, labels = shuffleobs((text, labels))
        (train_x, train_y), (test_x, test_y) = splitobs((text, labels); at=test_size)
        
        train_loader = load_data(train_x, train_y, tokenizer; truncate=truncate, max_length=max_length, shuffle=true, 
                    drop_last=true, args=args)
        
        test_loader = load_data(test_x, test_y, tokenizer; truncate=truncate, max_length=max_length, shuffle=false, 
                    drop_last=true, args=args)
        
        train!(discrim, train_loader; args=args)
        test!(discrim, test_loader; args=args)
    end    
    return discrim
end