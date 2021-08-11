# Discriminator Training

PPLM.jl allows it's used not allow to use pretrained Discriminators but also to train their custom discriminator. It provide functions ranging from Data Preprocessing, Caching to Training and the saving the discriminator.

Let's understand how to use PPLM.jl to train your own custom discriminator. 

Consider that you have a list of `text` and a corresponding list of `labels` (expected to range from 0 to n-1 where n is the number of classes) for which you want to train your discriminator on.

First let's load the PPLM package and the model:

```julia
using PPLM

tokenizer, model = PPLM.get_gpt2();

```
Once we have our model loaded, let's intialize the Hyperparameters required and the Discriminator:

```julia
args = PPLM.HyperParams(lr=5e-6,classification_type="MultiClass", epochs=50)
discrim = PPLM.get_discriminator(model; class_size=2);
```

Now you can use any of the following two Methods for training your own discriminator (usually even if its a binary problem, PPLM treat it as a Multiclass problem, same as in original repo of PPLM by Uber).

### Method 1

```julia
PPLM.train_discriminator(text, labels, 8, "Multiclass", 2; lr=5e-6, discrim=discrim, tokenizer=tokenizer, args=args, train_size=0.85, epochs=50);

# It will automatically create a test data split and evaluate the model on that data.
```

### Method 2

```julia
using StatsBase
using Random

(train_x, train_y), (test_x, test_y) = PPLM.splitobs((text_reduced, label_reduced); at=0.8);
        
train_loader = PPLM.load_cached_data(discrim, train_x, train_y, tokenizer; truncate=true, classification_type="Multiclass");

test_loader = PPLM.load_cached_data(discrim, test_x, test_y, tokenizer; truncate=true, classification_type="Multiclass");

PPLM.train!(discrim, train_loader; args=args);

PPLM.test!(discrim, test_loader; args=args)
```

## Save Discriminator

Once you have trained your Discriminator, you can save it as follows:

```julia
path = "replace_it_with_the_path_you_want_to_the_directory"
PPLM.save_discriminator(discrim, "custom_discriminator"; file_name="custom_model.bson", path= path)

```

## Load Discriminator 

To load the discriminator you saved, you can do as follows:

```julia

tokenizer, model = PPLM.get_gpt2();

path_file = joinpath(path, file_name) # path = path to the directory

discrim = PPLM.get_discriminator(model; load_from_pretrained=true, discrim="custom", path=path, class_size=2);

```

For more details, you can check out the code in the repo.