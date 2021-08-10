# API Functions

Here are some of the API functions provided with this package:

## GPT2 Tokenizer

```@docs
PPLM.load_pretrained_tokenizer
PPLM.tokenize
PPLM.encode
PPLM.decode
PPLM.detokenize
```

## Discriminator Model

### General

```@docs
PPLM.ClassifierHead
PPLM.get_discriminator
PPLM.save_classifier_head
PPLM.save_discriminator
```

### Data Processing

```@docs
PPLM.pad_seq
PPLM.get_mask
PPLM.data_preprocess
PPLM.load_data
PPLM.load_cached_data
PPLM.load_data_from_csv
```

### Training

```@docs
PPLM.train!
PPLM.test!
PPLM.train_discriminator
```

## Bag of Words

```@docs
PPLM.get_bow_indices
PPLM.build_bow_ohe
```

## Generation

### Normal

```@docs
PPLM.sample_normal
```

### PPLM

```@docs
PPLM.sample_pplm
PPLM.perturb_probs
PPLM.perturb_hidden_bow
PPLM.perturb_past_bow
PPLM.perturb_hidden_discrim
PPLM.perturb_past_discrim
```

## Utils

```@docs
PPLM.get_gpt2
PPLM.get_gpt2_medium
PPLM.set_device
PPLM.get_registered_file
PPLM.get_artifact
PPLM.register_custom_file
PPLM.top_k_sample
PPLM.nucleus_sample
PPLM.binary_accuracy
PPLM.categorical_accuracy
```