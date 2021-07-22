var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = PPLM","category":"page"},{"location":"#PPLM","page":"Home","title":"PPLM","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for PPLM.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [PPLM]","category":"page"},{"location":"#PPLM.GPT2Tokenizer-Tuple{AbstractString}","page":"Home","title":"PPLM.GPT2Tokenizer","text":"Example: for vector of texts -> map(x->encode(tokenizer, x), textvector) or tokenizer.(textvector)\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.data_preprocess","page":"Home","title":"PPLM.data_preprocess","text":"data_preprocess(data_x, data_y, classification_type::String=\"Binary\", num_classes::Integer=2; args=nothing)\n\nFunction to preprocess data_x and data_y along with creating mask for the data_x. \n\nPreprocessing for data_x consist of padding with pad token (expected to be provided as args.pad_token).\n\nPreprocessing for data_y consist of creating onehotbach for data_y (if classification_type is not \"Binary\"), for 1:num_classes else reshape the data as (1, length(data_y)) \n\nReturns data_x, data_y, mask after pre-processing.\n\n\n\n\n\n","category":"function"},{"location":"#PPLM.get_artifact-Tuple{Any}","page":"Home","title":"PPLM.get_artifact","text":"get_artifact(name)\n\nUtility function to download/install the artifact in case not already installed.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_mask-Union{Tuple{AbstractMatrix{T}}, Tuple{T}, Tuple{AbstractMatrix{T}, Integer}, Tuple{AbstractMatrix{T}, Integer, Integer}} where T","page":"Home","title":"PPLM.get_mask","text":"get_mask(seq::AbstractMatrix{T}, pad_token::Integer=0, embed_size::Integer=768)\n\nFunction to create mask for sequences against padding, so as to inform the model, that some part of sequenece is padded and hence to be ignored.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_registered_file-Tuple{Any}","page":"Home","title":"PPLM.get_registered_file","text":"get_registered_file(name)\n\nFetch registered file path from Artifacts.toml, based on the artifact name.  \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_cached_data-Tuple{Union{PPLM.DiscriminatorV1, DiscriminatorV2}, Any, Any, PPLM.PretrainedTokenizer}","page":"Home","title":"PPLM.load_cached_data","text":"load_cached_data(discrim::Union{DiscriminatorV1, DiscriminatorV2}, data_x, data_y, tokenizer::PretrainedTokenizer; truncate::Bool=false, max_length::Integer=256, shuffle::Bool=false, batchsize::Int=4, drop_last::Bool=false, classification_type=\"Binary\", num_classes=2, args=nothing)\n\nReturns a DataLoader with (x, y) which can directly be feeded into classifier layer for training. \n\nThe function first loads the data using load_data function with batchsize=1, then passes each batch to the transformer model of discrim after data preprocessing, and then the average representation of the hidden_states are stored in a vector, which are then further loaded into a DataLoader, ready to use for classification training. \n\nNote: This functions saves time by cacheing the average representation of hidden states beforehand, avoiding passing the data through model in each epoch of training. This can be done as the model itself is non-trainable while training discriminator classifier head.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_data-Tuple{Any, Any, PPLM.PretrainedTokenizer}","page":"Home","title":"PPLM.load_data","text":"load_data(data_x, data_y, tokenizer::PretrainedTokenizer;  batchsize::Integer=8, truncate::Bool=false, max_length::Integer=256, shuffle::Bool=false, drop_last::Bool=false, add_eos_start::Bool=true)\n\nReturns DataLoader for the data_x and data_y after processing the datax, with batchsize=batchsize. The processing consist of tokenization of datax and further truncation to max_len if truncate is set to be true. \n\nIf add_eos_start is set to true, add EOS token of tokenizer to the start. \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_data_from_csv-Tuple{Any}","page":"Home","title":"PPLM.load_data_from_csv","text":"load_data_from_csv(path_to_csv; text_col=\"text\", label_col=\"label\", delim=',', header=1)\n\nLoad the data from a csv file based on the specified text_col column for text and label_col for target label. Returns vectors for text and label.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.pad_seq-Union{Tuple{AbstractVector{T}}, Tuple{T}, Tuple{AbstractVector{T}, Integer}} where T","page":"Home","title":"PPLM.pad_seq","text":"pad_seq(batch::AbstractVector{T}, pad_token::Integer=0)\n\nFunction to add pad tokens in shorter sequence, to make the length of each sequence equal to the max_length ( calculated as max(map(length, batch))) in the batch. Pad token defaults to 0. \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.register_custom_file-Tuple{Any, Any, Any}","page":"Home","title":"PPLM.register_custom_file","text":"register_custom_file(artifact_name, file_name, path)\n\nFunction to register custom file under artifact_name in Artifacts.toml. path expects path of the directory where the file file_name is stored. Stores the complete path to the file as Artifact URL.\n\nExample\n\nregister_custom_file('custom', 'xyz.txt','./folder/folder/')\n\n\nNote: In case this gives permission denied error, change the Artifacts.toml file permissions using  chmod(path_to_file_in_julia_installation , 0o764)or similar.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.save_classifier_head-Tuple{Any}","page":"Home","title":"PPLM.save_classifier_head","text":"\n\n\n\n","category":"method"},{"location":"#PPLM.top_k_logits-Tuple{AbstractArray, Any}","page":"Home","title":"PPLM.top_k_logits","text":"top_k_logits(logits::AbstractArray, k; prob = false)\n\nMasks everything but the k top entries as -infinity (1e10). Incase of probs=true, everthing except top-k probabilities are masked to 0.0. logits is expected to be a vector.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.truncate_-Tuple{Any, Integer}","page":"Home","title":"PPLM.truncate_","text":"truncate_(x, max_length::Integer)\n\nTruncate the data to minimum of max_length and length of x.\n\n\n\n\n\n","category":"method"}]
}
