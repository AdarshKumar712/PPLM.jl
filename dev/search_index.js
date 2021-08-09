var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = PPLM","category":"page"},{"location":"#PPLM","page":"Home","title":"PPLM","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for PPLM.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [PPLM]","category":"page"},{"location":"#PPLM.ClassifierHead","page":"Home","title":"PPLM.ClassifierHead","text":"struct ClassifierHead     linearlayer::Dense     embedsize::Int     class_size::Int end\n\nStruct for ClassifiedHead, defined with a single linear layer and two paramters: embedsize-> Size of Embedding, classsize->Number of classes.\n\n\n\n\n\n","category":"type"},{"location":"#PPLM.ClassifierHead-2","page":"Home","title":"PPLM.ClassifierHead","text":"ClassifierHead(class_size::Int=1, embed_size::Int=768; load_from_pretrained=false, discrim=nothing, file_name=nothing, path=nothing)\n\nFunction to initiate ClassifierHead layer, with defined classsize and embedsize. If load_from_pretrained is set to be true, load model from pretrained models (based on discrim name) or from specified path. \n\n\n\n\n\n","category":"type"},{"location":"#PPLM.ClassifierHead-Tuple{Any}","page":"Home","title":"PPLM.ClassifierHead","text":"(m::ClassifierHead)(hidden_state)\n\nMethod for passing hidden state through Linear layer of ClassifierHead m, return the final logits.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.DiscriminatorV1","page":"Home","title":"PPLM.DiscriminatorV1","text":"struct DiscriminatorV1     clhead::ClassifierHead     model     embedsize::Int end\n\nStruct to contain the model and ClassifierHead of PPLM model, to be used in masked training.\n\n\n\n\n\n","category":"type"},{"location":"#PPLM.DiscriminatorV2","page":"Home","title":"PPLM.DiscriminatorV2","text":"struct DiscriminatorV2     clhead::ClassifierHead     model     embedsize::Int end\n\nStruct to contain the model and ClassifierHead of PPLM model, to be used in masked training.\n\n\n\n\n\n","category":"type"},{"location":"#PPLM.GPT2Tokenizer","page":"Home","title":"PPLM.GPT2Tokenizer","text":"struct GPT2Tokenizer <: GPT2     encoder::Vocabulary{String}     bpeencode::GenericBPE     bpedecode::UnMap     vocab::Dict{String, Any}     unktoken::String     unkid::Int        eostoken::String     eostokenid::Int     padtoken::String     padtokenid::Int end\n\nStructure to hold all essential information / functions for GPT2 tokenizer\n\n\n\n\n\n","category":"type"},{"location":"#PPLM.GPT2Tokenizer-Tuple{AbstractString}","page":"Home","title":"PPLM.GPT2Tokenizer","text":"(t::GPT2Tokenizer)(text::AbstractString; add_prefix_space=false)\n\nEncode the text with tokenizer and returns the encoded vector. If add_prefix_space=true, add space at the start of 'text' before tokenization. \n\nExamples:\n\nFor a single text:\n\ntokenizer(text; add_prefix_space=true)\n\nFor vector of texts, use:\n\nmap(x->encode(tokenizer, x), text_vector) \n\n# or\n\ntokenizer.(text_vector)\n\nAlso checkout encode\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.HyperParams","page":"Home","title":"PPLM.HyperParams","text":"@withkw struct HyperParams     batchsize::Int=8     eostoken::Int=50527     padtoken::Int=0     embedsize::Int=768     classificationtype::String=\"Binary\"     targetclass=1     numclasses::Int=2     cached::Bool=true     lr::Float64=1e-6     epochs::Int=20     loginterval::Int=100 end\n\nStructure for HyperParams to select hyperparamters for training the discriminator model. \n\n\n\n\n\n","category":"type"},{"location":"#PPLM.avg_representation-Tuple{DiscriminatorV2, Any, Any}","page":"Home","title":"PPLM.avg_representation","text":"avg_representation(m::DiscriminatorV2, input_ids, mask; args=nothing)\n\nFunction to create masked average representation of Input tokens input_ids, using the DiscriminatorV2 m.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.avg_representation-Tuple{PPLM.DiscriminatorV1, Any}","page":"Home","title":"PPLM.avg_representation","text":"avg_representation(m::DiscriminatorV1, input_ids; args=nothing)\n\nFunction to create average representation (without mask) of Input tokens input_ids, using the DiscriminatorV1 m.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.binary_accuracy-Tuple{Any, Any}","page":"Home","title":"PPLM.binary_accuracy","text":"binary_accuracy(y_pred, y_true; threshold=0.5)\n\nCalculates Averaged Binary Accuracy based on y_pred and y_true. Argument threshold is used to specify the minimum predicted probability y_pred required to be labelled as 1. Default value set as 0.5.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.build_bow_ohe-Tuple{Any, Any}","page":"Home","title":"PPLM.build_bow_ohe","text":"build_bow_ohe(bow_indices, tokenizer)\n\nBuild and return a list of one_hot_matrix for each Bag Of Words list from indices. Each item of the list is of dimension (num_of_BoW_words, tokenizer.vocab_size). \n\nNote: While building the OHE of word indices, it only keeps those words, which have length 1 after tokenization and discard the rest.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.categorical_accuracy-Tuple{Any, Any}","page":"Home","title":"PPLM.categorical_accuracy","text":"categorical_accuracy(y_pred, y_true)\n\nCalculates Averaged Categorical Accuracy based on y_pred and y_true.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.data_preprocess","page":"Home","title":"PPLM.data_preprocess","text":"data_preprocess(data_x, data_y, classification_type::String=\"Binary\", num_classes::Integer=2; args=nothing)\n\nFunction to preprocess data_x and data_y along with creating mask for the data_x. \n\nPreprocessing for data_x consist of padding with pad token (expected to be provided as args.pad_token).\n\nPreprocessing for data_y consist of creating onehotbach for data_y (if classification_type is not \"Binary\"), for 1:num_classes else reshape the data as (1, length(data_y)) \n\nReturns data_x, data_y, mask after pre-processing.\n\n\n\n\n\n","category":"function"},{"location":"#PPLM.decode-Tuple{PPLM.GPT2Tokenizer, Vector{Int64}}","page":"Home","title":"PPLM.decode","text":"decode(t::GPT2Tokenizer, tokens_ids::Vector{Int})\n\nReturn decoded vector of string tokens from the indices vector tokens_ids, using the tokenizer t encoder .\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.decode-Union{Tuple{T}, Tuple{Transformers.Basic.Vocabulary{T}, Vector{Int64}}} where T","page":"Home","title":"PPLM.decode","text":"decode(vocab::Vocabulary{T}, is::Vector{Int}) where T\n\nReturn decoded vector of string tokens from the indices vector is, using the vocab.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.detokenize-Tuple{PPLM.GPT2Tokenizer, Vector{Int64}}","page":"Home","title":"PPLM.detokenize","text":"detokenize(t::GPT2Tokenizer, tokens_ids::Vector{Int})\n\nDecode and Detokenize the vector of indices token_ids. Returns the final sentence after detokenization.\n\nExample\n\nFor single vector of token_ids:\n\ndetokenize(tokenizer, token_ids)\n\nFor vector of vector of token_ids, use:\n\nmap(x->decode(tokenizer, x), tokens_id_vector_of_vector)\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.detokenize-Tuple{PPLM.GPT2Tokenizer, Vector{String}}","page":"Home","title":"PPLM.detokenize","text":"detokenize(t::GPT2Tokenizer, tokens::Vector{String})\n\nBPE Decode the vector of strings, using the tokenizer t.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.encode-Tuple{PPLM.GPT2Tokenizer, AbstractString}","page":"Home","title":"PPLM.encode","text":"encode(t::GPT2Tokenizer, text::AbstractString; add_prefix_space=false)\n\nReturns the encoded vector of tokens (mapping from vocab of Tokenizer) for text. If add_prefix_space=true, add space at the start of 'text' before tokenization. \n\nExample\n\nFor single text:\n\nencode(tokenizer, text)\n\nFor vector of text:\n\nmap(x->encode(tokenizer, x), text_vector) \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.encode-Tuple{PPLM.GPT2Tokenizer, Vector{String}}","page":"Home","title":"PPLM.encode","text":"encode(t::GPT2Tokenizer, tokens::Vector{String})\n\nFunction to encode tokens vectors to their integer mapping from vocab of tokenizer.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_artifact-Tuple{Any}","page":"Home","title":"PPLM.get_artifact","text":"get_artifact(name)\n\nUtility function to download/install the artifact in case not already installed.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_bow_indices-Tuple{Vector{String}, Any}","page":"Home","title":"PPLM.get_bow_indices","text":"get_bow_indices(bow_key_or_path_list::Vector{String}, tokenizer)\n\nReturns a list of list of indices of words from each Bag of word in the bow_key_or_path_list, after tokenization. The functions looks for provided BoW key in the registered artifacts Artifacts.toml file. In case not present there, function expects that bow_key is provided as the complete path to the file the URL to download .txt file.\n\nExample\n\nget_bow_indices([\"legal\", \"military\"])\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_discriminator-Tuple{Any}","page":"Home","title":"PPLM.get_discriminator","text":"get_discriminator(model; load_from_pretrained=false, discrim=nothing, file_name=nothing, version=2, class_size::Int=1, embed_size::Int=768, path=nothing)\n\nFunction to create discriminator based on provided model. Incase, load_from_pretrained is set to be true, loads ClassifierHead layer from pretrained models or path provided.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_gpt2-Tuple{}","page":"Home","title":"PPLM.get_gpt2","text":"get_gpt2()\n\nFunction to load gpt2 lmheadmodel along with the tokenizer.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_gpt2_medium-Tuple{}","page":"Home","title":"PPLM.get_gpt2_medium","text":"get_gpt2_medium()\n\nFunction to load gpt2-medium lmhead model along with the tokenizer.\n\nNote: In case this function gives error of permission denied, try changing the file permissions for the Artifacts.toml file of Transformers.jl package (as it is read only by default) under the src/huggingface folder. \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_mask-Union{Tuple{AbstractMatrix{T}}, Tuple{T}, Tuple{AbstractMatrix{T}, Integer}, Tuple{AbstractMatrix{T}, Integer, Integer}} where T","page":"Home","title":"PPLM.get_mask","text":"get_mask(seq::AbstractMatrix{T}, pad_token::Integer=0, embed_size::Integer=768)\n\nFunction to create mask for sequences against padding, so as to inform the model, that some part of sequenece is padded and hence to be ignored.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.get_registered_file-Tuple{Any}","page":"Home","title":"PPLM.get_registered_file","text":"get_registered_file(name)\n\nFetch registered file path from Artifacts.toml, based on the artifact name.  \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.init_tokenizer_datadeps-Tuple{}","page":"Home","title":"PPLM.init_tokenizer_datadeps","text":"init_tokenizer_datadeps()\n\nInitialize datadeps for gpt2 tokenizer.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_cached_data-Tuple{Union{PPLM.DiscriminatorV1, DiscriminatorV2}, Any, Any, PPLM.PretrainedTokenizer}","page":"Home","title":"PPLM.load_cached_data","text":"load_cached_data(discrim::Union{DiscriminatorV1, DiscriminatorV2}, data_x, data_y, tokenizer::PretrainedTokenizer; truncate::Bool=false, max_length::Integer=256, shuffle::Bool=false, batchsize::Int=4, drop_last::Bool=false, classification_type=\"Binary\", num_classes=2, args=nothing)\n\nReturns a DataLoader with (x, y) which can directly be feeded into classifier layer for training. \n\nThe function first loads the data using load_data function with batchsize=1, then passes each batch to the transformer model of discrim after data preprocessing, and then the average representation of the hidden_states are stored in a vector, which are then further loaded into a DataLoader, ready to use for classification training. \n\nNote: This functions saves time by cacheing the average representation of hidden states beforehand, avoiding passing the data through model in each epoch of training. This can be done as the model itself is non-trainable while training discriminator classifier head.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_data-Tuple{Any, Any, PPLM.PretrainedTokenizer}","page":"Home","title":"PPLM.load_data","text":"load_data(data_x, data_y, tokenizer::PretrainedTokenizer;  batchsize::Integer=8, truncate::Bool=false, max_length::Integer=256, shuffle::Bool=false, drop_last::Bool=false, add_eos_start::Bool=true)\n\nReturns DataLoader for the data_x and data_y after processing the datax, with batchsize=batchsize. The processing consist of tokenization of datax and further truncation to max_len if truncate is set to be true. \n\nIf add_eos_start is set to true, add EOS token of tokenizer to the start. \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_data_from_csv-Tuple{Any}","page":"Home","title":"PPLM.load_data_from_csv","text":"load_data_from_csv(path_to_csv; text_col=\"text\", label_col=\"label\", delim=',', header=1)\n\nLoad the data from a csv file based on the specified text_col column for text and label_col for target label. Returns vectors for text and label.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_pretrained_tokenizer-NTuple{5, Any}","page":"Home","title":"PPLM.load_pretrained_tokenizer","text":"load_pretrained_tokenizer(path_bpe, path_vocab, unk_token, eos_token, pad_token)\n\nLoad pretrained tokenizer for GPT2 from provided bpe and vocab file path. Initialises unk_token, eos_token, pad_token as provided with the function. Returns tokenizer as GPT2Tokenizer structure.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.load_pretrained_tokenizer-Union{Tuple{Type{T}}, Tuple{T}} where T<:PPLM.PretrainedTokenizer","page":"Home","title":"PPLM.load_pretrained_tokenizer","text":"load_pretrained_tokenizer(ty::Type{T}; unk_token=\"<|endoftext|>\", eos_token=\"<|endoftext|>\", pad_token=\"<|endoftext|>\") where T<:PretrainedTokenizer\n\nLoad GPT2 tokenizer using Datadeps for pretrained bpe and vocab. Returns tokenizer as GPT2Tokenizer structure.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.nucleus_sample-Tuple{Any}","page":"Home","title":"PPLM.nucleus_sample","text":"nucleus_sample(probs; p=0.8)\n\nNuclues sampling function, to return after sampling reverse sorted probabilities probs till the index, where cumulative probability remains less than provided p. It removes tokens with cumulative probability above the threshold p before sampling.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.pad_seq-Union{Tuple{AbstractVector{T}}, Tuple{T}, Tuple{AbstractVector{T}, Integer}} where T","page":"Home","title":"PPLM.pad_seq","text":"pad_seq(batch::AbstractVector{T}, pad_token::Integer=0)\n\nFunction to add pad tokens in shorter sequence, to make the length of each sequence equal to the max_length ( calculated as max(map(length, batch))) in the batch. Pad token defaults to 0. \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.register_custom_file-Tuple{Any, Any, Any}","page":"Home","title":"PPLM.register_custom_file","text":"register_custom_file(artifact_name, file_name, path)\n\nFunction to register custom file under artifact_name in Artifacts.toml. path expects path of the directory where the file file_name is stored. Stores the complete path to the file as Artifact URL.\n\nExample\n\nregister_custom_file('custom', 'xyz.txt','./folder/folder/')\n\n\nNote: In case this gives permission denied error, change the Artifacts.toml file permissions using  chmod(path_to_file_in_julia_installation , 0o764)or similar.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.sample_normal-Tuple{}","page":"Home","title":"PPLM.sample_normal","text":"sample_normal(;prompt=\"I hate the customs\", tokenizer=nothing, model=nothing, max_length=100, method=\"top_k\", k=50, t=1.2, p=0.5, add_eos_start=true)\n\nFunction to generate normal Sentences with model and tokenizer provided. In case not provided, function itself create instance of GPT2-small tokenizer and LM Head Model. The sentences are started with the provided prompt, and generated till token length reaches max_length.\n\nTwo sampling methods of generation are provided with this function:\n\nmethod='top_k'\nmethod='nucleus'\n\nAny of these methods can be used provided with either k or p.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.sample_pplm-Tuple{Any}","page":"Home","title":"PPLM.sample_pplm","text":"function sample_pplm(pplm_args;tokenizer=nothing, model=nothing, prompt=\"I hate the customs\", add_eos_start=true)\n\nFunction for PPLM model based generation. Generate perturbed sentence using pplm_args, tokenizer and model (GPT2, in case not provided), starting with prompt. In this function the generation is based on the arguments/parameters provided in pplm_args, which is an instance of pplm struct.  \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.save_classifier_head-Tuple{Any}","page":"Home","title":"PPLM.save_classifier_head","text":"save_classifier_head(cl_head; file_name=nothing, path=nothing, args=nothing, register_discrim=true, discrim_name=\"\")\n\nFunction to save the ClassifiedHead as a BSON once the training is complete, based on the path provided. In case path is set as nothing, it saves the discriminators in ./pretrained_discriminators folder relative to the package directory.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.save_discriminator","page":"Home","title":"PPLM.save_discriminator","text":"save_discriminator(discrim, discrim_name=\"Custom\"; file_name=nothing, path=nothing, args=nothing)\n\nFunction to save ClassifiedHead part of discriminator (by calling save_classifier_head function), which is the only trainable part of discriminator\n\n\n\n\n\n","category":"function"},{"location":"#PPLM.set_device","page":"Home","title":"PPLM.set_device","text":"set_device(d_id=0)\n\nFunction to set cuda device if available and also to disallow scalar operations\n\n\n\n\n\n","category":"function"},{"location":"#PPLM.test!-Tuple{Any, Any}","page":"Home","title":"PPLM.test!","text":"test!(discrim, data_loader; args=nothing)\n\nTest the discriminator on test data provided using data_loader, based on Accuracy and NLL Loss. \n\n\n\n\n\n","category":"method"},{"location":"#PPLM.tokenize-Tuple{PPLM.GPT2Tokenizer, AbstractString}","page":"Home","title":"PPLM.tokenize","text":"tokenize(t::GPT2Tokenizer, text::AbstractString)\n\nFunction to tokenize given text with tokenizer bpe encoder (t.bpe_encode). Returns a string vector of tokens.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.top_k_logits-Tuple{AbstractArray, Any}","page":"Home","title":"PPLM.top_k_logits","text":"top_k_logits(logits::AbstractArray, k; prob = false)\n\nMasks everything but the k top entries as -infinity (1e10). Incase of probs=true, everthing except top-k probabilities are masked to 0.0. logits is expected to be a vector.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.top_k_sample-Tuple{Any}","page":"Home","title":"PPLM.top_k_sample","text":"top_k_sample(probs; k=10)\n\nSampling function to return index from top_k probabilities, based on provided k. Function removes all tokens with a probability less than the last token of the top_k before sampling.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.train!-Tuple{Any, Any}","page":"Home","title":"PPLM.train!","text":"train!(discrim, data_loader; args=args)\n\nTrain the discriminator using the provided data_loader training data and arguments args provided.\n\n\n\n\n\n","category":"method"},{"location":"#PPLM.train_discriminator","page":"Home","title":"PPLM.train_discriminator","text":"train_discriminator(text, labels, batchsize::Int=8, classification_type::String=\"Binary\", num_classes::Int=2; model=\"gpt2\", cached::Bool=true, discrim=nothing, tokenizer=nothing, truncate=true, max_length=256, train_size::Float64=0.9, lr::Float64=1e-5, epochs::Int=10, args=nothing)\n\nFunction to train discriminator for provided text and target labels, based on set of function paramters provided. Returns discrim discriminator after training.\n\nHere the cached=true allows cacheing of contexualized embeddings (forward pass) in GPT2 model, as the model itself is non-trainable. This reduces the time of training effectively as the forward pass through GPT2 model is to be done only once.\n\nExample\n\nConsider a Multiclass classification problem with class size of 5, it can trained on text and labels vectors using:\n\ntrain_discriminator(text, labels, 16, \"Multiclass\", 5)\n\n\n\n\n\n","category":"function"},{"location":"#PPLM.truncate_-Tuple{Any, Integer}","page":"Home","title":"PPLM.truncate_","text":"truncate_(x, max_length::Integer)\n\nTruncate the data to minimum of max_length and length of x.\n\n\n\n\n\n","category":"method"}]
}
