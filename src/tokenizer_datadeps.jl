using DataDeps

"""
    init_tokenizer_datadeps()

Initialize datadeps for gpt2 tokenizer.
"""
function init_tokenizer_datadeps()
    register(DataDep("Vocab", 
    "GPT2 Pretrained Tokenizer Vocabulary file from HuggingFace.",
    "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
    "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
    ))
    
    register(DataDep("BPE",
    "GPT2 Pretrained Tokenizer bpe file from HuggingFace.",
    "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
    "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    post_fetch_method = function(fn)
            mv(fn,"gpt2-bpe.out")
        end
    ))
end
