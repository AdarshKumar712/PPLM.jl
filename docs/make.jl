using PPLM
using Documenter

DocMeta.setdocmeta!(PPLM, :DocTestSetup, :(using PPLM); recursive=true)

makedocs(;
    modules=[PPLM],
    authors="Adarsh Kumar",
    repo="https://github.com/adarshkumar712/PPLM.jl/blob/{commit}{path}#{line}",
    sitename="PPLM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://adarshkumar712.github.io/PPLM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "How does it work?" => "how.md",
        "GPT2: Tokenization and Generation" => "gpt2.md",
        "Bag Of Words Model" => "bow.md",
        "Discriminator Model" => "discrim.md",
        "Discriminator Training" => "discrim_train.md",
        "API" => "api.md",
        "Contact Info" => "contact.md",
    ],
)

deploydocs(;
    repo="github.com/AdarshKumar712/PPLM.jl",
    devbranch = "main",
)
