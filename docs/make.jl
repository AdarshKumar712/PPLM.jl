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
    ],
)

deploydocs(;
    repo="github.com/adarshkumar712/PPLM.jl",
)
