# Inside make.jl
push!(LOAD_PATH,"../src/")
push!(LOAD_PATH,"../")
using Documenter, PRSFNN

makedocs(
         sitename = "PRSFNN.jl",
         modules  = [PRSFNN],
         checkdocs = :exports,
         #remotes = nothing,
         pages=[
                "Home" => "index.md"
               ]
)

deploydocs(;
    repo="github.com/weinstockj/PRS",
    devbranch = "master"
)
