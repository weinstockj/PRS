# Inside make.jl
push!(LOAD_PATH,"../src/")
using Documenter, PRSFNN

makedocs(
         sitename = "PRSFNN.jl",
         modules  = [PRSFNN],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/weinstockj/PRS",
    devbranch = "master"
)
