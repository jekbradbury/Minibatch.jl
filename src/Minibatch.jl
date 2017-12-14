module Minibatch

#using Cassette

## questions for Jarrett:
# 1. it seems like the mechanism for forcing the main overdubbing @generated to recompile at each world age update is broken
# 2. why does `replace_match!` only recurse if it doesn't match?
# 3. what exactly is the difference between writing a macro like DataFlow and writing a Cassette pass?
# 4. related: is there a way now to write something kind of like a DataFlow pass but that uses type info?
# 5. specifically, I think I can use Cassette's built-in passes to rewrite `:call`s but I also want to rewrite control flow in a similarly type-sensitive way

################
# NN functions #
################

function softmax(x, axis=1)
    out = exp.(x .- maximum(x, axis))
    out ./= sum(out, axis)
    return out
end

################
# DLArray type #
################

# what is the minimum set of methods we need to forward? do we eventually want
# to switch to Cassette-style overdubbing rather than a true type wrapper here?

struct DLArray{T, E, N, B} <: AbstractArray{E, N}
    data::T
end
DLArray(x::T, varlen::NTuple{N, Bool}) where T <: AbstractArray{E, N} where {E, N} = DLArray{T, E, N, varlen}(x)

Base.ndims(x::DLArray{T, E, N, B}) where {T, E, N, B} = N
Base.size(x::DLArray) = size(x.data)
Base.getindex(x::DLArray, ind...) = getindex(x.data, ind...)

###############
# Batch types #
###############

abstract type AbstractBatch{T} end

struct VectorBatch{T} <: AbstractBatch{T}
    data::Vector{T}
end
VectorBatch(data::Vector{T}) where T <: DLArray = VectorBatch{T}(data)

Base.length(b::VectorBatch) = length(b.data)

struct SizedBatch{T, N} <: AbstractBatch{T}
    data
    sizes::Matrix{Int}
end
function SizedBatch(data::Vector{T}) where T <: DLArray{T2, E, N, B} where {T2, E, N, B}
    dims = tuple((b ? maximum(size(v, d) for v in data)
                    : size(first(data), d) for (d, b) in enumerate(B))...,
                 length(data))
    batch = fill!(similar(first(data), dims), 0)
    sizes = zeros(Int, N, length(data))
    for (i, example) in enumerate(data)
        setindex!(batch, example, indices(example)..., i)
        sizes[:, i] = collect(size(example))
    end
    return SizedBatch{T, N}(batch, sizes)
end

Base.length(b::SizedBatch) = last(size(b.data))

struct MaskedBatch{T} <: AbstractBatch{T}
    data
    mask
end

# also (at least) CatBatch using CatArrays.jl

#####################
# Cassette overdubs #
#####################

#Cassette.@context Ctx
# function mypass(signature, body)
#     Core.println("&&&&&&&&&")
#     Core.println(body)
#     body
# end
# Cassette.getpass(::Type{Ctx{T}}, Void) where T = mypass
#
# display(methods(Cassette.getpass))

#Cassette.@isprimitive Ctx (::typeof(*))(W::T, x::PadBatch{T}) where T = @show W * x
Base.:*(W, x::PadBatch{T}) where T = PadBatch{T}(W * x.data, x.mask)
function Base.:*(x::PadBatch{S}, y::PadBatch{T}) where {S, T}

end

#Cassette.@isprimitive Ctx (::typeof(+))(x::PadBatch{T}, b::T) where T = @show x .+ b
Base.broadcast(f, x::PadBatch{T}) where T = PadBatch{T}(broadcast(f, x.data), x.mask)

Base.broadcast(f, x::PadBatch{T}, y::T)           where T = PadBatch{T}(broadcast(f, x.data, y), x.mask)
Base.broadcast(f, x::T,           y::PadBatch{T}) where T = PadBatch{T}(broadcast(f, x, y.data), y.mask)
#Base.broadcast(f, x::PadBatch{T}, y::PadBatch{T}, z::PadBatch{T}) where T = PadBatch{T}(broadcast(f, x.data, y.data, z.data))
#Base.broadcast(f, x::PadBatch{T}, y::PadBatch{T}, z::T)           where T = PadBatch{T}(broadcast(f, x.data, y.data, z))
#Base.broadcast(f, x::PadBatch{T}, y::T,           z::PadBatch{T}) where T = PadBatch{T}(broadcast(f, x.data, y, z.data))
#Base.broadcast(f, x::PadBatch{T}, y::T,           z::T)           where T = PadBatch{T}(broadcast(f, x.data, y, z))
#Base.broadcast(f, x::T,           y::PadBatch{T}, z::PadBatch{T}) where T = PadBatch{T}(broadcast(f, x, y.data, z.data))
#Base.broadcast(f, x::T,           y::PadBatch{T}, z::T)           where T = PadBatch{T}(broadcast(f, x, y.data, z))
#Base.broadcast(f, x::T,           y::T,           z::PadBatch{T}) where T = PadBatch{T}(broadcast(f, x, y, z.data))
# TODO figure out how the Cassette contextual dispatch thing works here

#Cassette.@isprimitive Ctx (::typeof(btanh))(x::PadBatch) = @show btanh(x.data)

# softmax(x::PadBatch{T}, axis) where T = ...

################
# Test network #
################

W = rand(5, 5)
b = rand(5)
f(x) = tanh.(W*x .+ b)

# function attention(q, k, v) # q::N×D, k::M×D, v::M×D
#     alpha = q*k' # ::N×M
#     # TODO tri mask
#     softmax(alpha, 2)*v # ::N×D
# end

function attention(q, k, v) # q::D×N, k::D×M, v::D×M
    alpha = k'*q # ::M×N
    # TODO tri mask
    v*softmax(alpha, 1) # ::D×N
end

#f(x::AbstractBatch) = Cassette.@execute Ctx f(x);
x = rand(5)
xb = PadBatch(rand(5, 3))
display(f(x))
println()
display(f(xb))
println()


end # module
