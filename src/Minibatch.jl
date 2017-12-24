#module Minibatch

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

function softmax(x::AbstractArray, axis=1)
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
DLArray(x::T, varlen::NTuple{N, Bool}) where T<:AbstractArray{E, N} where {E, N} = DLArray{T, E, N, varlen}(x)

function (f::Union{Type{Base.size}, Type{Base.getindex}})(xs::Vararg{Union{T, AbstractArray}}) where T<:DLArray
    T(f((x isa T ? x.data : x for x in xs)...))
end
function Base.broadcast(f, xs::Vararg{Union{T, AbstractArray}}) where T<:DLArray
    T(broadcast(f, (x isa T ? x.data : x for x in xs)...))
end
Base.ndims(x::DLArray{T, E, N, B}) where {T, E, N, B} = N
# Base.size(x::DLArray) = size(x.data)
# Base.getindex(x::DLArray, ind...) = getindex(x.data, ind...)
# Base.broadcast(f, x::DLArray) = broadcast(f, x.data)
# Base.broadcast(f, x::DLArray, y::DLArray) = broadcast(f, x.data, y.data)

###############
# Batch types #
###############

abstract type AbstractBatch{T} end

struct VectorBatch{T} <: AbstractBatch{T}
    data::Vector{T}
end
VectorBatch(data::Vector{T}) where T<:DLArray = VectorBatch{T}(data)

Base.length(b::VectorBatch) = length(b.data)

struct SizedBatch{T, N} <: AbstractBatch{T}
    data
    sizes::Matrix{Int}
end
function SizedBatch(data::Vector{T}) where T<:DLArray{T2, E, N, B} where {T2, E, N, B}
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
SizedBatch(b::VectorBatch) = SizedBatch(b.data)

Base.length(b::SizedBatch) = last(size(b.data))
Base.:(==)(a::SizedBatch, b::SizedBatch) = a.data == b.data && a.sizes == b.sizes

struct MaskedBatch{T} <: AbstractBatch{T}
    data
    mask
end

# also (at least) CatBatch using CatArrays.jl

####################################
# Methods/overdubs for batch types #
####################################

function (f::Union{Type{Base.:+}, Type{Base.:*}, Type{Base.broadcast}})(xs::Vararg{Union{T, AbstractArray, Function}}) where T<:VectorBatch
    sizes = [length(x) for x in xs if x isa T]
    batchsize = first(sizes)
    all(s == batchsize for s in sizes) || error("VectorBatch size mismatch in broadcast")
    T([f((x isa T ? x.data[i] : x for x in xs)...) for i in 1:batchsize])
end
function (f::Type{Base.:+})(xs::Vararg{Union{T, AbstractArray, Function}}) where T<:VectorBatch
    sizes = [length(x) for x in xs if x isa T]
    batchsize = first(sizes)
    all(s == batchsize for s in sizes) || error("VectorBatch size mismatch in broadcast")
    T([f((x isa T ? x.data[i] : x for x in xs)...) for i in 1:batchsize])
end

#(f)(x::VectorBatch) = VectorBatch(map(f, x.data))

#Base.broadcast(f, x::T) where T<:SizedBatch = T(broadcast(f, x.data), x.sizes)
# function Base.broadcast(f, x::T, y::T) where T<:SizedBatch
#     x.sizes == y.sizes || error("binary broadcast: SizedBatch size mismatch")
#     T(broadcast(f, x.data, y.data), x.sizes)
# end
# function Base.broadcast(f, x::T, y::AbstractArray) where T<:SizedBatch
#     T(broadcast(f, x.data, y), x.sizes)
# end
function Base.broadcast(f, xs::Vararg{Union{T, AbstractArray}}) where T<:SizedBatch
    sizes = [x.sizes for x in xs if x isa T]
    all(s == first(sizes) for s in sizes) || error("SizedBatch size mismatch in broadcast")
    T(broadcast(f, (x isa T ? x.data : x for x in xs)...))
end

#################
# Test networks #
#################

W = rand(5, 5)
b = rand(5)
f(x) = tanh.(x)
g(x) = tanh.(x .+ b)
h(x) = tanh.(W*x .+ b)

data(n) = DLArray(rand(5, n), (false,true))
x1 = VectorBatch([data(3), data(4)])
x2 = SizedBatch(x1)

# println()
# display(x1)
# display(f(x1))
# println()
# display(x2)
# display(f(x2))

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
# x = rand(5)
# xb = PadBatch(rand(5, 3))
# display(f(x))
# println()
# display(f(xb))
# println()


#end # module
