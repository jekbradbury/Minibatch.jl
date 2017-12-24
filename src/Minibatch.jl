module Minibatch

################
# NN functions #
################

function softmax(x::AbstractArray, axis=1)
    out = exp.(x .- maximum(x, axis))
    out ./= sum(out, axis)
    return out
end

#################
# AxisInfo type #
#################

# TODO non-varlen axes should have (optional?) static dimension
const AxisInfo{N} = NTuple{N, Bool}

###############
# Batch types #
###############

# currently it's intentionally not <:AbstractArray
# because that provides a lot of magic I want to avoid
abstract type AbstractBatch{T, A} end

struct VectorBatch{T, A} <: AbstractBatch{T, A}
    data::Vector{T}
end
VectorBatch(data::Vector{T}, axes::AxisInfo) where {T} = VectorBatch{T, axes}(data)

Base.length(b::VectorBatch) = length(b.data)

struct SizedBatch{T, A} <: AbstractBatch{T, A}
    data
    sizes::Matrix{Int}
end
function SizedBatch(data::Vector{T}, axes::AxisInfo{N}) where {T, N}
    dims = tuple((b ? maximum(size(v, d) for v in data)
                    : size(first(data), d) for (d, b) in enumerate(axes))...,
                 length(data))
    batch = fill!(similar(first(data), dims), 0)
    sizes = zeros(Int, N, length(data))
    for (i, example) in enumerate(data)
        setindex!(batch, example, indices(example)..., i)
        sizes[:, i] = collect(size(example))
    end
    return SizedBatch{T, axes}(batch, sizes)
end
SizedBatch(b::VectorBatch{T, A}) where {T, A} = SizedBatch(b.data, A)

# TODO make batch size a part of the type when we have static dims
Base.length(b::SizedBatch) = last(size(b.data))
Base.:(==)(a::SizedBatch, b::SizedBatch) = a.data == b.data && a.sizes == b.sizes

# TODO
struct MaskedBatch{T, A} <: AbstractBatch{T, A}
    data
    mask
end

# TODO also CatBatch using CatArrays.jl?

####################################
# Methods/overdubs for batch types #
####################################

function _checkbatchsizes(xs::Vararg{AbstractBatch})
    bs = length(first(xs))
    all(length(x) == bs for x in xs) || error("batch size mismatch")
    bs
end
function _vbcall(f, T, xs...)
    batchsize = _checkbatchsizes((x for x in xs if x isa T)...)
    T([f((x isa T ? x.data[i] : x for x in xs)...) for i in 1:batchsize])
end
Base.:+(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(+, T, xs...)
Base.:-(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(-, T, xs...)
Base.:*(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(*, T, xs...)
Base.:/(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(/, T, xs...)
Base.getindex(x::T, inds...) where T<:VectorBatch = _vbcall(getindex, T, x, inds...)
Base.broadcast(f, xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(broadcast, T, f, xs...)

function _checksizes(xs::Vararg{SizedBatch})
    fs = first(xs).sizes
    all(x.sizes == fs for x in xs) || error("SizedBatch size mismatch")
    fs
end
function Base.broadcast(f, xs::Vararg{Union{T, AbstractArray}}) where T<:SizedBatch
    sizes = _checksizes((x for x in xs if x isa T)...)
    T(broadcast(f, (x isa T ? x.data : x for x in xs)...), sizes)
end
Base.:+(xs::Vararg{Union{T, AbstractArray}}) where T<:SizedBatch = broadcast(+, xs...)
Base.:-(xs::Vararg{Union{T, AbstractArray}}) where T<:SizedBatch = broadcast(-, xs...)

function Base.dot(a::SizedBatch{T, A}, b::SizedBatch{T, B}) where {T<:AbstractVector, A, B}
    batchsize = last(size(_checksizes(a, b)))
    data = sum(a.data .* b.data, 1)[1, :] # TODO dotBatched?
    SizedBatch{T, ()}(data, Matrix{Int}(0, batchsize))
end
function Base.dot(a::AbstractVector, b::SizedBatch{T, (false,)}) where {T}
    SizedBatch{T, ()}(b.data'*a, Matrix{Int}(0, length(b)))
end
function Base.dot(a::SizedBatch{T, (false,)}, b::AbstractVector) where {T}
    SizedBatch{T, ()}(a.data'*b, Matrix{Int}(0, length(a)))
end

function Base.:*(a::AbstractMatrix, b::SizedBatch{T, (false,)}) where {T}
    s1, bs = size(b.data)
    data = reshape(a * reshape(b.data, s1*bs), size(a, 1), bs)
    sizes = fill(similar(b.sizes), size(a, 1))
    SizedBatch{T, (false,)}(data, sizes)
end
function Base.:*(a::SizedBatch{T, A}, b::AbstractVector) where {T<:AbstractMatrix, A}
    A[2] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(a.data)
    data = reshape(reshape(a.data, s1, s2*bs) * b, s1, bs)
    SizedBatch{T, A}(data, b.sizes[1:1, :])
end

function Base.:*(a::AbstractMatrix, b::SizedBatch{T, (false,)}) where {T}
    sizes = fill(similar(b.sizes), size(a, 1))
    SizedBatch{T, (false,)}(a * b.data, sizes)
end
function Base.:*(a::AbstractMatrix, b::SizedBatch{T, B}) where {T<:AbstractMatrix, B}
    B[1] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(b.data)
    data = reshape(a * reshape(b.data, s1, s2*bs), size(a, 1), s2, bs)
    sizes = deepcopy(b.sizes)
    sizes[1, :] .= size(a, 1)
    SizedBatch{T, B}(data, sizes)
end
function Base.:*(a::SizedBatch{T, A}, b::AbstractMatrix) where {T<:AbstractMatrix, A}
    A[2] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(a.data)
    data = reshape(reshape(a.data, s1, s2*bs) * b, s1, size(b, 2), bs)
    sizes = deepcopy(b.sizes)
    sizes[2, :] .= size(b, 2)
    SizedBatch{T, A}(data, sizes)
end

# these require gemmBatched/gemm_batched:
# Base.:*(a::SizedBatch{T, A}, b::SizedBatch{T, B}) where {T<:AbstractMatrix, A, B}
# Base.:*(a::SizedBatch{T1, A}, b::SizedBatch{T2, B}) where {T1<:AbstractMatrix, T2<:AbstractVector, A, B}
# Base.:*(a::SizedBatch{T1, A}, b::SizedBatch{T2, B}) where {T1<:AbstractVector, T2<:AbstractMatrix, A, B}

#################
# Test networks #
#################

W = rand(5, 5)
b = rand(5)
f(x) = tanh.(x)
g(x) = tanh.(x .+ b)
h(x) = tanh.(W*x .+ b)

axisinfo = (false, true)
data(n) = rand(5, n)
x1 = VectorBatch([data(3), data(4)], axisinfo)
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


end # module
