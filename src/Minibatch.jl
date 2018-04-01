module Minibatch

export VectorBatch, MaskedBatch, ϵ

using NNlib
using Flux

################
# NN functions #
################

const ϵ = 1e-7

function NNlib.softmax(x::AbstractArray, axis=1)
    out = exp.(x .- maximum(x, axis))
    out ./= sum(out, axis) .+ ϵ
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
axisinfo(::AbstractBatch{T, A}) where {T, A} = A

struct VectorBatch{T, A} <: AbstractBatch{T, A}
    data::Vector{T}
end
VectorBatch(data::Vector{T}, axes::AxisInfo) where {T} = VectorBatch{T, axes}(data)

Base.length(b::VectorBatch) = length(b.data)
Base.:(==)(a::VectorBatch{T1, A}, b::VectorBatch{T2, A}) where {T1, T2, A} = a.data == b.data
Base.isapprox(a::VectorBatch{T1, A}, b::VectorBatch{T2, A}) where {T1, T2, A} = a.data ≈ b.data

# TODO unsure if it should be mutable
mutable struct MaskedBatch{T, A} <: AbstractBatch{T, A}
    data
    mask
end
function MaskedBatch(data::AbstractVector, axes::AxisInfo)
    dims = (a ? maximum(size(v, d) for v in data) : size(first(data), d) for (d, a) in enumerate(axes))
    maskdims = (a ? maximum(size(v, d) for v in data) : 1 for (d, a) in enumerate(axes))
    batch = fill!(similar(first(data), dims..., length(data)), 0)
    mask = fill!(similar(first(data), maskdims..., length(data)), 0)
    for (i, x) in enumerate(data)
        setindex!(batch, x, indices(x)..., i)
        mask[(1:(a ? size(x, d) : 1) for (d, a) in enumerate(axes))..., i] .= 1
    end
    return MaskedBatch{typeof(first(data)), axes}(batch, mask)
end
function sizes(b::MaskedBatch{T, A}) where {T, A}
    bs = length(b)
    mask = Array{Int}(b.mask)
    return vcat((a ? sum(mask, d)[(1 for _ in A)..., :]' :
                 fill(size(b.data, d), bs)' for (d, a) in enumerate(A))...)
end
function VectorBatch(b::MaskedBatch{T, A}) where {T, A}
    s = sizes(b)
    xs = [b.data[(1:n for n in s[:, i])..., i] for i in 1:length(b)]
    return VectorBatch{T, A}(xs)
end

MaskedBatch(b::VectorBatch{T, A}) where {T, A} = MaskedBatch(b.data, A)
Base.length(b::MaskedBatch) = last(size(b.mask))
Base.:(==)(a::MaskedBatch{T1, A}, b::MaskedBatch{T2, A}) where {T1, T2, A} = a.data == b.data && a.mask == b.mask
Base.isapprox(a::MaskedBatch{T1, A}, b::MaskedBatch{T2, A}) where {T1, T2, A} = a.data ≈ b.data && a.mask == b.mask

# TODO also CatBatch using CatArrays.jl?

####################################
# Methods/overdubs for batch types #
####################################

function _samebatchsizes(xs::Vararg{AbstractBatch})
    bs = length(first(xs))
    all(length(x) == bs for x in xs) || error("batch size mismatch")
    return bs
end
function _vbcall(f, T, xs...)
    bs = _samebatchsizes((x for x in xs if x isa T)...)
    return T([f((x isa T ? x.data[i] : x for x in xs)...) for i in 1:bs])
end
Base.:+(xs::Vararg{Union{T, AbstractArray, Number}}) where T<:VectorBatch = _vbcall(+, T, xs...)
Base.:-(xs::Vararg{Union{T, AbstractArray, Number}}) where T<:VectorBatch = _vbcall(-, T, xs...)
Base.:*(xs::Vararg{Union{T, AbstractArray, Number}}) where T<:VectorBatch = _vbcall(*, T, xs...)
Base.:/(xs::Vararg{Union{T, AbstractArray, Number}}) where T<:VectorBatch = _vbcall(/, T, xs...)
Base.getindex(x::T, inds...) where T<:VectorBatch = _vbcall(getindex, T, x, inds...)
Base.broadcast(f, xs::Vararg{Union{T, AbstractArray, Number}}) where T<:VectorBatch = _vbcall(broadcast, T, f, xs...)
function Base.transpose(b::VectorBatch{T, A}) where {T, A}
    axes = tuple(A[2], A[1], A[3:end]...)
    return VectorBatch{T, axes}([transpose(x) for x in b.data])
end
softmax(x::T, axis=1) where T<:VectorBatch = _vbcall(softmax, T, x, axis)

# _vbcall is unsafe for contractions because it doesn't check that axes
# which must have static dimension actually do at the type level. We
# could fix that by including explicit checks, as in the methods below,
# but even when _vbcall wouldn't break.
function Base.:*(a::VectorBatch{T, A}, b::AbstractVector) where {T<:AbstractMatrix, A}
    A[2] == false || error("cannot contract axes with static and dynamic dimension")
    return VectorBatch{T, (A[1],)}([a.data[i] * b for i in 1:length(a)])
end

function Base.:*(a::VectorBatch{T, A}, b::VectorBatch{T, B}) where {T<:AbstractMatrix, A, B}
    A[2] == B[1] || error("cannot contract axes with static and dynamic dimension")
    bs = _samebatchsizes(a, b)
    return VectorBatch{T, (A[1], B[2])}([a.data[i] * b.data[i] for i in 1:bs])
end

function Base.:*(a::VectorBatch{T1, A}, b::VectorBatch{T2, B}) where {T1<:AbstractMatrix, T2<:AbstractVector, A, B}
    A[2] == B[1] || error("cannot contract axes with static and dynamic dimension")
    bs = _samebatchsizes(a, b)
    return VectorBatch{T, (A[1],)}([a.data[i] * b.data[i] for i in 1:bs])
end

_rezero!(b::MaskedBatch) = b.data .*= b.mask
# TODO this should really be promote_containersize
function _samemasks(xs::Vararg{<:MaskedBatch{T2, B}}) where {T2, B}
    fs = first(xs).mask
    if any(B)
        all(x.mask == fs for x in xs) || error("MaskedBatch size mismatch")
    end
    return fs
end
function Base.broadcast(f, xs::Vararg{Union{T, AbstractArray, Number}}) where T<:MaskedBatch
    mask = _samemasks((x for x in xs if x isa T)...)
    res = T(broadcast(f, (x isa T ? x.data : x for x in xs)...), mask)
    _rezero!(res)
    return res
end
Base.:+(xs::Vararg{Union{T, AbstractArray, Number}}) where T<:MaskedBatch = broadcast(+, xs...)
Base.:-(xs::Vararg{Union{T, AbstractArray, Number}}) where T<:MaskedBatch = broadcast(-, xs...)

# contractions between axes with static dimension

function Base.dot(a::AbstractVector, b::MaskedBatch{T, (false,)}) where {T}
    return MaskedBatch{T, ()}(b.data'*a, b.mask[1])
end
function Base.dot(a::MaskedBatch{T, (false,)}, b::AbstractVector) where {T}
    return MaskedBatch{T, ()}(a.data'*b, a.mask[1])
end

function Base.:*(a::AbstractMatrix, b::MaskedBatch{T, (false,)}) where {T}
    return MaskedBatch{T, (false,)}(a * b.data, b.mask)
end
function Base.:*(a::MaskedBatch{T, A}, b::AbstractVector) where {T<:AbstractMatrix, A}
    A[2] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(a.data)
    data = reshape(reshape(a.data, s1, s2*bs) * b, s1, bs)
    return MaskedBatch{T, (A[1],)}(data, a.mask[:, 1, :])
end

function Base.:*(a::AbstractMatrix, b::MaskedBatch{T, B}) where {T<:AbstractMatrix, B}
    B[1] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(b.data)
    data = reshape(a * reshape(b.data, s1, s2*bs), size(a, 1), s2, bs)
    return MaskedBatch{T, B}(data, b.mask)
end
function Base.:*(a::MaskedBatch{T, A}, b::AbstractMatrix) where {T<:AbstractMatrix, A}
    A[2] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(a.data)
    amat = reshape(permutedims(a.data, [2, 3]), s1*bs, s2)
    data = permutedims(reshape(amat * b, s1, bs, size(b, 2)), [2, 3])
    return MaskedBatch{T, A}(data, a.mask)
end

# contractions between axes with dynamic dimension

# TODO should things work when dynamic dimension = 0?
function Base.dot(a::MaskedBatch{T, A}, b::MaskedBatch{T, B}) where {T<:AbstractVector, A, B}
    data = sum(a.data .* b.data, 1)[1, :] # TODO replace with batchedmul when fast
    return MaskedBatch{T, ()}(data, a.mask[:, 1])
end

# the methods below use batched matrix-matrix and matrix-vector products;
# these are provided as BLAS extensions by cuBLAS and MKL but aren't
# wrapped in Base; here we implement sequential fallbacks but TODO these
# functions should actually be in another package and have those fast methods
function batchedmul(a::AbstractArray{T, 3}, b::AbstractArray{T, 2}) where {T}
    (bs = size(a, 3)) == size(b, 2) || error("batch size mismatch")
    res = similar(a, size(a, 1), bs)
    for i in 1:bs
        res[:, i] = a[:, :, i] * b[:, i]
    end
    return res
end
function batchedmul(a::AbstractArray{T, 3}, b::AbstractArray{T, 3}) where {T}
    (bs = size(a, 3)) == size(b, 3) || error("batch size mismatch")
    res = similar(a, size(a, 1), size(b, 2), bs)
    for i in 1:bs
        res[:, :, i] = a[:, :, i] * b[:, :, i]
    end
    return res
end

function Base.:*(a::MaskedBatch{T1, A}, b::MaskedBatch{T2, B}) where {T1<:AbstractMatrix, T2<:AbstractVector, A, B}
    A[2] == B[1] || error("cannot contract axes with static and dynamic dimension")
    data = batchedmul(a.data, b.data)
    mask = batchedmul(a.mask[:, 1:1, :], b.mask[1:1, :])
    return MaskedBatch{T2, (A[1],)}(data, mask)
end

function Base.:*(a::MaskedBatch{T, A}, b::MaskedBatch{T, B}) where {T<:AbstractMatrix, A, B}
    A[2] == B[1] || error("cannot contract axes with static and dynamic dimension")
    data = batchedmul(a.data, b.data)
    mask = batchedmul(a.mask[:, 1:1, :], b.mask[1:1, :, :])
    return MaskedBatch{T, (A[1], B[2])}(data, mask)
end

function NNlib.softmax(x::T, axis=1) where T<:MaskedBatch
    data = exp.(x.data .- maximum(x.data, axis)) .* x.mask
    data ./= sum(data, axis) .+ ϵ
    return T(data, x.mask)
end

function Base.transpose(x::MaskedBatch{T, A}) where {T, A}
    dims = collect(1:ndims(x.data))
    dims[1:2] = [2, 1]
    data = permutedims(x.data, dims)
    mask = permutedims(x.mask, dims)
    axes = tuple(A[2], A[1], A[3:end]...)
    return MaskedBatch{T, axes}(data, mask)
end

# generalize this
function Base.getindex(a::AbstractMatrix, ::Colon, b::MaskedBatch{T, B}) where {T<:AbstractVector, B}
    data = b.data .+ (1 .- b.mask)
    data = a[:, data]
    mask = reshape(b.mask, 1, size(b.mask)...)
    axes = tuple(false, B...)
    # TODO it's a nice coincidence that this typeof works here; it won't in other methods
    return MaskedBatch{typeof(a), axes}(data .* mask, mask)
end

function Base.mapslices(f, x::MaskedBatch{T, A}, dim::Int) where {T, A}
    A[dim] && error("reduction along dynamic dimension not implemented")
    return MaskedBatch{T, A}(mapslices(f, x.data, dim) .* x.mask, x.mask)
end

Base.mean(x::MaskedBatch, dim::Int) = mapslices(mean, x, dim)
Base.std(x::MaskedBatch, dim::Int) = mapslices(std, x, dim)
Base.sum(x::MaskedBatch, dim::Int) = mapslices(sum, x, dim)
Base.prod(x::MaskedBatch, dim::Int) = mapslices(prod, x, dim)

function Flux.normalise(x::MaskedBatch{<:AbstractVecOrMat})
  μ′ = mean(x, 1)
  σ′ = std(x, 1)
  return (x .- μ′) ./ (σ′ .+ ϵ)
end

end # module
