module Minibatch

export VectorBatch, SizedBatch, MaskedBatch, softmax

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
Base.:(==)(a::VectorBatch{T1, A}, b::VectorBatch{T2, A}) where {T1, T2, A} = a.data == b.data

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

# TODO make batch size a part of the type when we have static dims
Base.length(b::SizedBatch) = last(size(b.sizes))
Base.:(==)(a::SizedBatch{T1, A}, b::SizedBatch{T2, A}) where {T1, T2, A} = a.data == b.data && a.sizes == b.sizes

SizedBatch(b::VectorBatch{T, A}) where {T, A} = SizedBatch(b.data, A)
function VectorBatch(b::SizedBatch{T, A}) where {T, A}
    bs = last(size(b.sizes))
    xs = [b.data[(1:n for n in b.sizes[:, i])..., i] for i in 1:bs]
    return VectorBatch{T, A}(xs)
end

struct MaskedBatch{T, A} <: AbstractBatch{T, A}
    data
    mask
    sizes::Matrix{Int}
end
function MaskedBatch(b::SizedBatch{T, A}) where {T, A}
    dims = tuple((a ? size(b.data, d) : 1 for (d, a) in enumerate(A))...,
                 last(size(b.sizes)))
    mask = fill!(similar(b.data, dims), 0)
    for i in 1:last(size(b.sizes))
        mask[(1:(A[d] ? n : 1) for (d, n) in enumerate(b.sizes[:, i]))..., i] .= 1
    end
    return MaskedBatch{T, A}(b.data, mask, b.sizes)
end
SizedBatch(b::MaskedBatch{T, A}) where {T, A} = SizedBatch{T, A}(b.data, b.sizes)
MaskedBatch(data::Vector, axes::AxisInfo) = MaskedBatch(SizedBatch(data, axes))
MaskedBatch(b::VectorBatch) = MaskedBatch(SizedBatch(b))
VectorBatch(b::MaskedBatch) = VectorBatch(SizedBatch(b))

Base.length(b::MaskedBatch) = last(size(b.sizes))
Base.:(==)(a::MaskedBatch{T1, A}, b::MaskedBatch{T2, A}) where {T1, T2, A} = a.data == b.data && a.sizes == b.sizes

const SMBatch = Union{SizedBatch, MaskedBatch}

# TODO also CatBatch using CatArrays.jl?

####################################
# Methods/overdubs for batch types #
####################################

function _checkbatchsizes(xs::Vararg{AbstractBatch})
    bs = length(first(xs))
    all(length(x) == bs for x in xs) || error("batch size mismatch")
    return bs
end
function _vbcall(f, T, xs...)
    batchsize = _checkbatchsizes((x for x in xs if x isa T)...)
    return T([f((x isa T ? x.data[i] : x for x in xs)...) for i in 1:batchsize])
end
Base.:+(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(+, T, xs...)
Base.:-(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(-, T, xs...)
Base.:*(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(*, T, xs...)
Base.:/(xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(/, T, xs...)
Base.getindex(x::T, inds...) where T<:VectorBatch = _vbcall(getindex, T, x, inds...)
Base.broadcast(f, xs::Vararg{Union{T, AbstractArray}}) where T<:VectorBatch = _vbcall(broadcast, T, f, xs...)

_rezero!(b::MaskedBatch) = b.data .*= b.mask
function _rezero!(b::SizedBatch)
    b = MaskedBatch(b)
    _rezero!(b)
    return SizedBatch(b)
end
# TODO this should really be promote_containersize
function _checksizes(xs::Vararg{T}) where T<:SMBatch
    fs = first(xs).sizes
    all(x.sizes == fs for x in xs) || error("SizedBatch or MaskedBatch size mismatch")
    return fs
end
function _zeropreserving(f)
    # ideally would do this at compile time
    return method_exists(f, (Float64,)) ? f(0.0) == 0.0 : false
end
function Base.broadcast(f, xs::Vararg{Union{T, AbstractArray}}) where T<:SizedBatch
    sizes = _checksizes((x for x in xs if x isa T)...)
    ret = T(broadcast(f, (x isa T ? x.data : x for x in xs)...), sizes)
    _zeropreserving(f) || _rezero!(ret)
    return ret
end
function Base.broadcast(f, xs::Vararg{Union{T, AbstractArray}}) where T<:MaskedBatch
    sizes = _checksizes((x for x in xs if x isa T)...)
    ret = T(broadcast(f, (x isa T ? x.data : x for x in xs)...), first(xs).mask, sizes)
    _rezero!(ret)
    return ret
end
Base.:+(xs::Vararg{Union{T, AbstractArray}}) where T<:SMBatch = broadcast(+, xs...)
Base.:-(xs::Vararg{Union{T, AbstractArray}}) where T<:SMBatch = broadcast(-, xs...)

function Base.dot(a::SizedBatch{T, A}, b::SizedBatch{T, B}) where {T<:AbstractVector, A, B}
    batchsize = last(size(_checksizes(a, b)))
    data = sum(a.data .* b.data, 1)[1, :] # TODO dotBatched?
    return SizedBatch{T, ()}(data, Matrix{Int}(0, batchsize))
end
function Base.dot(a::AbstractVector, b::SizedBatch{T, (false,)}) where {T}
    return SizedBatch{T, ()}(b.data'*a, Matrix{Int}(0, length(b)))
end
function Base.dot(a::SizedBatch{T, (false,)}, b::AbstractVector) where {T}
    return SizedBatch{T, ()}(a.data'*b, Matrix{Int}(0, length(a)))
end

function Base.:*(a::SizedBatch{T, A}, b::AbstractVector) where {T<:AbstractMatrix, A}
    A[2] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(a.data)
    data = reshape(reshape(a.data, s1, s2*bs) * b, s1, bs)
    return SizedBatch{T, A}(data, b.sizes[1:1, :])
end
function Base.:*(a::AbstractMatrix, b::SizedBatch{T, (false,)}) where {T}
    sizes = fill(similar(b.sizes), size(a, 1))
    return SizedBatch{T, (false,)}(a * b.data, sizes)
end

function Base.:*(a::AbstractMatrix, b::SizedBatch{T, B}) where {T<:AbstractMatrix, B}
    B[1] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(b.data)
    data = reshape(a * reshape(b.data, s1, s2*bs), size(a, 1), s2, bs)
    sizes = deepcopy(b.sizes)
    sizes[1, :] .= size(a, 1)
    return SizedBatch{T, B}(data, sizes)
end
function Base.:*(a::SizedBatch{T, A}, b::AbstractMatrix) where {T<:AbstractMatrix, A}
    A[2] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(a.data)
    data = reshape(reshape(a.data, s1, s2*bs) * b, s1, size(b, 2), bs)
    sizes = deepcopy(b.sizes)
    sizes[2, :] .= size(b, 2)
    return SizedBatch{T, A}(data, sizes)
end

function Base.:*(a::AbstractMatrix, b::MaskedBatch{T, B}) where {T<:AbstractMatrix, B}
    B[1] == false || error("cannot contract axes with static and dynamic dimension")
    s1, s2, bs = size(b.data)
    data = reshape(a * reshape(b.data, s1, s2*bs), size(a, 1), s2, bs)
    sizes = deepcopy(b.sizes)
    sizes[1, :] .= size(a, 1)
    return MaskedBatch{T, B}(data, b.mask, sizes)
end

# these require gemmBatched/gemm_batched:
# Base.:*(a::SizedBatch{T, A}, b::SizedBatch{T, B}) where {T<:AbstractMatrix, A, B}
# Base.:*(a::SizedBatch{T1, A}, b::SizedBatch{T2, B}) where {T1<:AbstractMatrix, T2<:AbstractVector, A, B}
# Base.:*(a::SizedBatch{T1, A}, b::SizedBatch{T2, B}) where {T1<:AbstractVector, T2<:AbstractMatrix, A, B}

end # module
