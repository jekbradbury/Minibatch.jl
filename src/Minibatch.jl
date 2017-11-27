module Minibatch

using Cassette

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

###############
# Batch types #
###############

# eventually we want a flexible array wrapper type that lets us designate one
# axis as the batch axis and a subset of other axes as spatial/temporal (that
# is, potentially variable).

abstract type AbstractBatch{T} end

struct VectorBatch{T} <: AbstractBatch{T}
    data::Vector{T}
end
VectorBatch(data) = VectorBatch{typeof(first(data))}(data)

struct PadBatch{T} <: AbstractBatch{T}
    data # TODO check that perf is ok
    mask
end
function PadBatch(data::Array{E, N}, mask::Array{E, N}) where {E, N} = PadBatch{Array{E, N - 1}}(data, mask) # TODO generalize this

# also (at least) CatBatch using CatArrays.jl

length(b::VectorBatch) = length(b.data)
length(b::PadBatch) = last(size(b.data))

#####################
# Cassette overdubs #
#####################

Cassette.@context Ctx
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

softmax(x::PadBatch{T}, axis) where T = PadBatch{T}(softmax(x.data, axis), x.mask) # TODO NO BAD WRONG

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
