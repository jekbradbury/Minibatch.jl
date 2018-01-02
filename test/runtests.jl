using Minibatch
using Base.Test

#################
# Test networks #
#################

W = rand(5, 5)
b = rand(5)
f(x) = tanh.(x)
g(x) = tanh.(x .+ b)
h(x) = tanh.(W*x .+ b)

xs = [rand(5, 3), rand(5, 4)]
x1 = VectorBatch(xs, (false, true))
x2 = MaskedBatch(xs, (false, true))

@test x1 == VectorBatch(x2)
@test x2 == MaskedBatch(x1)

for fn in (f, g, h)
    @test MaskedBatch(fn(x1)) ≈ fn(x2)
end

ys = [rand(4, 5), rand(2, 5)]
y1 = VectorBatch(ys, (true, false))
y2 = MaskedBatch(ys, (true, false))

@test MaskedBatch(y1 * x1) ≈ y2 * x2

# function attention(q, k, v) # q::N×D, k::M×D, v::M×D
#     alpha = q*k' # ::N×M
#     # TODO tri mask
#     softmax(alpha, 2)*v # ::N×D
# end

function attention(q, k, v) # q::D×N, k::D×M, v::D×M
    alpha = k'*q # ::M×N
    # TODO tri mask
    return v*softmax(alpha, 1) # ::D×N
end
