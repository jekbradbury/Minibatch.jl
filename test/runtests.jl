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

axisinfo = (false, true)
data(n) = rand(5, n)
xs = [data(3), data(4)]
x1 = VectorBatch(xs, axisinfo)
x2 = SizedBatch(xs, axisinfo)
x3 = MaskedBatch(xs, axisinfo)

for other in (x1, x2, x3)
    @test x1 == VectorBatch(other)
    @test x2 == SizedBatch(other)
    @test x3 == MaskedBatch(other)
end

for fn in (f, g, h)
    @test SizedBatch(fn(x1)) == fn(x2)
    @test MaskedBatch(fn(x1)) == fn(x3)
end

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
