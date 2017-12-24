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
x1 = VectorBatch([data(3), data(4)], axisinfo)
x2 = SizedBatch(x1)

display(f(x1))
display(f(x2))
println()
display(g(x1))
display(g(x2))
println()
display(h(x1))
display(h(x2))

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
