using Minibatch
using NNlib
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

z1 = y1 * x1; z2 = y2 * x2
@test MaskedBatch(z1) ≈ z2
@test VectorBatch(z2) ≈ z1

for axis in (1, 2)
    @test MaskedBatch(softmax(z1, axis)) ≈ softmax(z2, axis)
end

function attention(q, k, v) # q::D×N, k::D×M, v::D×M
    alpha = k'*q # ::M×N
    # TODO tri mask
    return v*softmax(alpha, 1) # ::D×N
end

d_qk = 3
d_v = 4
enc = [rand(d_v, 6), rand(d_v, 5)]
dec = [rand(d_v, 3), rand(d_v, 7)]
enc1 = VectorBatch(enc, (false, true))
enc2 = MaskedBatch(enc, (false, true))
dec1 = VectorBatch(dec, (false, true))
dec2 = MaskedBatch(dec, (false, true))

Wq = rand(d_qk, d_v)
Wk = rand(d_qk, d_v)
Wv = rand(d_v, d_v)

out1 = attention(Wq*dec1, Wk*enc1, Wv*enc1)
out2 = attention(Wq*dec2, Wk*enc2, Wv*enc2)
@test MaskedBatch(out1) ≈ out2
