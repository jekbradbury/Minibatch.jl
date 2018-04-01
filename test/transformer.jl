using Minibatch
using Revtok
using DataStructures
using GPUArrays
using NNlib
using Flux

english = """This is an account of how magical thinking made us modern.
When people talk about magical thinking, it is usually as a cognitive feature of children, uneducated people, the mushy-minded, or the mentally ill.
If we notice magical thinking in ourselves, it is with a pang of shame: literate adults are supposed to be more sophisticated than that.
At the same time, magical thinking is obviously rampant in the world.
It’s hard not to be fascinated, even if it’s a horrified fascination.
I think that the reason that it has been so difficult to precisely define “magical thinking” is that what we call “magical thinking” is a collection of stigmatized examples of a more general, and generally useful, cognitive capacity.
This is the ability to think in “as if” mode: “as if” inanimate objects had minds, “as if” thoughts could affect reality, “as if” symbols had power over their referents."""

spanish = """Este es un relato de cómo el pensamiento mágico nos hizo modernos.
Cuando la gente habla de pensamiento mágico, generalmente es como un rasgo cognitivo de los niños, de las personas sin educación, de los que tienen la mente blanda o de los enfermos mentales.
Si nos damos cuenta del pensamiento mágico en nosotros mismos, es con una espiral de vergüenza: se supone que los adultos alfabetizados son más sofisticados que eso.
Al mismo tiempo, el pensamiento mágico es obviamente desenfrenado en el mundo.
Es difícil no estar fascinado, aunque sea una fascinación horrorizada.
Creo que la razón por la que ha sido tan difícil definir con precisión el “pensamiento mágico” es que lo que llamamos "pensamiento mágico" es una colección de ejemplos estigmatizados de una capacidad cognitiva más general, y en general útil.
Esta es la capacidad de pensar en modo “como si”: “como si” los objetos inanimados tuvieran mente, “como si” los pensamientos pudieran afectar a la realidad, “como si” los símbolos tuvieran poder sobre sus referentes."""

struct Vocab
    itos::Vector
    stoi::Associative
end
Base.length(vocab::Vocab) = length(vocab.itos)

function process(corpus, vocabsize=200, batchsize=4)
    corpus = tokenize.(split(corpus, '\n'))
    segments = buildvocab(counter(Iterators.flatten(corpus)), vocabsize)
    corpus = segment.(corpus, segments)
    itos = keys(segments)
    stoi = Dict((reverse(p) for p in enumerate(itos)))
    vocab = Vocab(collect(itos), stoi)
    corpus = [[vocab.stoi[x] for x in ex] for ex in corpus]
    corpus = Iterators.partition(corpus, batchsize)
    corpus = map(b -> MaskedBatch(b, (true,)), corpus)
    return corpus, vocab
end

# first step to patching issues that come up when we try to use Flux.Tracker
function Base.getindex(a::Flux.TrackedArray{T, 2} where T, ::Colon, b::MaskedBatch{<:AbstractVector})
    return invoke(getindex, Tuple{AbstractMatrix, Colon, MaskedBatch{<:AbstractVector}}, a, :, b)
end

function posenc(timestep::Integer, channel::Integer, nchannels::Integer)
    if iseven(channel)
        return sin(timestep/(10000^(channel/nchannels)))
    else
        return cos(timestep/(10000^((channel-1)/nchannels)))
    end
end

function posenc(idx::CartesianIndex, nchannels::Integer)
    return posenc(idx[2], idx[1], nchannels)
end

function posenc!(A::GPUArray, state, nchannels::Integer)
    idx = @cartesianidx A state
    @inbounds A[idx] += posenc(idx, nchannels)
    return A
end

function posenc!(A::GPUArray)
    nchannels = size(A, 1)
    gpucall(posenc, A, (nchannels,))
    return A
end

function posenc!(A::AbstractArray)
    nchannels = size(A, 1)
    for idx in CartesianRange(size(A))
        # @inbounds
        A[idx] += posenc(idx, nchannels)
    end
    return A
end

function posenc!(B::MaskedBatch)
    posenc!(B.data)
    B.data = B.data .* B.mask # TODO in-place
    return B
end

function posenc!(A::Flux.TrackedArray)
    return posenc!(Flux.Tracker.data(A))
end

struct Embedding
    W
end
Embedding(vocabsize, embedsize) = Embedding(param(randn(embedsize, vocabsize)))
(embedding::Embedding)(x) = embedding.W[:, x]
Flux.treelike(Embedding)

# struct LayerNorm
#     γ
#     β
# end
# LayerNorm(nchannels) = LayerNorm(param(ones(nchannels)), param(zeros(nchannels)))
# (l::LayerNorm)(x) = l.γ .* (x .- mean(x, 1)) ./ (std(x, 1) .+ ϵ) .+ l.β

FeedForward(d_model, d_hidden) = Chain(Dense(d_model, d_hidden, relu), Dense(d_hidden, d_model))

struct ResidualBlock
    l
    norm
    # TODO dropout
end
ResidualBlock(l, d_model::Integer) = ResidualBlock(l, LayerNorm(d_model))
(l::ResidualBlock)(x...) = x[1] .+ l.norm(l.l(x...))
Flux.treelike(ResidualBlock)

struct Attention
    scale
    # TODO dropout
    # TODO causal
end
Attention(d_key::Integer, causal=false) = Attention(sqrt(d_key))
function (l::Attention)(q, k, v)
    alpha = k'*q
    # TODO causal mask
    return v*softmax(alpha, 1)
end
Flux.mapchildren(f, l::Attention) = Attention(l.scale)

# TODO multihead

struct EncoderLayer
    selfattn
    feedforward
end
EncoderLayer(d_model::Integer, d_hidden::Integer) = EncoderLayer(
    ResidualBlock(Attention(d_model), d_model),
    ResidualBlock(FeedForward(d_model, d_hidden), d_model))
(l::EncoderLayer)(x) = l.feedforward(l.selfattn(x, x, x))
Flux.treelike(EncoderLayer)

struct Encoder
    embed
    layers
end
Encoder(d_model, d_hidden, n_layers, vocabsize) = Encoder(
    Embedding(vocabsize, d_model),
    Chain((EncoderLayer(d_model, d_hidden) for _ in 1:n_layers)...))
function (l::Encoder)(x)
    x = posenc!(l.embed(x))
    encoding = [x = l(x) for l in l.layers.layers]
    return encoding
end
Flux.treelike(Encoder)

#corpus = zip(english, spanish)
en, vocab_en = process(english)
es, vocab_es = process(spanish)

# d = 4
# embed = Embedding(length(vocab_en), d)
# layernorm = LayerNorm(d)
# linear = Dense(d, d)
# feedforward = FeedForward(d, 2d)
# residual = ResidualBlock(FeedForward(d, 2d), d)
#
# x = en[1]
# x = embed(x)
# x = posenc!(x)
# x = layernorm(x)
# x = linear(x)
# x = feedforward(x)
# x = residual(x)

encoder = Encoder(4, 8, 3, length(vocab_en))
encoder = Flux.mapleaves(Flux.Tracker.data, encoder)
encoder(en[1])
