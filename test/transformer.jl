using Minibatch
using Revtok
using DataStructures

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

type Vocab
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

type Embedding
    W
    Embedding(vocabsize, embedsize) = new(randn(embedsize, vocabsize))
end
(embedding::Embedding)(x) = embedding.W[:, x]

#corpus = zip(english, spanish)
en, vocab_en = process(english)
es, vocab_es = process(spanish)

model = Embedding(length(vocab_en), 2)

println("----")
for (src, trg) in zip(en, es)
    display(src)
    src = model(src)
    display(src)
end
