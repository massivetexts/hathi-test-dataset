wem_model = None
wem_vocab = None
from htrc_features import transformations

def WEM_transform(f):
    global wem_model
    global wem_vocab
    from gensim.models import KeyedVectors
    
    if not wem_model:
        wem_model = KeyedVectors.load_word2vec_format('/data/glove/glove.840B-300d.word2vec.bin', binary=True)
        
    if not wem_vocab:
        from spacy.lang.en.stop_words import STOP_WORDS
        wem_vocab = set(model.vocab.keys())
        wem_vocab = wem_vocab.difference(STOP_WORDS)
        # Cross-ref with stoplist and drop stopped words
        
    vec = transformations.chunk_to_wem(f, model, vocab=vocab, stop=False, log=True, min_ncount=10)
    return vec