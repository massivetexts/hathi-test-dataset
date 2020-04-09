wem_model = None
wem_vocab = None

from local import wem_location

from htrc_features import transformations

def WEM_transform(f):
    global wem_model
    global wem_vocab
    from gensim.models import KeyedVectors
    
    if not wem_model:
        # Updated to use gensim's saved format - faster and memmappable.
        # if you have files in another format that Gensim can open, you can
        # open and 'save()' as the native format.
        wem_model = KeyedVectors.load(wem_location, mmap='r')
        
    if not wem_vocab:
        from spacy.lang.en.stop_words import STOP_WORDS
        wem_vocab = set(wem_model.vocab.keys())
        wem_vocab = wem_vocab.difference(STOP_WORDS)
        # Cross-ref with stoplist and drop stopped words
        
    vec = transformations.chunk_to_wem(f, wem_model, vocab=wem_vocab, stop=False, log=True, min_ncount=10)
    return vec

