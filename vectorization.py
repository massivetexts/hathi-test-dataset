#!/usr/bin/env python
# coding: utf-8

# In[140]:


from htrc_features import FeatureReader
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np
import time
import uuid
import sys
import SRP

if len(sys.argv) < 5:
    print("Usage: `python vectorization.py 3 5 '../../hathi-ef' 'test_dataset.csv.gz'`, where this is the third of five threads and the ef root directory is '../../hathi-ef'. NOT zero-indexed.")

thread_no = int(sys.argv[1]) - 1 # Zero index.
threads   = int(sys.argv[2])
hathi_features_loc = sys.argv[3]
test_file_loc = sys.argv[4]

filenames = pd.read_csv("test_dataset.csv.gz", low_memory = False)

def yielder(ids, chunk_size = 5000, hathi_loc = "../../hathi-ef/"):
    """
    ids: a list of htids to iterate over.
    chunks_size: the chunk size.
    
    returns: an iterable over tuples of id, chunk number, and the grouped token counts.
    """
    locs = [hathi_loc + id_to_rsync(id) for id in ids]
    # Only do the ones allocated for this thread.
    locs = [loc for (i, loc) in enumerate(locs) if i % threads == thread_no]
    reader = FeatureReader(locs)
    for i, vol in enumerate(reader.volumes()):
        id = vol.id
        try:
            chunks = vol.chunked_tokenlist(chunk_size).groupby(["chunk","token"])['count'].sum().reset_index(1)
        except:
            print("Error chunking {}... skipping\n".format(id))
            continue
        grouped = chunks.groupby("chunk")
        for ix, group in grouped:
             yield (id, ix, group.reset_index(0, drop=True))

thread_name = "{}-of-{}_".format(thread_no + 1, threads)


hasher = SRP.SRP(640)
out_SRP = SRP.Vector_file(thread_name + "SRP_chunks.bin", dims=640, mode="w")
out_glove = SRP.Vector_file(thread_name + "Glove_chunks.bin", dims = 300, mode="w")

def SRP_transform(f):
    return hasher.stable_transform(words = f['token'], counts = f['count'], log = True, standardize = True)


# In[146]:


# Bookworm prep
import gzip
unigrams = gzip.open(thread_name + "bookworm.unigrams.gz", "wt")


# In[147]:

books = 0
last = None
start = time.time()

for i, (id, ix, group) in enumerate(yielder(filenames.htid)):
    # Count books too.
    if last != id:
        books += 1
        if books % 50 == 0:
            rate = books/(time.time()-start)
            print("{} books done, {:.02f} chunks per book, {:.02f} books per second".format(books, i/books, rate))
    last = id

    id = "{}-{:04d}".format(id, ix)
    
    # Do SRP
    SRP_rep = SRP_transform(group)
    out_SRP.add_row(id, SRP_rep)
    
    wordcount_rep = group       .reset_index(0, drop = True)       .to_csv(line_terminator="\n", header = False, index=False)       .replace("\n", "\f")
    
    unigrams.write("{}\t{}\n".format(id, wordcount_rep))


# In[168]:


out_SRP.close()
unigrams.close()
