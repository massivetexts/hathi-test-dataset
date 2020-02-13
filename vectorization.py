#!/usr/bin/env python
# coding: utf-8

# In[140]:

from htrc_features import Volume,transformations
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np
import time
import uuid
import sys
import SRP
from wem_hook import WEM_transform
from python.import_utils import already_imported_list
from python.hathi_resolver import my_resolver as customizable_resolver

if len(sys.argv) < 4:
    print("Usage: `python vectorization.py 3 5 'test_dataset.csv.gz'`, where this is the third of five threads. NOT zero-indexed.")

thread_no = int(sys.argv[1]) - 1 # Zero index.
threads   = int(sys.argv[2])
test_file_loc = sys.argv[3]

already_imported = already_imported_list()

print("There are {} files already imported".format(len(already_imported)))
filenames = pd.read_csv("test_dataset.csv.gz", low_memory = False)

def yielder(ids, chunk_size = 5000):
    """
    ids: a list of htids to iterate over.
    chunks_size: the chunk size.
    
    returns: an iterable over tuples of id, chunk number, and the grouped token counts.
    """
    locs = [id for id in ids if not id in already_imported]
    
    # Only do the ones allocated for this thread.
    locs = [loc for (i, loc) in enumerate(locs) if i % threads == thread_no]
    
    for i, id in enumerate(locs):
        vol = Volume(id, id_resolver = customizable_resolver)
        try:
            chunks = vol.tokenlist(chunk = True, chunk_size = 10000, overflow = 'ends', pos=False, page_ref = True)
            chunks.reset_index(level = 3, inplace = True)
            if chunks.empty:
                continue
            for (chunk, start, end) in set(chunks.index):
                yield (id, chunk, start, end, chunks.loc[(chunk, start, end)].reset_index(drop = True))
        except:
            print("Error chunking {}... skipping\n".format(id))
            continue

thread_name = "{}-of-{}_".format(thread_no + 1, threads)


already_seen_file = open("data_outputs/already_completed_files{}.csv".format(thread_name), "a")

hasher = SRP.SRP(640)
out_SRP = SRP.Vector_file("data_outputs/" + thread_name + "SRP_chunks.bin", dims=640, mode="w")
out_glove = SRP.Vector_file("data_outputs/" + thread_name + "Glove_chunks.bin", dims = 300, mode="w")

def SRP_transform(f):
    return hasher.stable_transform(words = f['token'], counts = f['count'], log = True, standardize = True)


# In[146]:


# Bookworm prep
import gzip
unigrams = gzip.open("data_outputs/" + thread_name + "bookworm.unigrams.gz", "wt")


# In[147]:

books = 0
last = None
start_time = time.time()

try:
    for i, (id, chunk, start, end, group) in enumerate(yielder(filenames.htid)):
        # Count books too.
        if last != id:
            books += 1
            if last is not None:
                already_seen_file.write("{}\n".format(last))
                if (books % 25 == 0):
                    rate = books/(time.time()-start_time)
                    print("{} books done on thread {} of {}, {:.02f} chunks per book, {:.02f} books per second".format(books, thread_no, threads, i/books, rate))
        last = id
                                    
        id = "{}-{:04d}-{}-{}".format(id, chunk, start, end)

        # Do SRP
        SRP_rep = SRP_transform(group)
        out_SRP.add_row(id, SRP_rep)

        # Convert to 300dim Glove vec
        WEM_rep = WEM_transform(group)
        out_glove.add_row(id, WEM_rep.astype('<f4'))

        if False:
            wordcount_rep = (group.reset_index(0, drop = True)
                                  .to_csv(line_terminator="\n", header = False, index=False)
                                  .replace("\n", "\f")
                            )

            unigrams.write("{}\t{}\n".format(id, wordcount_rep))

    already_seen_file.write("{}\n".format(last))

    out_SRP.close()
    unigrams.close()

except KeyboardInterrupt:
    out_glove.close()
    out_SRP.close()
