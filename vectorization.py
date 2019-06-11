#!/usr/bin/env python
# coding: utf-8

# In[140]:

from htrc_features import FeatureReader,transformations
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np
import time
import uuid
import sys
import SRP
from wem_hook import WEM_transform
from python.import_utils import already_imported_list

if len(sys.argv) < 5:
    print("Usage: `python vectorization.py 3 5 '../../hathi-ef' 'test_dataset.csv.gz'`, where this is the third of five threads and the ef root directory is '../../hathi-ef'. NOT zero-indexed.")

thread_no = int(sys.argv[1]) - 1 # Zero index.
threads   = int(sys.argv[2])
hathi_features_loc = sys.argv[3]
test_file_loc = sys.argv[4]


already_imported = already_imported_list()

print("There are {} files already imported".format(len(already_imported)))

def yielder(ids, chunk_size = 5000, hathi_loc = hathi_features_loc):
    """
    ids: a list of htids to iterate over.
    chunks_size: the chunk size.
    
    returns: an iterable over tuples of id, chunk number, and the grouped token counts.
    """
    locs = [hathi_loc + id_to_rsync(id) for id in ids if not id in already_completed]
    
    # Only do the ones allocated for this thread.
    locs = [loc for (i, loc) in enumerate(locs) if i % threads == thread_no]
    reader = FeatureReader(locs)

    
    
    for i, vol in enumerate(reader.volumes()):
        id = vol.id
        try:
            chunks = vol.chunked_tokenlist(chunk_size, pos=False)
            if chunks.empty:
                continue
            for ix in chunks.index.get_level_values('chunk').unique():
                 yield (id, ix, chunks.xs(ix, level='chunk').reset_index())
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
start = time.time()

try:
    for i, (id, ix, group) in enumerate(yielder(filenames.htid)):
        # Count books too.
        if last != id:
            already_seen_file.write("{}\n".format(last)
            books += 1
            if (books % 20 == 0):
                rate = books/(time.time()-start)
                print("{} books done, {:.02f} chunks per book, {:.02f} books per second".format(books, i/books, rate))
        last = id
                                    
        id = "{}-{:04d}".format(id, ix)

        # Do SRP
        SRP_rep = SRP_transform(group)
        out_SRP.add_row(id, SRP_rep)

        # Convert to 300dim Glove vec
        WEM_rep = WEM_transform(group)
        out_glove.add_row(id, WEM_rep.astype('<f4'))

        wordcount_rep = (group.reset_index(0, drop = True)
                              .to_csv(line_terminator="\n", header = False, index=False)
                              .replace("\n", "\f")
                        )

        unigrams.write("{}\t{}\n".format(id, wordcount_rep))

    already_seen_file.write("{}\n".format(last)

    out_SRP.close()
    unigrams.close()

except KeyboardInterrupt:
    out_glove.close()
    out_SRP.close()
