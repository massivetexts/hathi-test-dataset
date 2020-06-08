from htrc_features import Volume,transformations
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np
import time
import uuid
import os
import sys
import SRP
import argparse
from wem_hook import WEM_transform
from compare_tools.import_utils import already_imported_list
from python.hathi_resolver import my_resolver as customizable_resolver

def main():
    parser = argparse.ArgumentParser(description="Convert Extracted Features files to vectors, and save in SRP's Vector_file format.")

    parser.add_argument('threadno', type=int, help='Non-zero indexed number of thread.')
    parser.add_argument('totalthreads', type=int, help='Number of threads running in total.')
    parser.add_argument('idlist', type=str, help='CSV file of HTIDs to process. Needs a header, with column name \'htid\'.')
    parser.add_argument('--outdir', '-o', type=str, default='data_outputs/', help='Directory to save results.')
    parser.add_argument('--chunksize', '-c', type=int, default=10000, help='Size of chunks to roll pages into.')
    parser.add_argument('--no-srp', action='store_true', help='Turn off SRP saving')
    parser.add_argument('--no-glove', action='store_true', help='Turn off Glove saving')
    args = parser.parse_args()
    
    thread_no = args.threadno - 1 # Zero index.
    
    assert not (args.no_srp & args.no_glove)

    already_imported = already_imported_list(args.outdir)
    print("There are {} files already imported".format(len(already_imported)))
    filenames = pd.read_csv(args.idlist, low_memory = False)
    
    thread_name = "{}-of-{}_".format(thread_no + 1, args.totalthreads)

    already_seen_file = open(os.path.join(args.outdir, "already_completed_files{}.csv".format(thread_name)), "a")

    if not args.no_srp:
        hasher = SRP.SRP(640)
        out_SRP = SRP.Vector_file(os.path.join(args.outdir, thread_name + "SRP_chunks.bin"), dims=640, mode="w")
        
        def SRP_transform(f):
            return hasher.stable_transform(words = f['lowercase'], counts = f['count'], log = True, standardize = True)
    
    if not args.no_glove:
        out_glove = SRP.Vector_file(os.path.join(args.outdir, thread_name + "Glove_chunks.bin"), dims = 300, mode="w")

    books = 0
    last = None
    start_time = time.time()

    try:
        gen = yielder(filenames.htid, thread_no, args.totalthreads, 
                      chunk_size=args.chunksize, already_imported_list=already_imported)
        for i, (id, chunk, start, end, group) in enumerate(gen):
            # Count books too.
            if last != id:
                books += 1
                if last is not None:
                    already_seen_file.write("{}\n".format(last))
                    if (books % 25 == 0):
                        rate = books/(time.time()-start_time)
                        print("{} books done on thread {} of {}, {:.02f} chunks per book, {:.02f} books per second".format(books, thread_no, args.totalthreads, i/books, rate))
            last = id

            id = "{}-{:04d}-{}-{}".format(id, chunk, start, end)

            if not args.no_srp:
                SRP_rep = SRP_transform(group)
                out_SRP.add_row(id, SRP_rep)

            if not args.no_glove:
                WEM_rep = WEM_transform(group)
                out_glove.add_row(id, WEM_rep.astype('<f4'))

        already_seen_file.write("{}\n".format(last))

        if not args.no_srp:
            out_SRP.close()
        if not args.no_glove:
            out_glove.close()

    except KeyboardInterrupt:
        if not args.no_srp:
            out_SRP.close()
        if not args.no_glove:
            out_glove.close()


def yielder(ids, thread_no, totalthreads, chunk_size = 10000, already_imported_list=[]):
    """
    ids: a list of htids to iterate over.
    chunks_size: the chunk size.
    
    returns: an iterable over tuples of id, chunk number, and the grouped token counts.
    """
    locs = [id for id in ids if not id in already_imported_list]
    
    # Only do the ones allocated for this thread.
    locs = [loc for (i, loc) in enumerate(locs) if i % totalthreads == thread_no]
    
    for i, id in enumerate(locs):
        vol = Volume(id, id_resolver=customizable_resolver)
        try:
            chunks = vol.tokenlist(chunk = True, chunk_size = chunk_size, overflow = 'ends', case=False, pos=False, page_ref = True)
            chunks.reset_index(level = 3, inplace = True)
            if chunks.empty:
                continue
            for (chunk, start, end) in set(chunks.index):
                yield (id, chunk, start, end, chunks.loc[(chunk, start, end)].reset_index(drop = True))
        except:
            print("Error chunking {}... skipping\n".format(id))
            continue


if __name__ == '__main__':
    main()