from htrc_features import Volume, transformations
import pandas as pd
import time
import os
import SRP
import argparse
from compare_tools.import_utils import already_imported_list
from python.hathi_resolver import my_resolver as customizable_resolver
from compare_tools.configuration import wem_loader

def main():
    parser = argparse.ArgumentParser(description="Convert Extracted Features files to vectors, and save in SRP's Vector_file format.")

    parser.add_argument('threadno', type=int, help='Non-zero indexed number of thread.')
    parser.add_argument('totalthreads', type=int, help='Number of threads running in total.')
    parser.add_argument('idlist', type=str, help='CSV file of HTIDs to process. Needs a header, with column name \'htid\'.')
    parser.add_argument('--outdir', '-o', type=str, default='data_outputs/', help='Directory to save results.')
    parser.add_argument('--chunksize', '-c', type=int, default=10000, help='Size of chunks to roll pages into.')
    parser.add_argument('--no-srp', action='store_true', help='Turn off SRP saving')
    parser.add_argument('--in-memory', action='store_true', help='Turn off on-disk build if you have enough memory.')
    parser.add_argument('--no-glove', action='store_true', help='Turn off Glove saving')
    parser.add_argument('--glove-dims', '-g', type=int, default=300, help='Number of GloVe dimensions. Can be 50, 100, 200, or 300.')
    parser.add_argument('--srp-dims', '-s', type=int, default=640, help='Number of SRP dimensions.')
    args = parser.parse_args()
    
    thread_no = args.threadno - 1 # Zero index.
    
    assert not (args.no_srp & args.no_glove)

    already_imported = already_imported_list(args.outdir)
    print("There are {} files already imported".format(len(already_imported)))
    filenames = pd.read_csv(args.idlist, low_memory = False)
    
    thread_name = "{}-of-{}_".format(thread_no + 1, args.totalthreads)

    already_seen_file = open(os.path.join(args.outdir, "already_completed_files{}.csv".format(thread_name)), "a")

    if not args.no_srp:
        hasher = SRP.SRP(args.srp_dims)
        out_SRP = SRP.Vector_file(os.path.join(args.outdir, thread_name + "SRP_chunks.bin"), dims=args.srp_dims, mode="w")
        
        def SRP_transform(f):
            return hasher.stable_transform(words = f['lowercase'], counts = f['count'], log = True, standardize = True)
    
    if not args.no_glove:
        wem_model = wem_loader('glove-wiki-gigaword-{}'.format(args.glove_dims))
        
        # Cross-ref with stoplist and drop stopped words
        from spacy.lang.en.stop_words import STOP_WORDS
        wem_vocab = set(wem_model.vocab.keys())
        wem_vocab = wem_vocab.difference(STOP_WORDS)
        
        out_glove = SRP.Vector_file(os.path.join(args.outdir, thread_name + "Glove_chunks.bin"), dims = args.glove_dims, mode="w")
        
        def WEM_transform(f):
            return transformations.chunk_to_wem(f, wem_model, vocab=wem_vocab, stop=False, log=True, min_ncount=10)

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
                        print("{} books done on thread {} of {}, {:.02f} chunks per book, {:.02f} books per second".format(books, thread_no + 1, args.totalthreads, i/books, rate))
            last = id

            id = "{}-{:04d}-{}-{}".format(id, chunk, start, end)

            if not args.no_srp:
                SRP_rep = SRP_transform(group)
                out_SRP.add_row(id, SRP_rep)

            if not args.no_glove:
                WEM_rep = WEM_transform(group)
                if WEM_rep.shape[0] != args.glove_dims:
                    print(WEM_rep.shape, args.glove_dims)
                try:
                    out_glove.add_row(id, WEM_rep.astype('<f4'))
                except:
                    print(id, WEM_rep.shape, args.glove_dims, wem_model.vector_size)
                    raise
                    
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
    
    locs = [id for (i, id) in enumerate(ids) if i % totalthreads == thread_no]
    locs = [loc for loc in locs if loc not in already_imported_list]
    
    for i, id in enumerate(locs):
        vol = Volume(id, id_resolver=customizable_resolver)
        try:
            chunks = vol.tokenlist(chunk = True, chunk_target = chunk_size, overflow = 'ends', case=False, pos=False, page_ref = True)
            if chunks.empty:
                continue
            for (chunk, start, end), group in chunks.reset_index().groupby(['chunk', 'pstart', 'pend']):
                yield (id, chunk, start, end, group)
        except:
            print("Error chunking {}... skipping\n".format(id))
            continue


if __name__ == '__main__':
    main()