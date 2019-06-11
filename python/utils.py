import os
import pandas as pd
from htrc_features import FeatureReader
from htrc_features.feature_reader import group_tokenlist
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np

def page_counts(id, hathi_loc = "../../hathi-ef/"):
    loc = hathi_loc + id_to_rsync(id)
    parquet_loc = loc.replace(".json.bz2", ".parquet")
    if os.path.exists(parquet_loc):
        return pd.read_parquet(parquet_loc)
    vol = list(FeatureReader([loc]).volumes())[0]
    table = vol.tokenlist().groupby(["page", "token"])['count'].sum().reset_index()
    table.to_parquet(parquet_loc)
    return table

class Comparison(object):
    def __init__(self, left, right, labels = ['left', 'right']):
        """
        initialized with two dataframes columned ['page/chunk/etc', 'token', 'count']
        """
        self.left = left
        self.right = right

    def jaccard(self, document = "page"):
        """
        Return pairwise jaccard similarities across pages.

        'document': The column in the initialized dataframe containing page-level info. 
        """

        chunking = column
        
        ## Only works on page chunks for right now.
        left = self.left
        right = self.right
        
        l = left.drop(labels = ["count"], axis = 1)
        r = right.drop(labels = ["count"], axis = 1)

        l_lookup = l.groupby(chunking)['token'].count()
        r_lookup = r.groupby(chunking)['token'].count()

        llab = "{}_x".format(chunking)
        rlab ="{}_y".format(chunking)
        
        merged = pd.merge(l,
                          r,
                          on = 'token').groupby([llab, rlab]).count().reset_index()

        output = []

        for i, row in enumerate(merged.iterrows()):
            row = dict(row[1])
            output.append(row['token']/(l_lookup[row[llab]] + r_lookup[row[rlab]] - row['token']))

        merged['jaccard_sim'] = output
        return merged


