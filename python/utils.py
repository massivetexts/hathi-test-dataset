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



class MTVolume():
    def __init__(self, htid, vectorset = None, fullset = None):
        self.htid = htid
        if fullset is not None:
            vectorset = []
            for (i, name) in enumerate(fullset['names']):
                id = name.split("-")[0]
                if id == self.htid:
                    vectorset.append(fullset['matrix'][i])
            vectorset = np.array(vectorset)
            self.fullset = fullset
        self.vectorset = vectorset
        
    def brute_cosine(self, fullset = None):
        if fullset is None:
            fullset = self.fullset
        distances = np.dot(self.vectorset, np.transpose(fullset['matrix']))
        return distances
        
import json
import urllib

class HTID(object):
    def __init__(self, htid):
        self.htid = htid
        self.reader = None

    def _rsync_loc(self, root = "../../hathi-ef/"):
        loc = id_to_rsync(self.htid)
        return root + loc

    def volume(self):
        if self.reader is None:
            self.reader = FeatureReader(self._rsync_loc())
        return list(self.reader.volumes())[0]
    
    def _repr_html_(self):
        return self.volume()._repr_html_()
