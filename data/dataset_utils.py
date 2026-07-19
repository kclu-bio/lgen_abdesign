import numpy as np
from collections import defaultdict

def _read_clusters(cluster_path, multichain=False)-> dict[str, list]:
    pdb_to_cluster = defaultdict(list)
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                if multichain:
                    # One item may contain multiple chains, each with its own cluster ID.
                    pdb = chain.rsplit("_", maxsplit=1)[0].strip()
                else:
                    pdb = chain.strip()
                if i not in pdb_to_cluster[pdb.upper()]:
                    pdb_to_cluster[pdb.upper()].append(i)
    return pdb_to_cluster

def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res)
        & (data_csv.modeled_seq_len <= max_res)
    ]
