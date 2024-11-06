import itertools
import pandas as pd

import pandas as pd


def read_clstr(file_path):
    """
    Parses a CD-HIT cluster file and returns a dictionary where keys are cluster numbers
    and values are lists of sequence identifiers in each cluster.
    """
    with open(file_path) as cluster_file:
        # Initialize an empty dictionary to store clusters
        cluster_dic = {}
        
        # Group lines by whether they start with '>', indicating a new cluster
        cluster_groups = itertools.groupby(cluster_file, key=lambda line: line.startswith('>'))
        
        # Iterate through grouped lines
        for is_new_cluster, group in cluster_groups:
            if is_new_cluster:
                # Extract cluster number from the cluster header
                cluster_number = next(group)[1:].strip().split(' ')[1]
            else:
                # List comprehension to extract sequence identifiers for the current cluster
                seqs = [seq.split('>')[1].split('...')[0] for seq in group]
                cluster_dic[cluster_number] = seqs

    return cluster_dic