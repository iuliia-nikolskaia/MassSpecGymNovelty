

import argparse
import gc
import os
import random

import h5py
import numpy as np
from massspecgym.utils import load_massspecgym
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TOTAL_TEST_VAL_BUDGET = 5700


def load_distance_submatrix(mces_path, unique_smiles):

    unique_smiles_set = set(unique_smiles)

    with h5py.File(mces_path, "r") as f:
        dists_smiles = f["mces_smiles_order"][:].astype(str)
        condensed = f["mces"][:].astype(np.float32, copy=False)

    smiles_mask = np.array([s in unique_smiles_set for s in dists_smiles])
    print(
        f"loaded {len(dists_smiles)} molecules from hdf5, "
        f"{smiles_mask.sum()} match unique_smiles"
    )

    dists_full = squareform(condensed)
    del condensed
    gc.collect()

    dists_unique = dists_full[smiles_mask, :][:, smiles_mask]
    del dists_full
    gc.collect()

    unique_mols_order = dists_smiles[smiles_mask]
    return dists_unique, unique_mols_order


def cluster_at_threshold(matrix, threshold):
    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    ).fit(matrix)
    return clustering.labels_


def assign_groups(unique_inchikey_msg, dists_unique, unique_mols_order):
    unique_inchikey_msg['group'] = -1
    copy_matrix = dists_unique
    copy_order = unique_mols_order

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='single',
        distance_threshold=12,
        n_clusters=None
    ).fit(dists_unique)
    clusters12 = clustering.labels_
    unique_clusters, counts = np.unique(clusters12, return_counts=True)
    outliers = unique_clusters[counts < 3] 

    binary_clusters = np.isin(clusters12, outliers).astype(int)
    smiles_to_cluster = dict(zip(copy_order, binary_clusters))
    test_val_smiles = [smiles for smiles, cluster in smiles_to_cluster.items() if cluster != 0]
    unique_inchikey_msg.loc[unique_inchikey_msg['smiles'].isin(test_val_smiles), 'group'] = 12
    not_in_test_val_mask = [s not in test_val_smiles for s in copy_order]

    copy_matrix = copy_matrix[not_in_test_val_mask, :][:, not_in_test_val_mask]
    copy_order = copy_order[not_in_test_val_mask]

    for threshold in range(11, 0, -1):
        clustering = AgglomerativeClustering(
            metric='precomputed',
            linkage='single',
            distance_threshold=threshold,
            n_clusters=None
        ).fit(copy_matrix)
        clusters = clustering.labels_

        unique_clusters, counts = np.unique(clusters, return_counts=True)
        outliers = unique_clusters[counts < 2] 

        # train_clusters, test_val_clusters = train_test_split(outliers, test_size=0.07, random_state=42)

        binary_clusters = np.isin(clusters, outliers).astype(int)
        smiles_to_cluster = dict(zip(copy_order, binary_clusters))

        test_val_candidates = [smiles for smiles, cluster in smiles_to_cluster.items() if cluster != 0]
        unique_inchikey_msg.loc[unique_inchikey_msg['smiles'].isin(test_val_candidates), 'group'] = threshold


        not_in_test_val_mask = [s not in test_val_candidates for s in copy_order]
        
        copy_matrix = copy_matrix[not_in_test_val_mask, :][:, not_in_test_val_mask]
        copy_order = copy_order[not_in_test_val_mask]

    print("group counts:")
    print(unique_inchikey_msg["group"].value_counts().sort_index())


def sample_test_val_pool(unique_inchikey_msg):
    groups = []
    for i in range(1, 13):
        groups.append(unique_inchikey_msg[unique_inchikey_msg['group'] == i]['smiles'][:].astype(str).tolist())

    probabilty = []
    k = 0
    for i in range(1, 11):
        k += 1/i
    for i in range(1, 11):
        probabilty.append(round(TOTAL_TEST_VAL_BUDGET*((1/i)/k)))
    
    test_val_molecules = []

    for i in range(10):
        test_val_molecules.extend(random.sample(groups[i], probabilty[i]))
    test_val_molecules.extend(groups[10])
    test_val_molecules.extend(groups[11])

    return test_val_molecules


def assign_folds(unique_inchikey_msg, test_val_molecules):
    test_molecules, val_molecules = train_test_split(test_val_molecules, test_size=0.5, random_state=42)
    test_set = set(test_molecules)
    val_set = set(val_molecules)

    unique_inchikey_msg["fold"] = "train"
    unique_inchikey_msg.loc[
        unique_inchikey_msg["smiles"].isin(test_set), "fold"
    ] = "test"
    unique_inchikey_msg.loc[
        unique_inchikey_msg["smiles"].isin(val_set), "fold"
    ] = "val"

    print("fold counts:")
    print(unique_inchikey_msg["fold"].value_counts())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mces-path",
        default="all_smiles_mces.hdf5",
        help="path to the MCES distance hdf5 file",
    )
    parser.add_argument(
        "--output",
        default="cluster_split.csv",
        help="output CSV path",
    )
    parser.add_argument(
        "--smiles-dir",
        default=None,
        help="folder to save train_smiles.txt, val_smiles.txt and test_smiles.txt",
    )
    args = parser.parse_args()

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("loading MassSpecGym...")
    msg = load_massspecgym()
    # Remove all rows with spectra that has only one signal. 
    # some of these spectra can come from different molecules, but the only peak have the same mass
    # making them nearly identical for the models
    msg = msg[msg['mzs'].str.split(',').apply(len) >= 2]
    unique_inchikey_msg = msg.drop_duplicates(subset="inchikey")
    unique_smiles = unique_inchikey_msg["smiles"].astype(str).tolist()

    print(f"loading distance matrix from {args.mces_path}...")
    dists_unique, unique_mols_order = load_distance_submatrix(
        args.mces_path, unique_smiles
    )
    print(f"distance submatrix: {dists_unique.shape}, dtype={dists_unique.dtype}")

    assign_groups(unique_inchikey_msg, dists_unique, unique_mols_order)

    test_val_molecules = sample_test_val_pool(unique_inchikey_msg)
    assign_folds(unique_inchikey_msg, test_val_molecules)

    unique_inchikey_msg[["smiles", "inchikey", "group", "fold"]].to_csv(
        args.output, index=False
    )
    print(f"wrote split to {args.output}")

    if args.smiles_dir is not None:
        os.makedirs(args.smiles_dir, exist_ok=True)
        for fold in ("train", "val", "test"):
            smiles = unique_inchikey_msg.loc[
                unique_inchikey_msg["fold"] == fold, "smiles"
            ].astype(str).tolist()
            path = os.path.join(args.smiles_dir, f"{fold}_smiles.txt")
            with open(path, "w") as f:
                f.write("\n".join(smiles) + "\n")
            print(f"wrote {len(smiles)} smiles to {path}")


if __name__ == "__main__":
    main()
