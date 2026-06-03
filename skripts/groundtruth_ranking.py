import argparse
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from massspecgym.utils import morgan_fp
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from scipy.spatial.distance import squareform


# computation was done on MetaCentrum with 16 cpus and 50gb of RAM  memory 
# after parallelization with Pool, script runs in few minutes, without it ~12 hours

_TRAIN_FPS = None

# get number of edges in a molecule, used for normalizing MCES distances
def _num_edges_worker(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumBonds() if mol is not None else np.nan

# get normalized MCES distances: absolute MCES distance (mol1, mol2) / max (num_edges(mol1), num_edges(mol2))
def get_normalized_mces(dists, dists_smiles, pool):
    print("computing normalized distances")
    edge_counts = np.array(
        pool.map(_num_edges_worker, dists_smiles, chunksize=256)
    )
    norm_matrix = np.maximum.outer(edge_counts, edge_counts)
    norm_matrix[norm_matrix == 0] = 1
    return dists / norm_matrix

# get minimum normalized MCES distance to training set for each molecule in fold
def get_min_normalized_mces_ranking(normalized_dists, train_mask, fold_mask, fold):
    print(f"computing min normalized MCES ranking for {fold} fold")
    reduced_dists = normalized_dists[fold_mask, :][:, train_mask]
    return reduced_dists.min(axis=1)

# get scaffold for a molecule in smiles format
def _scaffold_worker(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(GetScaffoldForMol(mol))

# get scaffolds for a list of smiles
def get_scaffolds(smiles_list, fold, pool):
    print(f"computing scaffolds for {fold} fold")
    return pool.map(_scaffold_worker, smiles_list, chunksize=128)

# get scaffold ranking for fold: scaffold is considered unique if it is not present in training set
def get_scaffold_ranking(train_scaffolds, fold_scaffolds, fold):
    print(f"computing scaffold ranking for {fold} fold")
    train_set = set(train_scaffolds)
    is_unique = [s not in train_set for s in fold_scaffolds]
    print(f"in {fold} fold {sum(is_unique)} unique scaffolds were found")
    return is_unique

# get morgan fingerprint for a molecule in smiles format
def _fp_worker(smi):
    mol = Chem.MolFromSmiles(smi)
    return morgan_fp(mol, to_np=False)

def _init_tanimoto_worker(train_fps):
    global _TRAIN_FPS
    _TRAIN_FPS = train_fps

# get max tanimoto similarity to training set for a molecule in fold
# tanimoto distance is later defined as 1 - max_tanimoto_similarity (mol, training set)
def _max_tanimoto_worker(fp):
    return max(DataStructs.BulkTanimotoSimilarity(fp, _TRAIN_FPS))

# for molecules in train and fold of interest, get tanimoto rankng
# paralelizing reduces computational time from hours to minutes on MetaCentrum, tanimoto calculations were the biggest bottleneck
def get_tanimoto_ranking(train_smiles, fold_smiles, fold, pool, n_jobs):
    print(f"computing tanimoto ranking for {fold} fold")
    train_fps = pool.map(_fp_worker, train_smiles, chunksize=128)
    fold_fps = pool.map(_fp_worker, fold_smiles, chunksize=128)
    with Pool(n_jobs, initializer=_init_tanimoto_worker, initargs=(train_fps,)) as tpool:
        max_sims = tpool.map(_max_tanimoto_worker, fold_fps, chunksize=32)
    return [1 - s for s in max_sims]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mces-path",
        default="all_smiles_mces.hdf5",
        help="path to the MCES distance hdf5 file",
    )
    parser.add_argument(
        "--split-path",
        default="cluster_split.csv",
        help="path to CSV file with split",
    )
    parser.add_argument(
        "--output",
        default="cluster_split.csv",
        help="output CSV path",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="number of worker processes",
    )
    args = parser.parse_args()

    split = pd.read_csv(args.split_path)
    fold_smiles = {
        fold: split.loc[split["fold"] == fold, "smiles"].tolist()
        for fold in ("train", "val", "test")
    }

    pool = Pool(args.n_jobs)

    scaffolds = {
        fold: get_scaffolds(smiles, fold, pool)
        for fold, smiles in fold_smiles.items()
    }

    split["tanimoto_distance"] = pd.NA
    split["unique_scaffold"] = pd.NA
    split["min_normalized_mces"] = np.nan

    for fold in ("val", "test"):
        scaffold_ranking = get_scaffold_ranking(
            scaffolds["train"], scaffolds[fold], fold
        )
        tanimoto_ranking = get_tanimoto_ranking(
            fold_smiles["train"], fold_smiles[fold], fold, pool, args.n_jobs
        )
        fold_idx = split.index[split["fold"] == fold]
        split.loc[fold_idx, "unique_scaffold"] = scaffold_ranking
        split.loc[fold_idx, "tanimoto_distance"] = tanimoto_ranking

    with h5py.File(args.mces_path, "r") as f:
        dists = squareform(f["mces"][:])
        dists_smiles = f["mces_smiles_order"][:].astype(str).tolist()

    normalized_dists = get_normalized_mces(dists, dists_smiles, pool)
    pool.close()
    pool.join()
    fold_smiles_sets = {fold: set(smiles) for fold, smiles in fold_smiles.items()}
    masks = {
        fold: np.array([s in smiles for s in dists_smiles])
        for fold, smiles in fold_smiles_sets.items()
    }

    for fold in ("val", "test"):
        ranking = get_min_normalized_mces_ranking(
            normalized_dists, masks["train"], masks[fold], fold
        )
        ranking_by_smiles = dict(
            zip([s for s, m in zip(dists_smiles, masks[fold]) if m], ranking)
        )
        fold_idx = split.index[split["fold"] == fold]
        split.loc[fold_idx, "min_normalized_mces"] = split.loc[fold_idx, "smiles"].map(
            ranking_by_smiles
        )

    split.to_csv(args.output, index=False)
    print(f"wrote ranking to {args.output}")


if __name__ == "__main__":
    main()
