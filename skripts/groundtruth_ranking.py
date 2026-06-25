import argparse
import os
import sys
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
_TOKENIZER = None

# get minimum MCES distance to training set for each molecule in fold
def get_min_mces_ranking(dists, train_mask, fold_mask, fold):
    print(f"computing min MCES ranking for {fold} fold")
    reduced_dists = dists[fold_mask, :][:, train_mask]
    return reduced_dists.min(axis=1)

# per-metric novelty flags and the consensus "novel label" for a fold
# a molecule is novel by a metric if:
#   tanimoto:          (1 - tanimoto sim) in the top 10% of the fold
#   fragment coverage: (1 - fragment coverage) in the top 10% of the fold
#   scaffold:          its murcko scaffold is unique (not seen in train)
#   mces:              min MCES distance to train >= 10
# novel label is 1 when at least three of the four metrics agree, else 0
def get_novel_label(fold_df, fold):
    print(f"computing novel label for {fold} fold")
    tanimoto = fold_df["tanimoto_distance"].astype(float)
    frag_novelty = 1.0 - fold_df["fragment_coverage"].astype(float)

    novel_tanimoto = tanimoto >= tanimoto.quantile(0.90)
    novel_fragment = frag_novelty >= frag_novelty.quantile(0.90)
    novel_scaffold = fold_df["unique_scaffold"].astype(bool)
    novel_mces = fold_df["min_mces"].astype(float) >= 10

    agreement = (
        novel_tanimoto.astype(int)
        + novel_fragment.astype(int)
        + novel_scaffold.astype(int)
        + novel_mces.astype(int)
    )
    return (agreement >= 3).astype(int)

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


def _init_coverage_worker(psvae_path, vocab_path):
    global _TOKENIZER
    if psvae_path not in sys.path:
        sys.path.insert(1, psvae_path)
    from mol_bpe import Tokenizer
    _TOKENIZER = Tokenizer(vocab_path)

# fraction of atoms covered by fragments of at least min_frag_size atoms
# fragments smaller than this are considered not meaningfully covered by the vocabulary
def _piece_coverage(tokenizer, smiles, min_frag_size=3):
    mol = tokenizer(smiles)
    total = covered = 0
    for n in mol.nodes:
        size = len(mol.get_node(n).atom_mapping)
        total += size
        if size >= min_frag_size:
            covered += size
    return covered / total if total else 0.0

def _coverage_worker(smiles):
    try:
        return _piece_coverage(_TOKENIZER, smiles, min_frag_size=3)
    except Exception as e:
        print(f"error occurred while processing {smiles}: {e}")
        return 0.0

# fragment coverage ranking for a fold: fraction of each molecule's atoms covered
# by vocabulary fragments, computed with the PS-VAE BPE tokenizer
def get_fragment_coverage(fold_smiles, fold, psvae_path, vocab_path, n_jobs):
    print(f"computing fragment coverage for {fold} fold")
    with Pool(
        n_jobs,
        initializer=_init_coverage_worker,
        initargs=(psvae_path, vocab_path),
    ) as cpool:
        return cpool.map(_coverage_worker, fold_smiles, chunksize=128)


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
        default="cluster_split.tsv",
        help="output TSV path",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="number of worker processes",
    )
    parser.add_argument(
        "--psvae-path",
        default="../../../PS-VAE/ps",
        help="path to the PS-VAE 'ps' package providing the BPE Tokenizer",
    )
    parser.add_argument(
        "--vocab-path",
        default="data/fragmentation_vocab.txt",
        help="path to the fragmentation vocabulary file",
    )
    args = parser.parse_args()
    psvae_path = os.path.abspath(args.psvae_path)

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
    split["min_mces"] = np.nan
    split["fragment_coverage"] = np.nan

    for fold in ("val", "test"):
        scaffold_ranking = get_scaffold_ranking(
            scaffolds["train"], scaffolds[fold], fold
        )
        tanimoto_ranking = get_tanimoto_ranking(
            fold_smiles["train"], fold_smiles[fold], fold, pool, args.n_jobs
        )
        fragment_coverage = get_fragment_coverage(
            fold_smiles[fold], fold, psvae_path, args.vocab_path, args.n_jobs
        )
        fold_idx = split.index[split["fold"] == fold]
        split.loc[fold_idx, "unique_scaffold"] = scaffold_ranking
        split.loc[fold_idx, "tanimoto_distance"] = tanimoto_ranking
        split.loc[fold_idx, "fragment_coverage"] = fragment_coverage

    with h5py.File(args.mces_path, "r") as f:
        dists = squareform(f["mces"][:])
        dists_smiles = f["mces_smiles_order"][:].astype(str).tolist()
    pool.close()
    pool.join()
    fold_smiles_sets = {fold: set(smiles) for fold, smiles in fold_smiles.items()}
    masks = {
        fold: np.array([s in smiles for s in dists_smiles])
        for fold, smiles in fold_smiles_sets.items()
    }

    for fold in ("val", "test"):
        ranking = get_min_mces_ranking(
            dists, masks["train"], masks[fold], fold
        )
        ranking_by_smiles = dict(
            zip([s for s, m in zip(dists_smiles, masks[fold]) if m], ranking)
        )
        fold_idx = split.index[split["fold"] == fold]
        split.loc[fold_idx, "min_mces"] = split.loc[fold_idx, "smiles"].map(
            ranking_by_smiles
        )

    split["novel_label"] = pd.NA
    for fold in ("val", "test"):
        fold_idx = split.index[split["fold"] == fold]
        split.loc[fold_idx, "novel_label"] = get_novel_label(
            split.loc[fold_idx], fold
        )

    split.to_csv(args.output, sep="\t", index=False)
    print(f"wrote ranking to {args.output}")


if __name__ == "__main__":
    main()
