import pandas as pd
import json
import os
import typing as T
import numpy as np
import torch
import matchms
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from matchms.importing import load_from_mgf

import massspecgym.utils as utils
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey, MetaTransform
from massspecgym.simulation_utils.misc_utils import flatten_lol

class NoveltyDataset(MassSpecDataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures with novelty labeling
    """

    def __init__(
            self, 
            split_with_labels_pth: str = '../../cluster_split.tsv'
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_with_labels_pth = split_with_labels_pth
    
    
    def load_data(self):
        super().load_data()
        split_with_labels = pd.read_csv(self.split_with_labels_pth, sep='\t')

        if "inchikey" not in self.metadata.columns:
              self.metadata["inchikey"] = self.metadata["smiles"].apply(utils.smiles_to_inchi_key)

        key_to_fold = split_with_labels.set_index("inchikey")["fold"]
        self.metadata["fold"] = self.metadata["inchikey"].map(key_to_fold)

        mask = self.metadata["fold"].notna()
        if not mask.all():
                print(f"[NoveltyDataset] dropping {(~mask).sum()} spectra not covered by the split")
                self.metadata = self.metadata[mask]
                self.spectra  = self.spectra[self.metadata.index].reset_index(drop=True)
                self.metadata = self.metadata.reset_index(drop=True)

