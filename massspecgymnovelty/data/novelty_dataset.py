import pandas as pd
import torch
from massspecgym.data.datasets import MassSpecDataset
import massspecgym.utils as utils

class NoveltyDataset(MassSpecDataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures with novelty labeling
    """

    def __init__(
        self,
        split_with_labels_pth: str = '../../cluster_split.tsv',
        **kwargs,
    ):
        self.label_col = "novel_label"
        self.split_with_labels_pth = split_with_labels_pth
        super().__init__(**kwargs)


    def load_data(self):
        super().load_data()
        split_with_labels = pd.read_csv(self.split_with_labels_pth, sep='\t')

        if "inchikey" not in self.metadata.columns:
            self.metadata["inchikey"] = self.metadata["smiles"].apply(utils.smiles_to_inchi_key)

        indexed = split_with_labels.set_index("inchikey")
        key_to_fold = indexed["fold"]
        self.metadata["fold"] = self.metadata["inchikey"].map(key_to_fold)

        self.metadata["novelty_label"] = self.metadata["inchikey"].map(indexed[self.label_col]).astype(float)

        mask = self.metadata["fold"].notna()
        if not mask.all():
            print(f"[NoveltyDataset] dropping {(~mask).sum()} spectra not covered by the split")
            self.metadata = self.metadata[mask]
            self.spectra  = self.spectra[self.metadata.index].reset_index(drop=True)
            self.metadata = self.metadata.reset_index(drop=True)

        if "identifier" in self.metadata.columns:
            for spec, identifier in zip(self.spectra, self.metadata["identifier"]):
                spec.set("identifier", identifier)

    def __getitem__(self, i) -> dict:
        item = super().__getitem__(i)
        metadata = self.metadata.iloc[i]
        item["labels"] = torch.as_tensor(metadata["novelty_label"], dtype=self.dtype)
        return item
