import numpy as np
import h5py
import matchms


class DreaMSEmbedding:

    def __init__(self, emb_pth, id_key: str = "IDENTIFIER", emb_key: str = "DreaMS_embedding"):
        self.emb_pth = str(emb_pth)
        self.emb_key = emb_key
        with h5py.File(self.emb_pth, "r") as f:
            ids = f[id_key][()]
            self.embeddings = f[emb_key][()].astype(np.float32)
        # HDF5 strings come back as bytes; normalise to str for dict lookup.
        ids = [i.decode() if isinstance(i, bytes) else str(i) for i in ids]
        self.id_to_row = {identifier: row for row, identifier in enumerate(ids)}

    def __call__(self, spec: matchms.Spectrum) -> np.ndarray:
        identifier = spec.get("identifier")
        if identifier is None:
            raise KeyError(
                "Spectrum has no 'identifier' in its metadata; cannot look up its DreaMS "
                "embedding. NoveltyDataset is expected to attach identifiers to spectra."
            )
        identifier = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
        row = self.id_to_row.get(identifier)
        if row is None:
            raise KeyError(
                f"No precomputed DreaMS embedding for identifier {identifier!r} in "
                f"{self.emb_pth}. Re-run the precompute script on the same dataset."
            )
        return self.embeddings[row]
