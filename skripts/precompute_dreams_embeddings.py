import argparse
from pathlib import Path

import h5py
import numpy as np

from dreams.api import dreams_embeddings
from dreams.utils.data import MSData

MASSSPECGYM_NOVELTY_ROOT = Path(__file__).parent.parent.absolute()


def main(args):
    # mgf -> MSData object
    msdata = MSData.from_mgf(args.mgf_pth)
    print(f"[precompute] {len(msdata)} spectra to embed")

    # calc embeddings and store them in output .hdf5 file 
    msdata = None
    msdata = MSData.from_hdf5(Path(args.mgf_pth).with_suffix('.hdf5'), mode='a')
    embeddings = dreams_embeddings(msdata, batch_size=args.batch_size, store_embs=True)
    print(f"[precompute] embeddings shape: {embeddings.shape}")

    identifiers = [str(i) for i in msdata.get_values("IDENTIFIER")]
    assert embeddings.shape[0] == len(identifiers), (
        f"embedding count {embeddings.shape[0]} != identifier count {len(identifiers)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf_pth", type=str,
        default=str(MASSSPECGYM_NOVELTY_ROOT / "data" / "MassSpecGym1.5_updated.mgf"),
        help="Path to the input .mgf.")
    parser.add_argument("--out_pth", type=str,
        default=str(MASSSPECGYM_NOVELTY_ROOT / "data" / "dreams_embeddings.hdf5"))
    parser.add_argument("--batch_size", type=int, default=32)
    main(parser.parse_args())
