import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from massspecgym.data.data_module import MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer
from massspecgym.models.base import Stage
from massspecgymnovelty.data.novelty_dataset import NoveltyDataset
from massspecgymnovelty.data.transforms import DreaMSEmbedding
from massspecgymnovelty.models.dreams_baseline import DreaMSNoveltyBaseline
from massspecgymnovelty.models.modified_cosine import ModifiedCosineBaseline

# Majority of the code is taken from MassSpecGym run.py script and updated for novelty detection task

MASSSPECGYM_NOVELTY_ROOT = Path(__file__).parent.parent.absolute()
MASSSPECGYM_TEST_RESULTS_DIR = MASSSPECGYM_NOVELTY_ROOT / "data" / "test_results"
EMB_KEY = "DreaMS_embedding"

parser = argparse.ArgumentParser()

# submission
parser.add_argument("--job_key", type=str, required=True)

# experiment setup
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--project_name", type=str, default="MassSpecGym_novelty")
parser.add_argument("--wandb_entity_name", type=str, default="mass-spec-ml")
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--test_only", action="store_true")

# data
parser.add_argument("--dataset_pth", type=str, default=None,
    help="Path to the dataset file in the .tsv or .mgf format.")
parser.add_argument("--split_with_labels_pth", type=str,
    default=str(MASSSPECGYM_NOVELTY_ROOT / "data" / "cluster_split.tsv"))
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--max_mz", type=int, default=1005)
parser.add_argument("--n_peaks", type=int, default=60)
parser.add_argument("--dreams_emb_pth", type=str,
    default=str(MASSSPECGYM_NOVELTY_ROOT / "data" / "MassSpecGym1.5_updated.hdf5"),
    help="Path to precomputed DreaMS embeddings (see skripts/precompute_dreams_embeddings.py).")

# training setup
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)

# model
parser.add_argument("--model", type=str, required=True, choices=["dreams", "modified_cosine"])  # add more as you build them
parser.add_argument("--log_only_loss_at_stages", default=(),
    type=lambda s: [Stage(x) for x in s.strip().replace(" ", "").split(",")] if s else ())
parser.add_argument("--df_test_pth", type=Path, default=None)
parser.add_argument("--checkpoint_pth", type=Path, default=None)


def main(args):
    pl.seed_everything(args.seed)
    now_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.df_test_pth is None:
        args.df_test_pth = MASSSPECGYM_TEST_RESULTS_DIR / f"{args.run_name}_{now_formatted}.pkl"

    spec_tokenizer = SpecTokenizer(n_peaks=args.n_peaks, matchms_kwargs=dict(mz_to=args.max_mz))

    # The spectrum representation is algorithm-specific and chosen here, not in the
    # dataset. For DreaMS we attach a precomputed embedding (looked up by identifier)
    # alongside the tokenized spectrum.
    if args.model == "dreams":
        spec_transform = {
            "spec": spec_tokenizer,
            EMB_KEY: DreaMSEmbedding(args.dreams_emb_pth),
        }
    else:
        spec_transform = spec_tokenizer

    dataset = NoveltyDataset(
        pth=args.dataset_pth,
        split_with_labels_pth=args.split_with_labels_pth,
        spec_transform=spec_transform,
        mol_transform=None,
    )

    data_module = MassSpecDataModule(
        dataset=dataset,
        split_pth=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    common_kwargs = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_only_loss_at_stages=args.log_only_loss_at_stages,
        df_test_path=args.df_test_pth,
    )
    if args.model == "dreams":
        model = DreaMSNoveltyBaseline(embedding_key=EMB_KEY, **common_kwargs)
    elif args.model == "modified_cosine":
        model = ModifiedCosineBaseline(**common_kwargs)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

    if args.checkpoint_pth is not None:
        model = type(model).load_from_checkpoint(
            args.checkpoint_pth,
            log_only_loss_at_stages=args.log_only_loss_at_stages,
            df_test_path=args.df_test_pth,
        )

    if args.no_wandb:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            name=args.run_name,
            project=args.project_name,
            log_model=False,
            config=args,
        )

    callbacks = []
    for i, monitor in enumerate(model.get_checkpoint_monitors()):
        monitor_name = monitor["monitor"]
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor_name,
            save_top_k=1,
            mode=monitor["mode"],
            dirpath=Path(args.project_name) / args.job_key,
            filename=f"{{step:06d}}-{{{monitor_name}:03.03f}}",
            auto_insert_metric_name=True,
            save_last=(i == 0),
        )
        callbacks.append(checkpoint)
        if monitor.get("early_stopping", False):
            callbacks.append(EarlyStopping(
                monitor=monitor_name,
                mode=monitor["mode"],
                verbose=True,
            ))

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )

    data_module.prepare_data()
    data_module.setup()

    if not args.test_only:
        trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
