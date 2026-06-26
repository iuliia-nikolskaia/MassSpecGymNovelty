import argparse
import csv
from pathlib import Path


def load_inchikey_to_fold(tsv_path):
    mapping = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapping[row["inchikey"]] = row["fold"]
    return mapping


def update_mgf(in_path, out_path, inchikey_to_fold):
    block = []
    block_inchikey = None
    fold_line_idx = None

    n_blocks = 0
    n_updated = 0
    n_missing = 0
    missing_inchikeys = set()

    with open(in_path, "r") as fin, open(out_path, "w", newline="\n") as fout:
        for line in fin:
            stripped = line.rstrip("\r\n")

            if stripped == "BEGIN IONS":
                block = [line]
                block_inchikey = None
                fold_line_idx = None
                continue

            if not block:
                fout.write(line)
                continue

            block.append(line)

            if stripped.startswith("INCHIKEY="):
                block_inchikey = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("FOLD="):
                fold_line_idx = len(block) - 1

            if stripped == "END IONS":
                n_blocks += 1
                new_fold = inchikey_to_fold.get(block_inchikey)
                if new_fold is None:
                    n_missing += 1
                    if block_inchikey is not None:
                        missing_inchikeys.add(block_inchikey)
                else:
                    new_fold_line = f"FOLD={new_fold}\n"
                    if fold_line_idx is not None:
                        if block[fold_line_idx] != new_fold_line:
                            n_updated += 1
                        block[fold_line_idx] = new_fold_line
                    else:
                        block.insert(-1, new_fold_line)
                        n_updated += 1

                fout.writelines(block)
                block = []
                block_inchikey = None
                fold_line_idx = None

        if block:
            fout.writelines(block)

    print(f"blocks processed: {n_blocks}")
    print(f"blocks with updated FOLD: {n_updated}")
    print(f"blocks with inchikey missing from split: {n_missing}")
    if missing_inchikeys:
        sample = list(missing_inchikeys)[:10]
        print(f"  {len(missing_inchikeys)} unique missing inchikeys, sample: {sample}")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite FOLD= lines in an MGF using an inchikey->fold mapping."
    )
    parser.add_argument(
        "--mgf",
        default="data/MassSpecGym.mgf",
        help="input MGF path",
    )
    parser.add_argument(
        "--split",
        default="data/cluster_split.tsv",
        help="TSV with inchikey,fold columns",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output MGF path (default: <input>.relabeled.mgf next to the input)",
    )
    args = parser.parse_args()

    in_path = Path(args.mgf)
    out_path = (
        Path(args.output)
        if args.output is not None
        else in_path.with_suffix(".relabeled.mgf")
    )

    print(f"loading inchikey->fold from {args.split}...")
    mapping = load_inchikey_to_fold(args.split)
    print(f"  {len(mapping)} inchikeys in split")

    print(f"rewriting {in_path} -> {out_path}...")
    update_mgf(in_path, out_path, mapping)


if __name__ == "__main__":
    main()
