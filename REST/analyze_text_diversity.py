import argparse
import csv
import glob
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine similarity for each embedding group in a pickle file."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Pickle file paths or glob patterns such as '/path/*_anomaly.pkl'.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=40,
        help="Only analyze embedding groups whose first dimension matches this size.",
    )
    parser.add_argument(
        "--pooling",
        choices=["masked_mean", "mean", "cls"],
        default="masked_mean",
        help="How to convert 3D embeddings (N, C, T) into per-sample vectors.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save each group's cosine similarity matrix as CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to save a summary CSV across all analyzed files and groups.",
    )
    return parser.parse_args()


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def pool_embeddings(embedding, attention_mask, pooling):
    if embedding.ndim == 2:
        return embedding.float()

    if embedding.ndim != 3:
        raise ValueError(f"Unsupported embedding shape: {tuple(embedding.shape)}")

    embedding = embedding.float()
    if pooling == "cls":
        return embedding[:, :, 0]

    if pooling == "mean":
        return embedding.mean(dim=-1)

    if attention_mask is None:
        raise ValueError("masked_mean pooling requires a matching attention mask.")

    mask = attention_mask.float().unsqueeze(1)
    valid_count = mask.sum(dim=-1).clamp_min(1.0)
    return (embedding * mask).sum(dim=-1) / valid_count


def cosine_similarity_matrix(vectors):
    normalized = F.normalize(vectors, p=2, dim=1)
    return normalized @ normalized.t()


def strict_upper_values(matrix):
    n = matrix.shape[0]
    row_idx, col_idx = torch.triu_indices(n, n, offset=1)
    return matrix[row_idx, col_idx]


def save_matrix_csv(matrix, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in matrix.tolist():
            f.write(",".join(f"{value:.6f}" for value in row) + "\n")


def analyze_group(file_path, name, embedding, attention_mask, pooling, save_dir):
    vectors = pool_embeddings(embedding, attention_mask, pooling)
    sim_matrix = cosine_similarity_matrix(vectors)
    pair_values = strict_upper_values(sim_matrix)
    cosine_mean = pair_values.mean().item()
    cosine_std = pair_values.std(unbiased=False).item()
    cosine_min = pair_values.min().item()
    cosine_max = pair_values.max().item()
    diversity = 1.0 - cosine_mean

    print(f"[{name}]")
    print(f"  raw_shape={tuple(embedding.shape)} pooled_shape={tuple(vectors.shape)}")
    print(f"  pair_count={pair_values.numel()}")
    print(f"  cosine_mean={cosine_mean:.6f}")
    print(f"  cosine_std={cosine_std:.6f}")
    print(f"  cosine_min={cosine_min:.6f}")
    print(f"  cosine_max={cosine_max:.6f}")
    print(f"  diversity_1_minus_mean={diversity:.6f}")

    if save_dir is not None:
        out_path = save_dir / f"{file_path.stem}__{name}_cosine_similarity.csv"
        save_matrix_csv(sim_matrix, out_path)
        print(f"  saved_matrix={out_path}")

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "dataset": file_path.stem.replace("_anomaly", ""),
        "group_name": name,
        "raw_shape": str(tuple(embedding.shape)),
        "pooled_shape": str(tuple(vectors.shape)),
        "pair_count": int(pair_values.numel()),
        "cosine_mean": cosine_mean,
        "cosine_std": cosine_std,
        "cosine_min": cosine_min,
        "cosine_max": cosine_max,
        "diversity_1_minus_mean": diversity,
    }


def expand_inputs(inputs):
    paths = []
    for item in inputs:
        if any(char in item for char in "*?[]"):
            paths.extend(Path(p) for p in sorted(glob.glob(item)))
        else:
            paths.append(Path(item))

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def write_summary_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path",
        "file_name",
        "dataset",
        "group_name",
        "raw_shape",
        "pooled_shape",
        "pair_count",
        "cosine_mean",
        "cosine_std",
        "cosine_min",
        "cosine_max",
        "diversity_1_minus_mean",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_file(file_path, group_size, pooling, save_dir):
    info = load_pickle(file_path)

    if not isinstance(info, dict):
        raise TypeError(f"Expected a dict in {file_path}, got {type(info)}")

    analyzed = 0
    rows = []
    print(f"file={file_path}")
    print(f"pooling={pooling}")
    print(f"group_size={group_size}")

    for key, value in info.items():
        if not key.endswith("_embedding"):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if value.shape[0] != group_size:
            continue

        mask_key = key.replace("_embedding", "_attention_mask")
        attention_mask = info.get(mask_key)
        rows.append(
            analyze_group(
                file_path=file_path,
                name=key.replace("_embedding", ""),
                embedding=value,
                attention_mask=attention_mask,
                pooling=pooling,
                save_dir=save_dir,
            )
        )
        analyzed += 1

    if analyzed == 0:
        raise ValueError(
            f"No embedding groups with first dimension {group_size} were found in {file_path}."
        )

    return rows


def main():
    args = parse_args()
    file_paths = expand_inputs(args.inputs)
    if not file_paths:
        raise ValueError("No input pickle files matched the provided paths or glob patterns.")

    all_rows = []
    for index, file_path in enumerate(file_paths):
        if index > 0:
            print()
        all_rows.extend(
            analyze_file(
                file_path=file_path,
                group_size=args.group_size,
                pooling=args.pooling,
                save_dir=args.save_dir,
            )
        )

    if args.summary_csv is not None:
        write_summary_csv(all_rows, args.summary_csv)
        print()
        print(f"summary_csv={args.summary_csv}")


if __name__ == "__main__":
    main()
