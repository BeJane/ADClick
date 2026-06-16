import argparse
import csv
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pairwise Levenshtein edit distance for the first N sentences in each group."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON file paths or glob patterns such as '/path/*.json'.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset key override. Only use this when all inputs share the same top-level key.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Number of leading sentences per group to analyze.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to save the summary CSV.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save each group's full distance matrix as CSV.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


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


def resolve_dataset_key(data, json_path, dataset_arg):
    if dataset_arg is not None:
        if dataset_arg not in data:
            raise KeyError(f"Dataset key {dataset_arg!r} not found in {json_path}")
        return dataset_arg

    if len(data) == 1:
        return next(iter(data))

    stem = json_path.stem
    if stem in data:
        return stem

    raise ValueError(
        f"Could not infer dataset key from {json_path}. Available keys: {list(data.keys())}"
    )


def levenshtein_distance(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_distance_matrix(sentences):
    n = len(sentences)
    matrix = [[0] * n for _ in range(n)]
    pair_values = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = levenshtein_distance(sentences[i], sentences[j])
            matrix[i][j] = dist
            matrix[j][i] = dist
            pair_values.append(dist)
    return matrix, pair_values


def mean(values):
    return sum(values) / len(values)


def std(values, mean_value):
    return (sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5


def save_matrix_csv(sentences, matrix, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence'] + [f's{i:02d}' for i in range(len(sentences))])
        for idx, (sentence, row) in enumerate(zip(sentences, matrix)):
            writer.writerow([f's{idx:02d}: {sentence}'] + row)


def write_summary_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'json_path',
        'json_name',
        'dataset',
        'group_name',
        'sentence_count',
        'pair_count',
        'edit_distance_mean',
        'edit_distance_std',
        'edit_distance_min',
        'edit_distance_max',
    ]
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_group(json_path, dataset_key, group_name, sentences, save_dir):
    matrix, pair_values = compute_distance_matrix(sentences)
    mean_value = mean(pair_values)
    std_value = std(pair_values, mean_value)
    min_value = min(pair_values)
    max_value = max(pair_values)

    print(f'[{group_name}]')
    print(f'  sentence_count={len(sentences)}')
    print(f'  pair_count={len(pair_values)}')
    print(f'  edit_distance_mean={mean_value:.6f}')
    print(f'  edit_distance_std={std_value:.6f}')
    print(f'  edit_distance_min={min_value}')
    print(f'  edit_distance_max={max_value}')

    if save_dir is not None:
        out_path = save_dir / f'{json_path.stem}__{group_name}_edit_distance.csv'
        save_matrix_csv(sentences, matrix, out_path)
        print(f'  saved_matrix={out_path}')

    return {
        'json_path': str(json_path),
        'json_name': json_path.name,
        'dataset': dataset_key,
        'group_name': group_name,
        'sentence_count': len(sentences),
        'pair_count': len(pair_values),
        'edit_distance_mean': mean_value,
        'edit_distance_std': std_value,
        'edit_distance_min': min_value,
        'edit_distance_max': max_value,
    }


def analyze_file(json_path, dataset_arg, limit, save_dir):
    data = load_json(json_path)
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict in {json_path}, got {type(data)}')

    dataset_key = resolve_dataset_key(data, json_path, dataset_arg)
    groups = data[dataset_key]
    if not isinstance(groups, dict):
        raise TypeError(f'Expected dict at dataset key {dataset_key!r}, got {type(groups)}')

    print(f'json_path={json_path}')
    print(f'dataset={dataset_key}')
    print(f'limit={limit}')

    rows = []
    for group_name, sentences in groups.items():
        subset = sentences[:limit]
        if len(subset) < 2:
            print(f'[{group_name}] skipped: sentence_count={len(subset)}')
            continue
        rows.append(
            analyze_group(
                json_path=json_path,
                dataset_key=dataset_key,
                group_name=group_name,
                sentences=subset,
                save_dir=save_dir,
            )
        )
    return rows


def main():
    args = parse_args()
    json_paths = expand_inputs(args.inputs)
    if not json_paths:
        raise ValueError('No input JSON files matched the provided paths or glob patterns.')

    all_rows = []
    for index, json_path in enumerate(json_paths):
        if index > 0:
            print()
        all_rows.extend(
            analyze_file(
                json_path=json_path,
                dataset_arg=args.dataset,
                limit=args.limit,
                save_dir=args.save_dir,
            )
        )

    if args.summary_csv is not None:
        write_summary_csv(all_rows, args.summary_csv)
        print()
        print(f'summary_csv={args.summary_csv}')


if __name__ == '__main__':
    main()
