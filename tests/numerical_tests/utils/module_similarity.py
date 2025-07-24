import argparse
import torch
import json

def load_stats(path):
    return torch.load(path, map_location='cpu')

def compare_stats(stats_a, stats_b):
    result = {}

    for key in stats_a:
        list_a = stats_a[key]
        list_b = stats_b[key]

        cosine_similarity = [
            torch.nn.functional.cosine_similarity(
                a.view(-1).double(),
                b.view(-1).double(),
                dim=0
            ).item()
            for a, b in zip(list_a, list_b)
        ]

        result[key] = {
            'cosine_similarity': cosine_similarity,
        }

    return result

def main(args):
    stats_a = load_stats(args.stats_a)
    stats_b = load_stats(args.stats_b)

    comparison_result = compare_stats(stats_a, stats_b)

    with open(args.output_file, 'w') as f:
        json.dump(comparison_result, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two tensor stats .pt files and compute max relative errors.")
    parser.add_argument('--stats-a', type=str, required=True, help='Path to stats .pt file from setting A')
    parser.add_argument('--stats-b', type=str, required=True, help='Path to stats .pt file from setting B')
    parser.add_argument('--output-file', type=str, required=True, help='Path to save the cosine similarity result (JSON)')
    args = parser.parse_args()
    main(args)
