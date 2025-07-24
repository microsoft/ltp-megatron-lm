import argparse
import torch
import json

def load_stats(path):
    return torch.load(path, map_location='cpu')

def compare_stats(stats_ref, stats_test):
    result = {}

    for key in stats_ref:
        list_ref = stats_ref[key]
        list_test = stats_test[key]

        cos_similarity = [
            torch.nn.functional.cosine_similarity(
                ref.view(-1).double(),
                test.view(-1).double(),
                dim=0
            ).item()
            for ref, test in zip(list_ref, list_test)
        ]

        result[key] = {
            'cos_similarity': cos_similarity,
        }

    return result

def main(args):
    stats_ref = load_stats(args.stats_ref)
    stats_test = load_stats(args.stats_test)

    comparison_result = compare_stats(stats_ref, stats_test)

    with open(args.output_file, 'w') as f:
        json.dump(comparison_result, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two tensor stats .pt files and compute max relative errors.")
    parser.add_argument('--stats-ref', type=str, required=True, help='Path to stats .pt file from reference platform')
    parser.add_argument('--stats-test', type=str, required=True, help='Path to stats .pt file from test platform')
    parser.add_argument('--output-file', type=str, required=True, help='Path to save the max relative error comparison result (JSON)')
    args = parser.parse_args()
    main(args)
