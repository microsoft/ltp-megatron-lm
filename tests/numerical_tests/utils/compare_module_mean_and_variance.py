import argparse
import torch
import json

def load_stats(path):
    return torch.load(path, map_location='cpu')

def relative_error_max(ref, test, eps=1e-5):
    denom = torch.maximum(torch.abs(ref), torch.abs(test))
    denom = torch.clamp(denom, min=eps)
    return torch.max(torch.abs(ref - test) / denom).item()

def ratio_max(ref, test, eps=1e-5):
    if (ref <= eps).all() and (test <= eps).all():
        return 1.0
    denom = torch.clamp(torch.abs(ref), min=eps)
    return torch.max(torch.abs(test) / denom).item()

def compare_stats(stats_ref, stats_test):
    result = {}

    for key in stats_ref:
        list_ref = stats_ref[key]
        list_test = stats_test[key]

        if len(list_ref) == 0:
            max_mean_err = [0.0]
            max_std_ratio = [0.0]
            mean_cos_similarity = [0.0]
            std_cos_similarity = [0.0]
        else:
            max_mean_err = [
                relative_error_max(
                    ref['mean'].view(-1).float(),
                    test['mean'].view(-1).float(),
                )
                for ref, test in zip(list_ref, list_test)
            ]
            max_std_ratio = [
                ratio_max(
                    ref['std'].view(-1).float(),
                    test['std'].view(-1).float(),
                )
                for ref, test in zip(list_ref, list_test)
            ]
            mean_cos_similarity = [
                torch.nn.functional.cosine_similarity(
                    ref['mean'].view(-1).float(),
                    test['mean'].view(-1).float(),
                    dim=0
                ).item()
                for ref, test in zip(list_ref, list_test)
            ]
            std_cos_similarity = [
                torch.nn.functional.cosine_similarity(
                    ref['std'].view(-1).float(),
                    test['std'].view(-1).float(),
                    dim=0
                ).item()
                for ref, test in zip(list_ref, list_test)
            ]

        result[key] = {
            'max_mean_err': max_mean_err,
            'max_std_ratio': max_std_ratio,
            'mean_cos_similarity': mean_cos_similarity,
            'std_cos_similarity': std_cos_similarity,
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
