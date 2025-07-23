import argparse
import torch
import gc

class OnlineTensorStats:
    """
    Welford algorithm to calculate mean and std in stream.
    """
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None

    def update(self, x):
        x = x.detach().float().cpu()
        if self.mean is None:
            self.mean = torch.zeros_like(x)
            self.M2 = torch.zeros_like(x)

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean_std(self):
        if self.n < 2:
            std = torch.zeros_like(self.mean)
        else:
            std = torch.sqrt(self.M2 / (self.n - 1))
        return self.mean, std

def process_file(path, stats_dict):
    data = torch.load(path, map_location='cpu')
    for key in data:
        if key not in stats_dict:
            stats_dict[key] = []
        for idx, tensor in enumerate(data[key]):
            if idx == len(stats_dict[key]):
                stats_dict[key].append(OnlineTensorStats())
            stats_dict[key][idx].update(tensor)
    del data
    gc.collect()

def main(args):
    stats_dict = dict()

    with open(args.input_list, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]

    for path in file_paths:
        process_file(path, stats_dict)

    mean_result = {}
    std_result = {}

    for key, stats in stats_dict.items():
        mean_result[key] = []
        std_result[key] = []
        for s in stats:
            mean, std = s.get_mean_std()
            mean_result[key].append(mean)
            std_result[key].append(std)

    torch.save(mean_result, args.output_mean_file)
    torch.save(std_result, args.output_std_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Streamed tensor mean/std computation from .pt files.")
    parser.add_argument('--input-list', type=str, required=True, help='Text file containing list of .pt file paths')
    parser.add_argument('--output-mean-file', type=str, required=True, help='Output .pt file to save mean values')
    parser.add_argument('--output-std-file', type=str, required=True, help='Output .pt file to save std values')
    args = parser.parse_args()
    main(args)
