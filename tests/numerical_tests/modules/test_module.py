import os
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from tests.numerical_tests.modules.test_utilities import Utils


class TestModule:
    """
    Test general modules
    """

    def setup_method(self, method):
        self.setup_distributed()
        self.setup_parallelism()
        self.setup_random_seed()

    def setup_distributed(self):
        Utils.initialize_distributed()

    def setup_parallelism(self):
        Utils.initialize_model_parallel()

    def setup_random_seed(self):
        seed = 42
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def setup_model_and_optimizer(self, config, model):
        model = model.cuda()
        if config.bf16:
            model = Float16Module(config, model)
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        model = DistributedDataParallel(config, ddp_config, model)
        optimizer_config = OptimizerConfig(
            optimizer='adam',
            bf16=config.bf16,
            use_distributed_optimizer=True,
            lr=1e-3,
            clip_grad=0.0
        )
        optimizer = get_megatron_optimizer(optimizer_config, [model])
        return model, optimizer

    def save_output(self, inputs, outputs, params, grads, step, request):
        output_dict = {
            'inputs': inputs,
            'outputs': outputs,
            'params': params,
            'grads': grads,
        }
        result_dir = request.config.getoption('--result-dir')
        file_tag = request.node.nodeid[request.node.nodeid.find(self.__class__.__name__):]
        file_tag = file_tag.replace('::', '-').replace('[', '-').replace(']', '')
        rank = torch.distributed.get_rank()
        file_name = f'{file_tag}-rank-{rank}-step-{step}.pt'
        torch.save(output_dict, os.path.join(result_dir, file_name))
