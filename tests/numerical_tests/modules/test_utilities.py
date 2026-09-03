import os

from tests.unit_tests.test_utilities import Utils as UtilsWithNvtePop


class Utils(UtilsWithNvtePop):

    _nvte_keys = ['NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN']
    _nvte_backup = {}

    @staticmethod
    def _save_nvte_envs():
        Utils._nvte_backup = {k: os.environ.get(k) for k in Utils._nvte_keys}

    @staticmethod
    def _restore_nvte_envs():
        for k, v in Utils._nvte_backup.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    @staticmethod
    def initialize_distributed():
        Utils._save_nvte_envs()
        try:
            UtilsWithNvtePop.initialize_distributed()
        finally:
            Utils._restore_nvte_envs()

    @staticmethod
    def destroy_model_parallel():
        Utils._save_nvte_envs()
        try:
            UtilsWithNvtePop.destroy_model_parallel()
        finally:
            Utils._restore_nvte_envs()

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        Utils._save_nvte_envs()
        try:
            UtilsWithNvtePop.initialize_model_parallel(
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                **kwargs,
            )
        finally:
            Utils._restore_nvte_envs()
