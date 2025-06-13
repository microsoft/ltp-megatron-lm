import megatron
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig


def set_manual_pipeline_split_patch(args):
    """
    Monkey-patch note:
    - The original function will be replaced at runtime by this implementation.

    """

    # patch TransformerConfig.__post_init__ to add validation
    TransformerConfig.decoder_pipeline_manual_split_list = (
        args.decoder_pipeline_manual_split_list
    )
    TransformerConfig.num_layers_per_virtual_pipeline_stage = (
        args.num_layers_per_virtual_pipeline_stage
    )

    def validate_manual_split(func):
        def wrapper(self, *args, **kwargs):

            if (
                self.num_layers_per_virtual_pipeline_stage is not None
                or self.num_layers_in_first_pipeline_stage is not None
                or self.num_layers_in_last_pipeline_stage is not None
                or self.account_for_embedding_in_pipeline_split
                or self.account_for_loss_in_pipeline_split
            ):
                raise ValueError(
                    "decoder_pipeline_manual_split_list is not compatible "
                    "with num_layers_per_virtual_pipeline_stage/"
                    "decoder_first_pipeline_num_layers/"
                    "decoder_last_pipeline_num_layers/"
                    "account_for_embedding_in_pipeline_split/"
                    "account_for_loss_in_pipeline_split yet"
                )

            num_layers = self.num_layers
            pp_size = self.pipeline_model_parallel_size
            vp_size = self.virtual_pipeline_model_parallel_size
            pp_split = self.decoder_pipeline_manual_split_list

            if pp_size <= 1:
                raise ValueError(
                    f"pipeline_model_parallel_size={pp_size} should be larger "
                    f"than 1 when decoder_pipeline_manual_split_list is used"
                )

            if not isinstance(pp_split, list):
                raise ValueError(
                    f"type of decoder_pipeline_manual_split_list={pp_split} "
                    f"is {type(pp_split)} and should be a list")

            split_size = pp_size if vp_size is None else pp_size * vp_size
            if len(pp_split) != split_size:
                raise ValueError(
                    f"the size of decoder_pipeline_manual_split_list="
                    f"{pp_split} should be {split_size} "
                    f"given pipeline_model_parallel_size={pp_size} and "
                    f"virtual_pipeline_model_parallel_size={vp_size}"
                )

            if not all(x > 0 for x in pp_split):
                raise ValueError(
                    f"layer numbers in decoder_pipeline_manual_split_list"
                    f"={pp_split} should all be larger than 0"
                )

            if sum(pp_split) != num_layers:
                raise ValueError(
                    f"the sum of decoder_pipeline_manual_split_list="
                    f"{pp_split} is {sum(pp_split)} and "
                    f"should be equal to num_layers={num_layers}"
                )
            
            func(self, *args, **kwargs)
        return wrapper

    TransformerConfig.__post_init__ = validate_manual_split(
        TransformerConfig.__post_init__
    )

    # patch get_num_layers_to_build
    def get_num_layers_to_build_patch(config):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vp_size = config.virtual_pipeline_model_parallel_size
        vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        pp_idx = pp_rank if vp_size is None else pp_rank * vp_size + vp_rank
        num_layers_to_build = config.decoder_pipeline_manual_split_list[pp_idx]
        return num_layers_to_build

    megatron.core.transformer.transformer_block.get_num_layers_to_build = (
        get_num_layers_to_build_patch
    )
    megatron.core.models.gpt.gpt_layer_specs.get_num_layers_to_build = (
        get_num_layers_to_build_patch
    )

    # patch get_transformer_layer_offset
    def get_transformer_layer_offset_patch(config):
        pp_size = config.pipeline_model_parallel_size
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vp_size = config.virtual_pipeline_model_parallel_size
        vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()

        if not parallel_state.is_inside_encoder():
            pp_decoder_start = (
                parallel_state.get_pipeline_model_parallel_decoder_start()
            )
            if pp_decoder_start is not None:
                pp_rank = pp_rank - pp_decoder_start

        offset = 0
        if vp_rank is not None:
            for _ in range(vp_rank):
                for pp_idx in range(pp_size):
                    offset += config.decoder_pipeline_manual_split_list[
                        pp_idx * vp_size
                    ]
            for pp_idx in range(pp_rank):
                offset += config.decoder_pipeline_manual_split_list[
                    pp_idx * vp_size + vp_rank
                ]
        else:
            offset = sum(config.decoder_pipeline_manual_split_list[:pp_rank])
        return offset

    megatron.core.transformer.transformer_layer.get_transformer_layer_offset = (
        get_transformer_layer_offset_patch
    )
    megatron.core.transformer.transformer_block.get_transformer_layer_offset = (
        get_transformer_layer_offset_patch
    )
    megatron.core.models.gpt.gpt_layer_specs.get_transformer_layer_offset = (
        get_transformer_layer_offset_patch
    )
