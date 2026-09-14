# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.training.arguments import parse_args, validate_args


def test_te_cross_entropy_loss_fusion_warns_in_model_parallel_config():
    with pytest.warns(UserWarning, match="known stability issues"):
        config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'te'


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'


@pytest.mark.parametrize("fusion,impl", [(False, "native"), (False, "te"), (True, "native")])
def test_cross_entropy_input_reuse_requires_te_fusion(fusion, impl):
    with pytest.raises(ValueError, match="requires TE cross-entropy fusion"):
        ModelParallelConfig(
            cross_entropy_loss_fusion=fusion,
            cross_entropy_fusion_impl=impl,
            cross_entropy_fusion_overwrite_input=True,
        )


@pytest.mark.parametrize("overwrite_input", [False, True])
def test_te_cross_entropy_loss_fusion_is_disabled_by_training_args(monkeypatch, overwrite_input):
    monkeypatch.setattr(sys, 'argv', ['test_model_parallel_config.py'])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    # Let validate_args derive a global batch size that is valid for the
    # active data-parallel size in distributed unit-test jobs.
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024
    args.cross_entropy_loss_fusion = True
    args.cross_entropy_fusion_impl = 'te'
    args.cross_entropy_fusion_overwrite_input = overwrite_input

    with pytest.raises(AssertionError, match="Transformer Engine cross entropy loss fusion"):
        validate_args(args)
