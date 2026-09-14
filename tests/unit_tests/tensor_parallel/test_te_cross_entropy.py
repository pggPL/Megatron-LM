# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.models.backends import get_backend_from_config
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("capturable", [False, True])
def test_te_cross_entropy_keeps_legacy_call_signature(monkeypatch, capturable):
    monkeypatch.setattr(te_extension, "_TE_SUPPORTS_CE_OVERWRITE_INPUT", False)
    monkeypatch.setattr(te_extension, "_TE_SUPPORTS_CG_CAPTURABLE", capturable)
    kernel = Mock()
    monkeypatch.setattr(te_extension, "parallel_cross_entropy", kernel)
    logits, labels, group = object(), object(), object()

    te_extension.te_parallel_cross_entropy(logits, labels, group, capturable)

    args = (logits, labels, 0.0, False, group)
    if capturable:
        args += (-100, True)
    kernel.assert_called_once_with(*args)


def test_te_cross_entropy_rejects_unsupported_input_reuse(monkeypatch):
    monkeypatch.setattr(te_extension, "_TE_SUPPORTS_CE_OVERWRITE_INPUT", False)
    kernel = Mock()
    monkeypatch.setattr(te_extension, "parallel_cross_entropy", kernel)
    with pytest.raises(RuntimeError, match="TransformerEngine >= 2.19.0"):
        te_extension.te_parallel_cross_entropy(None, None, None, overwrite_input=True)
    kernel.assert_not_called()


@pytest.mark.skipif(not is_te_min_version("2.19.0"), reason="requires TE input reuse")
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("overwrite_input", [False, True])
def test_te_language_model_loss_and_projection_gradients(tp_size, dtype, overwrite_input):
    if Utils.world_size < tp_size:
        pytest.skip("requires multiple GPUs")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = torch.distributed.get_rank(group)
        torch.manual_seed(1234)
        inp = torch.randn(16, 2, 32, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(256, 32, device="cuda", dtype=dtype, requires_grad=True)
        labels = torch.randint(0, 256, (2, 16), device="cuda")
        labels[0, 0] = -100
        upstream = torch.randn(2, 16, device="cuda")
        expected_logits = torch.nn.functional.linear(inp, weight)
        expected = (
            torch.nn.functional.cross_entropy(
                expected_logits.float().reshape(-1, 256),
                labels.transpose(0, 1).reshape(-1),
                reduction="none",
            )
            .view(16, 2)
            .transpose(0, 1)
        )
        expected_grads = torch.autograd.grad(expected, (inp, weight), upstream)

        local_weight = weight.detach().chunk(tp_size)[tp_rank].clone().requires_grad_()
        inp = inp.detach().clone().requires_grad_()
        logits = torch.nn.functional.linear(inp, local_weight)
        original_logits = logits.detach().clone()
        config = SimpleNamespace(
            transformer_impl="transformer_engine",
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="te",
            cross_entropy_fusion_overwrite_input=overwrite_input,
        )
        module = SimpleNamespace(
            tp_group=group,
            vocab_parallel_cross_entropy=get_backend_from_config(
                config
            ).vocab_parallel_cross_entropy(),
        )
        loss = LanguageModule.compute_language_model_loss(module, labels, logits)
        torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(loss, (inp, local_weight), upstream)
        torch.distributed.all_reduce(actual_grads[0], group=group)

        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(loss, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            actual_grads[0], expected_grads[0], rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            actual_grads[1],
            expected_grads[1].chunk(tp_size)[tp_rank],
            rtol=tolerance,
            atol=tolerance,
        )
        if overwrite_input:
            assert not torch.equal(logits, original_logits)
        else:
            torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
