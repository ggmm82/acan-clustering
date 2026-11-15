"""The dyadic concept graph: training and generation agree, dense in the limit, causal, bounded memory."""
import math

import pytest
import torch

from acan.lm.graph_scan import GraphScanAttention


def written(layer, h):
    """The layer one token at a time, as in generation."""
    b, length, d = h.shape
    q, raw, keys, v, energy = layer.project(h)
    stream = layer.stream()
    stream.directions, stream.frequency = layer.directions.to(h.dtype), layer.inverse_frequency.to(h.dtype)
    steps = [stream.step(q[:, :, t], raw[:, :, t], keys[:, :, t], v[:, :, t], None if energy is None else energy[:, :, t])
             for t in range(length)]
    return layer.o_proj(torch.stack(steps, 2).transpose(1, 2).reshape(b, length, d)), stream


@pytest.mark.parametrize('length,window,bins,lloyd,reader', [
    (70, 8, 4, 1, False), (200, 16, 4, 1, False), (131, 4, 8, 1, False), (64, 16, 2, 0, False), (10, 16, 4, 1, False),
    (33, 16, 4, 1, False), (300, 16, 16, 1, False), (150, 8, 8, 'aggregate', False), (200, 16, 8, 1, True)])
def test_training_and_generation_agree(length, window, bins, lloyd, reader):
    torch.manual_seed(0)
    layer = GraphScanAttention(32, 2, window=window, bins=bins, lloyd=lloyd, reader=reader).double()
    h = torch.randn(2, length, 32, dtype=torch.double)
    one_by_one, stream = written(layer, h)
    assert torch.allclose(layer.attend(h), one_by_one, atol=1e-10)
    old = max(length-window, 0)
    assert stream.kept_vectors() <= window+bins*(max(old, 1).bit_length()+1)


def test_enough_concepts_is_dense_attention():
    torch.manual_seed(0)
    layer = GraphScanAttention(32, 2, window=8, bins=64).double()
    h = torch.randn(1, 40, 32, dtype=torch.double)
    q, raw, keys, v, _ = layer.project(h)
    place = torch.arange(40)
    logits = (q@keys.transpose(-1, -2)*layer.head_dim**-.5).masked_fill(place[None] > place[:, None], -torch.inf)
    dense = layer.o_proj((logits.softmax(-1)@v).transpose(1, 2).reshape(1, 40, 32))
    assert torch.allclose(layer.attend(h), dense, atol=1e-10)


def test_no_position_reads_the_future_and_old_tokens_are_read():
    torch.manual_seed(0)
    layer = GraphScanAttention(32, 2, window=8, bins=4).double()
    h = torch.randn(1, 90, 32, dtype=torch.double, requires_grad=True)
    layer.attend(h)[0, 50].sum().backward()
    assert float(h.grad[0, 51:].abs().max()) == 0.
    assert float(h.grad[0, :10].abs().max()) > 0.


def test_memory_grows_with_the_logarithm_of_the_context():
    torch.manual_seed(0)
    layer = GraphScanAttention(16, 2, window=8, bins=4)
    with torch.no_grad():
        _, stream = written(layer, torch.randn(1, 2000, 16))
    assert stream.kept_vectors() <= 8+4*(math.floor(math.log2(2000))+1)
