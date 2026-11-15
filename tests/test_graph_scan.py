"""The dyadic concept graph: training and generation agree, dense in the limit, causal, bounded memory."""
import math

import pytest
import torch

from acan.lm.graph_scan import GraphScanAttention, aggregate


def written(layer, h, loaded=0):
    """The layer one token at a time, as in generation; the first ``loaded`` tokens given at once."""
    b, length, d = h.shape
    q, raw, keys, v = layer.project(h)
    stream = layer.stream(h.dtype)
    steps = []
    if loaded:
        out, received = __import__('acan.lm.graph_scan', fromlist=['graph_scan']).graph_scan(
            q[:, :, :loaded], raw[:, :, :loaded], keys[:, :, :loaded], v[:, :, :loaded], layer.window, layer.bins,
            layer.head_dim**-.5, layer.inverse_frequency.to(h.dtype), details=True)
        stream.load(raw[:, :, :loaded], keys[:, :, :loaded], v[:, :, :loaded], received)
        steps = list(out.unbind(2))
    steps += [stream.step(q[:, :, t], raw[:, :, t], keys[:, :, t], v[:, :, t]) for t in range(loaded, length)]
    return layer.o_proj(torch.stack(steps, 2).transpose(1, 2).reshape(b, length, d)), stream


@pytest.mark.parametrize('length,window,bins', [(70, 8, 4), (200, 16, 4), (131, 4, 8), (64, 16, 2), (10, 16, 4), (33, 16, 4),
                                                (300, 16, 16), (150, 8, 6), (90, 5, 3), (40, 8, 1)])
def test_training_and_generation_agree(length, window, bins):
    torch.manual_seed(0)
    layer = GraphScanAttention(32, 2, window=window, bins=bins).double()
    h = torch.randn(2, length, 32, dtype=torch.double)
    whole = layer.attend(h)
    one_by_one, stream = written(layer, h)
    assert torch.allclose(whole, one_by_one, atol=1e-10)
    old = max(length-window, 0)
    assert stream.kept_vectors() <= window+bins*(max(old, 1).bit_length()+1)
    for loaded in (1, window, length//2, length-1):              # a prompt read at once, then token by token
        assert torch.allclose(whole, written(layer, h, loaded)[0], atol=1e-10)


def test_enough_concepts_is_dense_attention():
    torch.manual_seed(0)
    layer = GraphScanAttention(32, 2, window=8, bins=64).double()
    h = torch.randn(1, 40, 32, dtype=torch.double)
    q, raw, keys, v = layer.project(h)
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


def test_copies_are_added_together_first_and_read_tokens_stay_apart():
    """Two copies of one concept cost nothing; among unread concepts a read one is the last to be added."""
    centre = torch.tensor([[0., 0.], [0., 0.], [5., 0.], [5.2, 0.], [9., 9.]], dtype=torch.double)
    count = torch.ones(5, dtype=torch.double)
    square = (centre*centre).sum(-1)
    assert aggregate(count, centre, square, 4).tolist() == [0, 0, 1, 2, 3]
    assert aggregate(count, centre, square, 3).tolist() == [0, 0, 1, 1, 2]
    reads = torch.tensor([0., 0., 3., 3., 0.], dtype=torch.double)
    assert aggregate(count, centre, square, 3, reads).tolist()[2:4] == [1, 2]      # the read pair is kept apart
