# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from wy_snapshot_reference import metrics, reference


class WYReferenceTest(unittest.TestCase):
    def test_chunk_reconstruction_matches_independent_token_recurrence(self):
        torch.manual_seed(17)
        q, k = [torch.nn.functional.normalize(torch.randn(1, 8, 1, 4), dim=-1).half() for _ in range(2)]
        v = torch.randn(1, 8, 2, 3).half()
        g = -torch.rand(1, 8, 2) * .3
        beta = torch.rand(1, 8, 2).half()
        out = reference((q, k, v, g, beta), 4)
        key = k.float().repeat_interleave(2, dim=2).transpose(1, 2)
        # At each chunk, apply W/U to arbitrary incoming state; compare final state
        # to direct token recurrence. This tests signs, beta, head expansion and resets.
        state = torch.randn(1, 2, 4, 3)
        recurrent = state.clone()
        for start in (0, 4):
            cumulative = out[4][..., start:start+4]
            residual = out[3][..., start:start+4, :] - out[2][..., start:start+4, :] @ state
            decay_to_end = (cumulative[..., -1:] - cumulative).exp()
            state = state * cumulative[..., -1:, None].exp() + key[..., start:start+4, :].transpose(-1, -2) @ (residual * decay_to_end[..., None])
            for t in range(start, start+4):
                kt = key[..., t, :]
                recurrent = recurrent * g[:, t, :, None, None].exp()
                update = (v[:, t].float() - (kt[..., None] * recurrent).sum(-2)) * beta[:, t, :, None].float()
                recurrent = recurrent + kt[..., None] * update[..., None, :]
            torch.testing.assert_close(state, recurrent, atol=2e-6, rtol=2e-5)

    def test_independent_row_update_matches_triangular_solve(self):
        torch.manual_seed(19)
        inputs = [torch.randn(1, 8, 1, 4).half() for _ in range(2)]
        inputs += [torch.randn(1, 8, 2, 3).half(), -torch.rand(1, 8, 2), torch.rand(1, 8, 2).half()]
        q, k, v, g, beta = inputs
        key = k.float().repeat_interleave(2, 2).transpose(1, 2).reshape(1, 2, 2, 4, 4)
        gate = g.transpose(1, 2).reshape(1, 2, 2, 4).cumsum(-1)
        weight = beta.float().transpose(1, 2).reshape(1, 2, 2, 4)
        attn = -((key * weight[..., None]) @ key.transpose(-1, -2) * (gate[..., None] - gate[..., None, :]).tril().exp().tril()).tril(-1)
        for i in range(1, 4):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
        attn += torch.eye(4)
        actual_u = attn @ (v.float().transpose(1, 2).reshape(1, 2, 2, 4, 3) * weight[..., None])
        actual_w = attn @ (key * weight[..., None] * gate.exp()[..., None])
        expected = reference(inputs, 4)
        torch.testing.assert_close(actual_w.reshape_as(expected[2]), expected[2], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_u.reshape_as(expected[3]), expected[3], atol=2e-5, rtol=2e-5)

    def test_gate_rejects_localized_error_even_when_rmse_small(self):
        expected = [torch.ones(1, 1, 64, 2) for _ in range(4)] + [torch.ones(1, 1, 64)]
        actual = [x.clone() for x in expected]
        actual[2][0, 0, 0, 0] += .1
        result = metrics(actual, expected)
        self.assertFalse(result[2]['passed'])
        self.assertTrue(result[0]['passed'])


if __name__ == '__main__':
    unittest.main()
