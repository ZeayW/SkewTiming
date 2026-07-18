import pathlib
import sys
import unittest

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from dag_ops import dag_correlation, segment_softmax_aggregate


def reference_correlation(initial_hp, edge_weight, topo_edges):
    hp = initial_hp
    for src, dst, eid in topo_edges:
        msg = hp[src] * edge_weight[eid]
        hp_next = hp.clone()
        hp_next[dst] = 0
        hp = hp_next.index_add(0, dst, msg)
    return hp


class DAGCorrelationTest(unittest.TestCase):
    def setUp(self):
        self.seed = torch.zeros((5, 2), dtype=torch.double)
        self.seed[0, 0] = 1
        self.seed[1, 1] = 1
        self.topo_edges = [
            (
                torch.tensor([0, 1, 0]),
                torch.tensor([2, 2, 3]),
                torch.tensor([0, 1, 2]),
            ),
            (
                torch.tensor([2, 3]),
                torch.tensor([4, 4]),
                torch.tensor([3, 4]),
            ),
        ]

    def test_output_and_weight_gradient_match_reference(self):
        weight_ref = torch.tensor([[0.4], [0.6], [0.8], [0.3], [0.7]],
                                  dtype=torch.double, requires_grad=True)
        weight_custom = weight_ref.detach().clone().requires_grad_(True)
        projection = torch.arange(10, dtype=torch.double).reshape(5, 2) / 10

        output_ref = reference_correlation(self.seed, weight_ref, self.topo_edges)
        output_custom = dag_correlation(self.seed, weight_custom, self.topo_edges)
        grad_ref, = torch.autograd.grad(torch.sum(output_ref * projection), weight_ref)
        grad_custom, = torch.autograd.grad(torch.sum(output_custom * projection), weight_custom)

        torch.testing.assert_close(output_custom, output_ref, rtol=0, atol=0)
        torch.testing.assert_close(grad_custom, grad_ref, rtol=1e-12, atol=1e-12)

    def test_gradcheck(self):
        weight = torch.rand((5, 1), dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: dag_correlation(self.seed, value, self.topo_edges),
            (weight,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        ))


    def test_disconnected_graphs_can_reuse_local_endpoint_columns(self):
        topo_edges = [
            (
                torch.tensor([0, 1, 3]),
                torch.tensor([2, 2, 4]),
                torch.tensor([0, 1, 2]),
            ),
            (
                torch.tensor([4]),
                torch.tensor([5]),
                torch.tensor([3]),
            ),
        ]
        global_seed = torch.zeros((6, 3), dtype=torch.double)
        global_seed[0, 0] = 1
        global_seed[1, 1] = 1
        global_seed[3, 2] = 1
        local_seed = torch.zeros((6, 2), dtype=torch.double)
        local_seed[0, 0] = 1
        local_seed[1, 1] = 1
        local_seed[3, 0] = 1
        global_weight = torch.rand((4, 1), dtype=torch.double, requires_grad=True)
        local_weight = global_weight.detach().clone().requires_grad_(True)

        global_output = dag_correlation(global_seed, global_weight, topo_edges)
        local_output = dag_correlation(local_seed, local_weight, topo_edges)
        torch.testing.assert_close(local_output[:3, :2], global_output[:3, :2], rtol=0, atol=0)
        torch.testing.assert_close(local_output[3:, :1], global_output[3:, 2:3], rtol=0, atol=0)

        global_projection = torch.randn_like(global_output)
        local_projection = torch.zeros_like(local_output)
        local_projection[:3, :2] = global_projection[:3, :2]
        local_projection[3:, :1] = global_projection[3:, 2:3]
        global_grad, = torch.autograd.grad(
            torch.sum(global_output * global_projection), global_weight)
        local_grad, = torch.autograd.grad(
            torch.sum(local_output * local_projection), local_weight)
        torch.testing.assert_close(local_grad, global_grad, rtol=1e-12, atol=1e-12)


class SegmentSoftmaxTest(unittest.TestCase):
    @staticmethod
    def reference(z, score, dst_pos, num_dst):
        aggregated = []
        alpha = torch.zeros_like(score)
        for dst in range(num_dst):
            mask = dst_pos == dst
            dst_alpha = torch.softmax(score[mask], dim=0)
            alpha[mask] = dst_alpha
            aggregated.append(torch.sum(dst_alpha * z[mask], dim=0))
        return torch.stack(aggregated), alpha

    def test_output_and_gradients_match_reference(self):
        dst_pos = torch.tensor([0, 1, 0, 2, 1, 2])
        z_ref = torch.randn((6, 4), dtype=torch.double, requires_grad=True)
        score_ref = torch.randn((6, 1), dtype=torch.double, requires_grad=True)
        z_scatter = z_ref.detach().clone().requires_grad_(True)
        score_scatter = score_ref.detach().clone().requires_grad_(True)
        projection = torch.arange(12, dtype=torch.double).reshape(3, 4) / 10

        output_ref, alpha_ref = self.reference(z_ref, score_ref, dst_pos, 3)
        output_scatter, alpha_scatter = segment_softmax_aggregate(
            z_scatter, score_scatter, dst_pos, 3)
        grads_ref = torch.autograd.grad(torch.sum(output_ref * projection), (z_ref, score_ref))
        grads_scatter = torch.autograd.grad(
            torch.sum(output_scatter * projection), (z_scatter, score_scatter))

        torch.testing.assert_close(output_scatter, output_ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(alpha_scatter, alpha_ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grads_scatter[0], grads_ref[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grads_scatter[1], grads_ref[1], rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
