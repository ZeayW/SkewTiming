import sys
import unittest
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

from training_utils import (
    ModelEMA,
    filter_endpoint_rows,
    normalize_endpoint_correlation,
    replace_case_50_with_case_0,
    supervision_loss_weights,
    valid_endpoint_mask,
)
from options import get_options


class TrainingUtilsTest(unittest.TestCase):
    def test_endpoint_correlation_normalization_removes_mass_scale(self):
        weights = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0]])

        normalized = normalize_endpoint_correlation(weights)

        self.assertTrue(torch.allclose(normalized[0], torch.tensor([0.25, 0.5, 0.25])))
        self.assertEqual(normalized[1].tolist(), [0.0, 0.0, 0.0])

    def test_case_50_is_replaced_before_dataset_case_slicing(self):
        cases = [('case-{}'.format(index),) for index in range(100)]
        graph_info = {
            'delay-label_pairs': cases,
            'case_indices': list(range(100)),
        }

        replace_case_50_with_case_0(graph_info)

        self.assertIs(graph_info['delay-label_pairs'][50], graph_info['delay-label_pairs'][0])
        self.assertEqual(graph_info['case_indices'][50], 50)

    def test_case_50_replacement_allows_short_datasets(self):
        graph_info = {'delay-label_pairs': [('case-0',)]}

        replace_case_50_with_case_0(graph_info)

        self.assertEqual(graph_info['delay-label_pairs'], [('case-0',)])

    def test_case_50_replacement_uses_case_ids_for_sparse_datasets(self):
        graph_info = {
            'delay-label_pairs': [('case-{}'.format(index),) for index in range(88)],
            'case_indices': [0] + list(range(7, 94)),
        }

        replace_case_50_with_case_0(graph_info)

        case_50_position = graph_info['case_indices'].index(50)
        self.assertIs(
            graph_info['delay-label_pairs'][case_50_position],
            graph_info['delay-label_pairs'][0],
        )
        self.assertEqual(graph_info['delay-label_pairs'][50], ('case-50',))
        self.assertEqual(graph_info['case_indices'][50], 56)

    def test_case_50_replacement_falls_back_to_legacy_positions(self):
        graph_info = {
            'delay-label_pairs': [('case-{}'.format(index),) for index in range(51)],
        }

        replace_case_50_with_case_0(graph_info)

        self.assertIs(graph_info['delay-label_pairs'][50], graph_info['delay-label_pairs'][0])

    def test_optimization_subdefaults_match_canonical_training_config(self):
        options = get_options([])

        self.assertEqual(options.lr_scheduler_patience, 3)
        self.assertEqual(options.lr_scheduler_factor, 0.25)
        self.assertEqual(options.ema_start_epoch, 5)
        self.assertFalse(options.lr_scheduler)
        self.assertEqual(options.ema_decay, 0.0)
        self.assertEqual(options.fse_aggregation, 'raw_sum')
        self.assertEqual(options.cpe_depth_encoding, 'absolute')

    def test_model_ema_updates_and_restores_raw_parameters(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, decay=0.75)

        with torch.no_grad():
            model.weight.fill_(4.0)
        ema.update(model)
        with torch.no_grad():
            model.weight.fill_(8.0)
        ema.update(model)

        with ema.average_parameters(model):
            expected = torch.full_like(model.weight, 2.75 / (1.0 - 0.75 ** 2))
            self.assertTrue(torch.allclose(model.weight, expected))

        self.assertTrue(torch.equal(model.weight, torch.full_like(model.weight, 8.0)))

    def test_model_ema_without_updates_keeps_raw_parameters(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3.0)
        ema = ModelEMA(model, decay=0.999)

        with ema.average_parameters(model):
            self.assertTrue(torch.equal(model.weight, torch.full_like(model.weight, 3.0)))

        self.assertEqual(ema.num_updates, 0)

    def test_smooth_supervision_matches_three_epoch_average(self):
        alternating = [supervision_loss_weights(epoch % 3 == 0, False) for epoch in range(3)]
        mean_ccal = sum(weights[0] for weights in alternating) / 3.0
        mean_residual = sum(weights[1] for weights in alternating) / 3.0

        smooth_ccal, smooth_residual = supervision_loss_weights(True, True)
        self.assertAlmostEqual(smooth_ccal, mean_ccal)
        self.assertAlmostEqual(smooth_residual, mean_residual)

    def test_missing_endpoint_labels_are_filtered_consistently(self):
        labels = torch.tensor([[4.0], [-1.0], [0.0], [7.0]])
        predictions = torch.tensor([[4.5], [99.0], [0.5], [6.5]])
        mask = valid_endpoint_mask(labels)

        self.assertEqual(mask.tolist(), [True, False, True, True])
        self.assertEqual(labels[mask].reshape(-1).tolist(), [4.0, 0.0, 7.0])
        self.assertEqual(
            filter_endpoint_rows(predictions, mask).reshape(-1).tolist(),
            [4.5, 0.5, 6.5],
        )

    def test_non_endpoint_sentinel_is_not_indexed_by_endpoint_mask(self):
        sentinel = torch.tensor([0.0])
        mask = torch.tensor([True, False, True])
        self.assertIs(filter_endpoint_rows(sentinel, mask), sentinel)

    def test_missing_endpoint_has_no_supervised_gradient(self):
        labels = torch.tensor([[4.0], [-1.0], [7.0]])
        predictions = torch.tensor(
            [[3.0], [100.0], [9.0]], requires_grad=True
        )
        mask = valid_endpoint_mask(labels)
        loss = torch.nn.functional.l1_loss(predictions[mask], labels[mask])
        loss.backward()

        self.assertEqual(predictions.grad[1].item(), 0.0)
        self.assertNotEqual(predictions.grad[0].item(), 0.0)
        self.assertNotEqual(predictions.grad[2].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
