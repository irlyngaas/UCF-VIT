import tempfile
import unittest
from pathlib import Path

from plot_training_losses import collect_records, parse_log


class PlotTrainingLossesTest(unittest.TestCase):
    def test_parses_dynamic_modalities_and_rank_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'run.out')
            path.write_text(
                "[rank3]: epoch: 7 epoch_loss tensor(0.25, device='cuda:0')\n"
                "epoch: 7 modality_losses (complete) "
                "{'xray[xct]': 0.3, 'future[mri+pet]': 0.2}\n"
                "epoch: 8 modality_losses (partial) {'xray[xct]': 9.0}\n"
            )
            records = parse_log(path)

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].rank, 3)
        self.assertEqual(
            {record.modality for record in records[1:]},
            {'xray[xct]', 'future[mri+pet]'},
        )

    def test_latest_replayed_epoch_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, 'first.out')
            second = Path(directory, 'second.out')
            first.write_text(
                'Starting epoch  4\n'
                'epoch: 4 epoch_loss tensor(0.8)\n'
                'Starting epoch  5\n'
                'epoch: 5 epoch_loss tensor(0.9)\n'
            )
            second.write_text(
                'Starting epoch  4\n'
                'epoch: 4 epoch_loss tensor(0.5)\n'
            )
            records = collect_records([first, second])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].loss, 0.5)


if __name__ == '__main__':
    unittest.main()
