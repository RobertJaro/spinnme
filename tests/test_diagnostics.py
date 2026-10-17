import json
from unittest.mock import Mock, patch

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from spinnme.train.diagnostics import GPUActivityMonitor, training_diagnostics


class TinyTraining(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.01)


def test_profile_window_writes_reports_without_stopping_training(tmp_path):
    profiler, monitor = training_diagnostics(tmp_path)
    trainer = Trainer(accelerator='cpu', devices=1, max_steps=36,
                      logger=False, enable_checkpointing=False,
                      enable_model_summary=False, enable_progress_bar=False,
                      profiler=profiler, callbacks=[monitor])
    trainer.fit(TinyTraining(), DataLoader(TensorDataset(torch.ones(160, 4)), batch_size=4))
    assert trainer.global_step == 36
    summaries = list(tmp_path.glob('*.summary.json'))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary['train_dataloader_next']
    assert summary['backward']
    assert summary['batch_to_device']
    assert summary['optimizer_step']
    assert len(list(tmp_path.glob('*.trace.json'))) == 1
    assert monitor._process is None


def test_gpu_monitor_stops_and_only_launches_on_local_rank_zero(tmp_path):
    monitor = GPUActivityMonitor(tmp_path, batches=2)
    process = Mock()
    trainer = Mock(local_rank=1)
    with patch('spinnme.train.diagnostics.shutil.which', return_value='/usr/bin/nvidia-smi'), \
         patch('spinnme.train.diagnostics.torch.cuda.is_available', return_value=True), \
         patch('spinnme.train.diagnostics.subprocess.Popen', return_value=process) as launch:
        monitor.on_train_start(trainer, None)
        launch.assert_not_called()
        trainer.local_rank = 0
        monitor.on_train_start(trainer, None)
        launch.assert_called_once()
        monitor.on_train_batch_end(trainer, None, None, None, 0)
        process.terminate.assert_not_called()
        monitor.on_train_batch_end(trainer, None, None, None, 1)
        process.terminate.assert_called_once()
        assert monitor._stream is None
        monitor.teardown(trainer, None, 'fit')  # Cleanup is idempotent.
        process.terminate.assert_called_once()
