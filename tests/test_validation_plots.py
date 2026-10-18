"""Render validation figures through the trainer logger, without a W&B service."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from _training_helpers import _module, _physical_batch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from spinnme.train.me_spherical_module import MESphericalModule


def test_profile_comparison_statistics_and_polarization_extrema(monkeypatch):
    module = _module()
    truth = np.ones((2, 3, 4, 5))
    truth[0, 0, 1:] = 0
    truth[1, 1, 1:] = -4  # Absolute polarization, including negative signals.
    truth[0, 2] = np.nan
    prediction = truth + 0.25
    prediction[1, 2] = np.inf  # Exclude nonfinite predictions from paired statistics.
    figures = []
    monkeypatch.setattr(module, '_log_validation_image', lambda fig, name: figures.append(fig))
    module.plot_profile(prediction, truth)
    fig, = figures
    valid = np.isfinite(truth).all(axis=(-2, -1)) & np.isfinite(prediction).all(axis=(-2, -1))
    for row in range(4):
        mean_ax, max_ax, min_ax = fig.axes[3*row:3*row + 3]
        np.testing.assert_allclose(mean_ax.lines[0].get_ydata(), truth[valid][:, row].mean(axis=0))
        np.testing.assert_allclose(mean_ax.lines[1].get_ydata(), prediction[valid][:, row].mean(axis=0))
        mean = truth[valid][:, row].mean(axis=0)
        std = truth[valid][:, row].std(axis=0)
        shaded_y = mean_ax.collections[0].get_paths()[0].vertices[:, 1]
        assert shaded_y.min() == pytest.approx((mean - std).min())
        assert shaded_y.max() == pytest.approx((mean + std).max())
        np.testing.assert_array_equal(max_ax.lines[0].get_ydata(), truth[1, 1, row])
        np.testing.assert_array_equal(min_ax.lines[0].get_ydata(), truth[0, 0, row])
        assert mean_ax.lines[0].get_color() == 'black'
        assert mean_ax.lines[1].get_color() == 'tab:orange'
    assert not plt.fignum_exists(fig.number)


def test_profile_comparison_skips_empty_valid_pixels(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, '_log_validation_image', lambda *args: pytest.fail('Unexpected image'))
    module.plot_profile(np.ones((1, 1, 4, 5)), np.full((1, 1, 4, 5), np.nan))


class ImageRecorder(WandbLogger):
    """Exercise the WandbLogger contract without an optional W&B installation."""
    def __init__(self):
        self.images = []
        self.metrics = []

    @property
    def name(self):
        return 'validation-test'

    @property
    def version(self):
        return '0'

    @property
    def save_dir(self):
        return None

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def log_image(self, key, images, step=None, **kwargs):
        for path in images:
            with Image.open(path) as image:
                image.verify()
        self.images.append((key, step, [Path(path) for path in images]))

    def save(self):
        pass

    def finalize(self, status):
        pass


def validation_batch():
    batch = _physical_batch()['HMI']
    batch['instrument_id'] = 'HMI'
    batch['lin_idx'] = torch.tensor([0])
    batch['pix'] = torch.zeros(16, 2)
    return batch


@pytest.mark.parametrize('enabled', [True, False])
def test_validation_images_follow_interval_and_training_steps(tmp_path, enabled):
    config = _module().snapshot_init
    config['plot_validation'] = enabled
    config['instrument_config'][0].update(limb_correction=True, artifact_correction=True)
    module = MESphericalModule(**config)
    logger = ImageRecorder()
    trainer = Trainer(default_root_dir=tmp_path, accelerator='cpu', devices=1,
                      max_steps=4, max_epochs=-1, check_val_every_n_epoch=None,
                      val_check_interval=2, logger=logger, enable_checkpointing=False,
                      enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0)
    train = DataLoader([{'HMI': validation_batch()} for _ in range(4)], batch_size=None)
    valid = DataLoader([validation_batch()], batch_size=None)
    trainer.fit(module, train_dataloaders=train, val_dataloaders=valid)
    assert trainer.global_step == 4
    if enabled:
        expected = {'Parameter Overview', 'B', 'B_rtp', 'v', 'Limb Correction Factors',
                    'Artifact leakage and correction gate', 'Integrated Stokes vector - Comparison',
                    'Stokes profiles - Comparison'}
        for step in (2, 4):
            assert {key for key, value, _ in logger.images if value == step} == expected
        assert len(logger.images) == 2*len(expected)
        assert all(not path.exists() for _, _, paths in logger.images for path in paths)
    else:
        assert not logger.images
    assert module.val_outputs == []
    assert not plt.get_fignums()


def test_partial_validation_keeps_metrics_but_skips_incomplete_maps(monkeypatch):
    module = _module()
    logger = ImageRecorder()
    module._trainer = SimpleNamespace(global_step=2, sanity_checking=False, loggers=[logger])
    batch = {key: value[:8] if isinstance(value, torch.Tensor) and len(value) == 16 else value
             for key, value in validation_batch().items()}
    monkeypatch.setattr(module, 'log_dict', lambda metrics: None)
    module.on_validation_epoch_start()
    module.on_validation_batch_end(module.validation_step(batch, 0), batch, 0)
    module.on_validation_epoch_end()
    assert not logger.images
    assert module.val_outputs == []
