"""Training samplers retain small fields and the end of each epoch."""
import pytest

from spinnme.train.data_loader import BatchesDataset, TensorsDataset, TimeStratifiedBatchSampler


def test_small_field_produces_one_batch_and_tail_chunks_are_retained():
    sampler = TimeStratifiedBatchSampler([[(0, 0)]], chunks_per_batch=8)
    assert len(sampler) == 1
    assert list(sampler) == [[(0, 0)]]
    indices = [(0, index) for index in range(10)]
    sampler = TimeStratifiedBatchSampler([indices], chunks_per_batch=4)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 3
    assert sorted(item for batch in batches for item in batch) == indices


def test_balanced_tail_preserves_pool_weights():
    sampler = TimeStratifiedBatchSampler([[(0, index) for index in range(5)]], 4,
                                         additional_dataset_groups=[[[(1, 0)]]])
    batches = list(sampler)
    assert len(sampler) == len(batches) == 2
    assert {item for batch in batches for item in batch[:4]} == {(0, index) for index in range(5)}
    for batch in batches:
        assert len(batch) == 8
        assert all(item[0] == 0 for item in batch[:4])
        assert all(item[0] == 1 for item in batch[4:])


def test_small_distributed_field_still_steps_on_every_rank(monkeypatch):
    for rank in range(4):
        monkeypatch.setattr(TimeStratifiedBatchSampler, '_distributed_context', staticmethod(lambda: (rank, 4)))
        sampler = TimeStratifiedBatchSampler([[(0, 0)]], 4)
        assert len(sampler) == 1
        assert list(sampler) == [[(0, 0)]]


def test_distributed_tail_retains_all_data_with_equal_rank_steps(monkeypatch):
    indices = [(0, index) for index in range(10)]
    rank_batches = []
    for rank in range(2):
        monkeypatch.setattr(TimeStratifiedBatchSampler, '_distributed_context', staticmethod(lambda: (rank, 2)))
        sampler = TimeStratifiedBatchSampler([indices], 4)
        rank_batches.append(list(sampler))
        assert len(sampler) == 2
    assert len(rank_batches[0]) == len(rank_batches[1])
    assert {item for batches in rank_batches for batch in batches for item in batch} == set(indices)


def test_empty_primary_pool_is_rejected():
    with pytest.raises(ValueError, match='primary sampling pool'):
        TimeStratifiedBatchSampler([[]], 4)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_batch_sizes_fail_before_reading_or_writing_cache(tmp_path, value):
    with pytest.raises(ValueError, match='batch_size'):
        BatchesDataset({}, batch_size=value)
    with pytest.raises(ValueError, match='batch_size'):
        TensorsDataset({}, work_directory=tmp_path, batch_size=value)
    with pytest.raises(ValueError, match='chunks_per_batch'):
        TimeStratifiedBatchSampler([[(0, 0)]], value)
    assert not list(tmp_path.iterdir())
