"""M54–M58 known answers, including historical half-ownership disagreement."""
import numpy as np
import pytest

pytestmark = [pytest.mark.evaluation, pytest.mark.segmentation, pytest.mark.contract, pytest.mark.validation]


def test_instance_f1_iou_grid():
    from starfinder.evaluation.segmentation import instance_f1

    truth = np.array([[[1, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0]]])
    pred = np.array([[[3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4]]])
    result = instance_f1(truth, pred)
    assert result.config["iou_thresholds"] == pytest.approx([.5, .6, .7, .8, .9], rel=1e-9, abs=1e-12)
    for row in result.details["curve"]:
        assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 1)
        assert row["f1"] == pytest.approx(.5, rel=1e-9, abs=1e-12)
    boundary = instance_f1(np.array([[[1, 1, 1, 1]]]), np.array([[[2, 2, 0, 0]]]))
    assert [r["f1"] for r in boundary.details["curve"]] == pytest.approx([1, 0, 0, 0, 0], rel=1e-9, abs=1e-12)
    empty = instance_f1(np.zeros((1, 1, 1), int), np.zeros((1, 1, 1), int))
    assert all(r["f1"] is None and r["reason"] for r in empty.details["curve"])


def test_half_ownership_ties():
    from starfinder.evaluation.segmentation import split_merge_counts
    from starfinder.evaluation._segmentation_helpers import match_nuclei

    truth = np.array([[[1, 1, 2, 2]]])
    pred = np.array([[[3, 4, 3, 4]]])
    historical = match_nuclei(truth, pred)
    assert historical["split"] == {2}
    result = split_merge_counts(truth, pred)
    assert result.counts["splits"] == 2
    assert result.details["split_truth_ids"] == [1, 2]
    assert result.counts["merges"] == 2 and result.counts["merged_truth"] == 2


def test_iou_greedy_label_id_ties():
    from starfinder.evaluation.segmentation import instance_f1

    # Both candidates have IoU .5; the smaller truth ID wins independently of voxel order.
    truth = np.array([[[9, 4]]])
    pred = np.array([[[12, 12]]])
    curve = instance_f1(truth, pred).details["curve"]
    assert [p[:2] for p in curve[0]["matched_pairs"]] == [(4, 12)]
    assert curve[0]["matched_pairs"][0][2] == pytest.approx(.5, rel=1e-9, abs=1e-12)
    assert curve[0]["f1"] == pytest.approx(2 / 3, rel=1e-9, abs=1e-12)
    assert all(r["matched_pairs"] == [] for r in curve[1:])


@pytest.mark.parametrize("truth_dtype", [np.int64, np.uint64])
def test_half_ownership_sparse_ids(truth_dtype):
    from starfinder.evaluation.segmentation import split_merge_counts

    truth = np.array([[[2**55, 2**55, 2**55 + 1, 2**55 + 1]]], truth_dtype)
    pred = np.array([[[2**48, 2**52, 2**48, 2**52]]], np.uint64)
    result = split_merge_counts(truth, pred)
    assert result.details["split_truth_ids"] == [2**55, 2**55 + 1]
    assert result.details["merge_predicted_ids"] == [2**48, 2**52]


def test_empty_label_populations():
    from starfinder.evaluation.segmentation import instance_f1, split_merge_counts

    empty = np.zeros((1, 2, 2), np.uint32)
    foreground = np.ones_like(empty)
    for pred in (empty, foreground):
        result = instance_f1(empty, pred)
        assert all(v is None for v in result.values.values())
        assert result.status == "undefined" and all(result.reasons.values())
    missed = instance_f1(foreground, empty)
    assert list(missed.values.values()) == pytest.approx([0.] * 5, rel=1e-9, abs=1e-12)
    for truth, pred in ((empty, empty), (empty, foreground), (foreground, empty)):
        counts = split_merge_counts(truth, pred).counts
        assert counts["splits"] == counts["merges"] == counts["merged_truth"] == 0


@pytest.mark.parametrize("function", ["instance_f1", "split_merge_counts"])
def test_instance_metrics_require_aligned_integer_labels(function):
    from starfinder.evaluation import segmentation

    metric = getattr(segmentation, function)
    truth = np.ones((1, 2, 2), np.uint32)
    for pred in (np.ones((2, 2), int), np.ones((1, 2, 3), int),
                 np.ones((1, 2, 2), float), -np.ones((1, 2, 2), int)):
        with pytest.raises(ValueError):
            metric(truth, pred)


def test_split_merge_hand_labels():
    from starfinder.evaluation.segmentation import split_merge_counts
    from starfinder.evaluation._segmentation_helpers import match_nuclei

    truth = np.zeros((1, 12, 24), np.uint32)
    truth[0, 1:4, 1:4] = 1
    truth[0, 1:4, 6:12] = 2
    truth[0, 6:9, 1:4] = 3
    truth[0, 6:9, 4:6] = 4
    truth[0, 9:12, 20:23] = 5
    truth[0, 1:3, 15:18] = 6
    truth[0, 5, 15] = 6
    pred = np.zeros_like(truth)
    pred[0, 1:4, 1:4] = 11
    pred[0, 1:4, 6:8], pred[0, 1:4, 8:10], pred[0, 1:4, 10:12] = 12, 13, 18
    pred[0, 6:9, 1:6] = 14
    pred[0, 10:12, 10:13] = 15
    pred[0, 1:3, 15:18] = 16
    pred[0, 5, 15] = 17
    historical = match_nuclei(truth, pred)
    result = split_merge_counts(truth, pred)
    assert result.counts == dict(truth=6, predicted=8, splits=2, merges=1, merged_truth=2)
    assert result.details["split_truth_ids"] == sorted(historical["split"]) == [2, 6]
    assert result.details["merge_predicted_ids"] == sorted(historical["merges"]) == [14]
    assert result.details["merged_truth_ids"] == sorted(historical["merged"]) == [3, 4]


def test_label_size_summary():
    from starfinder.evaluation.segmentation import label_size_summary

    labels = np.array([[[0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 3, 4, 3, 4, 4, 4]]])
    result = label_size_summary(labels)
    assert result.counts["labels"] == 4
    assert result.values == pytest.approx(dict(median=5, q1=3.5, q3=6.5, iqr=3), rel=1e-9, abs=1e-12)
    assert result.details["label_sizes"] == {1: 2, 2: 4, 3: 6, 4: 8}
    empty = label_size_summary(np.zeros((1, 2, 2), int))
    assert empty.counts["labels"] == 0
    assert all(v is None for v in empty.values.values()) and all(empty.reasons.values())


@pytest.mark.parametrize("labels", [np.zeros((2, 2), int), np.array([[[-1]]]), np.ones((1, 1, 1), float)])
def test_label_size_invalid_labels(labels):
    from starfinder.evaluation.segmentation import label_size_summary

    with pytest.raises(ValueError):
        label_size_summary(labels)
