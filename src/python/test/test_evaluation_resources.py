"""M79 reduces held-out predictions without fitting a resource model."""
import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.evaluation, pytest.mark.benchmark, pytest.mark.contract, pytest.mark.validation]


@pytest.mark.parametrize("target,units", [("wall_time", "second"), ("peak_rss", "KiB")])
def test_held_out_prediction_errors(target, units):
    from starfinder.evaluation.resources import resource_prediction_errors

    records = pd.DataFrame(dict(dataset_id=["test-a", "test-b", "test-c"],
                               measured=[10., 20., 40.], predicted=[8., 22., 50.]))
    config = dict(training_dataset_ids=["train-a", "train-b"], target=target, units=units)
    result = resource_prediction_errors(records, **config)
    assert result.values == pytest.approx(dict(median_absolute_relative_error=.2, max_absolute_relative_error=.25),
                                         rel=1e-9, abs=1e-12)
    assert result.counts == dict(total=3, defined=3, measured_zero=0, underpredictions=1)
    assert result.details["absolute_relative_errors"] == pytest.approx([.2, .1, .25], rel=1e-9, abs=1e-12)
    assert result.config["target"] == target and result.config["resource_units"] == units
    assert set(result.config["evaluation_dataset_ids"]).isdisjoint(result.config["training_dataset_ids"])
    zero = pd.concat([records, pd.DataFrame(dict(dataset_id=["zero"], measured=[0.], predicted=[5.]))], ignore_index=True)
    result = resource_prediction_errors(zero, **config)
    assert result.counts == dict(total=4, defined=3, measured_zero=1, underpredictions=1)
    assert result.details["excluded_records"] == [dict(position=3, dataset_id="zero", reason="zero measured resource")]
    assert result.values["median_absolute_relative_error"] == pytest.approx(.2, rel=1e-9, abs=1e-12)
    for empty in (records.iloc[:0], zero.iloc[3:]):
        undefined = resource_prediction_errors(empty, **config)
        assert all(v is None for v in undefined.values.values()) and all(undefined.reasons.values())
    with pytest.raises(ValueError, match="held.out"):
        resource_prediction_errors(records, **(config | dict(training_dataset_ids=["test-a"])))


@pytest.mark.parametrize("measured,predicted", [(-1., 1.), (1., -1.), (np.nan, 1.), (1., np.inf)])
def test_invalid_resources(measured, predicted):
    from starfinder.evaluation.resources import resource_prediction_errors

    records = pd.DataFrame(dict(dataset_id=["held-out"], measured=[measured], predicted=[predicted]))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        resource_prediction_errors(records, training_dataset_ids=["train"], target="wall_time", units="second")


def test_resource_target_units_and_dataset_ids_are_explicit():
    from starfinder.evaluation.resources import resource_prediction_errors

    records = pd.DataFrame(dict(dataset_id=[None], measured=[1.], predicted=[1.]))
    config = dict(training_dataset_ids=["train"], target="wall_time", units="second")
    with pytest.raises(ValueError, match="dataset"):
        resource_prediction_errors(records, **config)
    with pytest.raises(ValueError, match="units"):
        resource_prediction_errors(records.assign(dataset_id="held-out"), **(config | dict(units="KiB")))
