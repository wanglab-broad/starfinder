# Coordination reference

Configuration fields and defaults come from the code. See the
{doc}`coordination guide <coordination>` for a complete example and the
{doc}`dataset API <api/dataset>` for signatures and parameter details.

## Pipeline

{py:class}`~starfinder.dataset.PipelineConfig`

```{config-options} starfinder.dataset.PipelineConfig
```

## Execution

{py:class}`~starfinder.dataset.ExecutionConfig`

```{config-options} starfinder.dataset.ExecutionConfig
```

## Morphology

{py:class}`~starfinder.dataset.MorphologyConfig`

```{config-options} starfinder.dataset.MorphologyConfig
```

## Channels

{py:class}`~starfinder.dataset.ChannelInfo`

```{config-options} starfinder.dataset.ChannelInfo
```

## Round roles

{py:class}`~starfinder.dataset.RoundState`

```{config-options} starfinder.dataset.RoundState
```

## Results

| Stage | FOV attribute |
| --- | --- |
| Registration | `registration_results`, `registration_chains`, `registration_attempts`, `registration_record` |
| Spot finding | `spot_result` |
| Extraction | `intensity_result` |
| Decoding | `decoding_result` |
| Scoring | `scoring_result` |
| Deduplication | `deduplication_result` |
| Filtering | `filtering_result` |
| Segmentation | `segmentation_results` |
| Assignment | `assignment_results` |

[Full coordination specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/coordination.md).
