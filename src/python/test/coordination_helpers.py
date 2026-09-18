"""Test-only projections of structured results for existing truth comparisons."""
from starfinder.io.spots import _join_spots


def spot_table(fov, accepted=False):
    reads = fov.filtering_result if accepted else fov.decoding_result
    return _join_spots(fov.spot_result, reads, accepted_only=accepted).rename(
        columns={'gene_id': 'gene', 'observed_color_sequence': 'color_seq'})


def detected_shifts(fov):
    return {name: tuple(-v for v in results[0].transform.correction_zyx)
            for name, results in fov.registration_results.items()}
