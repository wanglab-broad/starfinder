"""Generate a cellular context scene and a real-appearance scene, save both and read their truth.

``context-tissue2d-v1`` is the routine 2D tissue preset of the cellular scene: closed-form
nuclei, territories, cell types and stains. ``appearance-tissue2d-v1`` places nuclei of
the same context taken from an appearance bank. The routine appearance presets use the
closed-form bank, which is built in memory from scenes of the closed-form generator, so
this example reads no real data and needs no ``STARFINDER_APPEARANCE_BANKS``.
It takes about 10 s on one CPU, almost all of it the generation of the two presets.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

from starfinder.synthetic import (appearance_scene_preset, context_scene_preset, generate_appearance_scene,
                                  generate_context_scene, save_appearance_scene, save_context_scene)


def truth_summary(directory):
    """The saved truth of a scene: its marker pairs and its numbers of nuclei, cells and molecules."""
    directory = Path(directory)
    return {'marker_pairs': pd.read_csv(directory / 'marker_pairs.csv'),
            'nuclei': len(pd.read_csv(directory / 'nuclei.csv')),
            'cells': len(pd.read_csv(directory / 'cells.csv')),
            'molecules': len(pd.read_csv(directory / 'molecules.csv'))}


def run(output):
    """Generate and save both scenes under ``output``; return each scene's saved truth by preset name."""
    output = Path(output)
    book, config = context_scene_preset('context-tissue2d-v1')
    context = save_context_scene(generate_context_scene(book, config=config), output / 'context-tissue2d-v1')
    book, config = appearance_scene_preset('appearance-tissue2d-v1')
    assert config.bank.preset is not None  # the closed-form bank: no file is read
    appearance = save_appearance_scene(generate_appearance_scene(book, config=config),
                                       output / 'appearance-tissue2d-v1')
    return {'context-tissue2d-v1': truth_summary(context), 'appearance-tissue2d-v1': truth_summary(appearance)}


def report(summaries):
    for name, truth in summaries.items():
        print(f"{name}: {truth['nuclei']} nuclei, {truth['cells']} cells, {truth['molecules']} molecules")
        print(truth['marker_pairs'].to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        report(run(sys.argv[1]))
    else:
        with tempfile.TemporaryDirectory() as output:
            report(run(output))
