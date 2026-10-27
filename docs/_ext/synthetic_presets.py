"""The ``synthetic-presets`` directive: a table of one scene's presets, read from the code.

The registries of ``starfinder.synthetic`` are the single source of the preset tables on
the synthetic pages, so a preset that changes in code changes on the site at the next
build. Usage, in MyST::

    ```{synthetic-presets} cellular
    ```
"""
from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive


def _shape(zyx):
    return " × ".join(str(v) for v in zyx)


def _grid(entry):
    return f"{entry['columns']} × {entry['rows']}"


def _spacing(zyx):
    return " × ".join(f"{v:.3g}" for v in zyx)


def _amplicon(synthetic):
    header = ["Preset", "Z × Y × X", "Amplicons per FOV", "Genes", "FOVs", "Seed"]
    rows = [[name, _shape(p["shape_zyx"]), f"{p['count']:,}", str(p["genes"]), str(p["fovs"]), str(p["seed"])]
            for name, p in synthetic.BENCHMARK_PRESETS.items()]
    return header, rows


def _sample(synthetic):
    header = ["Preset", "Tier", "FOV grid", "FOV Z × Y × X", "Voxel (µm)", "Overlap", "Molecules per cell"]
    rows = [[name, p["tier"], _grid(p), _shape(p["fov_shape_zyx"]), _spacing(p["spacing_zyx"]),
             f"{p['overlap']:.2g}", f"{p['molecules_per_cell']:g}"]
            for name, p in synthetic.SAMPLE_PRESETS.items()]
    return header, rows


def _cellular(synthetic):
    header = ["Preset", "Context", "Tier", "FOV grid", "FOV Z × Y × X", "Nuclei", "Molecules"]
    rows = [[name, p["context"], p["tier"], _grid(p), _shape(p["fov_shape_zyx"]),
             f"about {p['expected_nuclei']:,}", f"about {p['expected_molecules']:,}"]
            for name, p in synthetic.CELLULAR_PRESETS.items()]
    return header, rows


def _appearance(synthetic):
    header = ["Preset", "Context", "Tier", "FOV grid", "FOV Z × Y × X", "Appearance bank", "Placed objects"]

    def placed(p):
        return f"{p['count']} per FOV" if p["density"] == "count" else f"to a visible share of {p['target']:.2f}"

    rows = [[name, p["context"], p["tier"], _grid(p), _shape(p["fov_shape_zyx"]),
             "closed-form (no data)" if p["bank_sha256"] is None else f"real: {p['bank']}", placed(p)]
            for name, p in synthetic.APPEARANCE_PRESETS.items()]
    return header, rows


TABLES = {"amplicon": _amplicon, "sample": _sample, "cellular": _cellular, "appearance": _appearance}


def _table(header, rows):
    table = nodes.table(classes=["sf-preset-table"])
    group = nodes.tgroup(cols=len(header))
    table += group
    group.extend(nodes.colspec(colwidth=1) for _ in header)

    def row(cells, literal_first):
        line = nodes.row()
        for i, text in enumerate(cells):
            entry = nodes.entry()
            paragraph = nodes.paragraph()
            paragraph += nodes.literal(text=text) if (literal_first and i == 0) else nodes.Text(text)
            entry += paragraph
            line += entry
        return line

    head = nodes.thead()
    head += row(header, False)
    body = nodes.tbody()
    body.extend(row(cells, True) for cells in rows)
    group += head
    group += body
    return table


class SyntheticPresets(Directive):
    required_arguments = 1
    has_content = False

    def run(self):
        import starfinder.synthetic as synthetic

        scene = self.arguments[0]
        if scene not in TABLES:
            raise self.error(f"synthetic-presets: unknown scene {scene!r}; use one of {sorted(TABLES)}")
        return [_table(*TABLES[scene](synthetic))]


def setup(app):
    app.add_directive("synthetic-presets", SyntheticPresets)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
