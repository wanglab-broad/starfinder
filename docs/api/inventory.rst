Public API coverage inventory
=============================

Each namespace's supported names are its explicit ``__all__``, listed on its
page under :doc:`python`; this page lists the root convenience exports and the
rule that keeps the lists in step.
Replaced Python interfaces have no compatibility aliases. Private implementation
modules, imported dependencies and CLI implementation helpers are excluded.
Class pages include public methods, properties and dataclass fields; signatures
show literal defaults (``<factory>`` means a fresh per-instance value).
This inventory is software coverage, not scientific qualification.

Root convenience exports
------------------------

Root exports refer to the same objects as their owning namespaces; modules and
``__version__`` are listed as metadata rather than callable APIs.

* ``starfinder.__version__`` (module or version metadata)
* ``starfinder.apply_transform`` → :py:obj:`starfinder.registration.apply_transform`
* ``starfinder.barcode`` (module or version metadata)
* ``starfinder.Dataset`` → :py:obj:`starfinder.dataset.Dataset`
* ``starfinder.decode_barcodes`` → :py:obj:`starfinder.barcode.decode_barcodes`
* ``starfinder.estimate_transform`` → :py:obj:`starfinder.registration.estimate_transform`
* ``starfinder.extract_intensities`` → :py:obj:`starfinder.barcode.extract_intensities`
* ``starfinder.filter_reads`` → :py:obj:`starfinder.barcode.filter_reads`
* ``starfinder.filter_tophat`` → :py:obj:`starfinder.preprocessing.filter_tophat`
* ``starfinder.find_spots`` → :py:obj:`starfinder.spot_finding.find_spots`
* ``starfinder.FOV`` → :py:obj:`starfinder.dataset.FOV`
* ``starfinder.ImageMetadata`` → :py:obj:`starfinder.image.ImageMetadata`
* ``starfinder.load_codebook`` → :py:obj:`starfinder.barcode.load_codebook`
* ``starfinder.load_round`` → :py:obj:`starfinder.io.load_round`
* ``starfinder.load_volume`` → :py:obj:`starfinder.io.load_volume`
* ``starfinder.match_histogram`` → :py:obj:`starfinder.preprocessing.match_histogram`
* ``starfinder.normalize_intensity`` → :py:obj:`starfinder.preprocessing.normalize_intensity`
* ``starfinder.preprocessing`` (module or version metadata)
* ``starfinder.project_image`` → :py:obj:`starfinder.preprocessing.project_image`
* ``starfinder.reconstruct_background`` → :py:obj:`starfinder.preprocessing.reconstruct_background`
* ``starfinder.registration`` (module or version metadata)
* ``starfinder.save_volume`` → :py:obj:`starfinder.io.save_volume`
* ``starfinder.spot_finding`` (module or version metadata)

Maintenance
-----------

A public export is listed in two places: the ``__all__`` of its owning
namespace and that namespace's autosummary list on its ``docs/api`` page.
A root export is also listed in the root section above.
``docs/check_reference.py`` checks all three lists. The
:ref:`general index <genindex>` lists every documented object.
Generated object stubs and HTML are disposable build products, never authored API.
