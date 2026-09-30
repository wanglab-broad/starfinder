Registration
============

Global helpers return correction parameters applied directly in MATLAB axis
order. Python exposes pull transforms whose translation is the detected displacement, the negated MATLAB correction; check axis order and sign before transfer. Local
registration uses Image Processing Toolbox demons; Python also offers other
algorithms. See :doc:`../registration` and :doc:`../matlab` for the cross-backend
boundary. These references establish API correspondence, not numerical parity.

.. mat:currentmodule:: .

.. mat:autofunction:: DFTApply2D

.. mat:autofunction:: DFTApply3D

.. mat:autofunction:: DFTRegister2D

.. mat:autofunction:: DFTRegister3D

.. mat:autofunction:: RegisterImagesGlobal

.. mat:autofunction:: RegisterImagesLocal
