Registration
============

Global helpers return correction parameters applied directly in MATLAB axis
order. Python also exposes correction transforms; check axis order before transfer. Local
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
