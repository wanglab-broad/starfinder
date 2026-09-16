Registration
============

Global helpers return correction parameters applied directly in MATLAB axis
order. Do not negate them using the Python displacement convention. Local
registration uses Image Processing Toolbox demons; Python also offers other
algorithms. See :doc:`../registration` and :doc:`../matlab` for the cross-backend
boundary. These references establish API correspondence, not numerical parity.

.. mat:currentmodule:: .

.. mat:autofunction:: RegisterImagesGlobal

.. mat:autofunction:: RegisterImagesLocal

.. mat:autofunction:: DFTRegister3D

.. mat:autofunction:: DFTApply3D

.. mat:autofunction:: DFTRegister2D

.. mat:autofunction:: DFTApply2D
