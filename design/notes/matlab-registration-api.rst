Registration
============

Global helpers return correction parameters applied directly in MATLAB axis
order. Python exposes pull transforms whose translation is the detected displacement, the negated MATLAB correction; check axis order and sign before transfer. Local
registration uses Image Processing Toolbox demons; Python also offers other
algorithms. See `starfinder.registration <../../docs/api/registration.rst>`_ and `MATLAB API <../../docs/api/matlab.md>`_ for the cross-backend
boundary. These references establish API correspondence, not numerical parity.

