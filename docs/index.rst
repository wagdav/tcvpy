.. mcf documentation master file, created by
   sphinx-quickstart on Tue Jul  9 22:26:36 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tcvpy: Data access for the TCV experiment in Python
===================================================

Release v\ |version|. (:ref:`Installation <install>`)

This library provides easy access to experimental data of the Tokamak à
Configuration Variable (TCV_) in Python.  Here's a quick example how to get the
plasma current in the latest experiment::

    >>> import tcv
    >>> conn = tcv.shot()
    >>> ip = conn.tdi(r'tcv_ip()')
    <xray.DataArray 'tcv_ip()' (dim_0: 16384)>
    array([  2229.64526367,   2116.44580078,   2209.29492188, ...,
           -23054.24414062, -22913.67578125, -23049.359375  ], dtype=float32)
    Coordinates:
      * dim_0    (dim_0) float32 -0.318 -0.3178 -0.3176 -0.3174 -0.3172  ...
    Attributes:
        units: A
        query: tcv_ip()
        shot: 0

This example opens the latest TCV shot, executes a TDI_ query and returns an xray_ DataArray.

User Guide
----------

.. toctree::
   :maxdepth: 2

   readme
   installation
   usage

API documentation
-----------------

If you are looking for information on a specific function, class or method,
this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   modules

Contributor Guide
-----------------

If you want to contribute to the project, this part of the documentation is for
you.

.. toctree::
   :maxdepth: 2

   contributing
   authors
   history


.. _TCV: http://spc.epfl.ch/tcv
.. _TDI: http://mdsplus.org/index.php?title=Documentation:Reference:TDI
.. _xray: http://xray.readthedocs.org
