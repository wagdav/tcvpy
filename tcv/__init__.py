# -*- coding: utf-8 -*-

"""
This library provides easy access to TCV experimental data using Python.
Here's a quick example how to get the plasma current in the latest experiment:

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
"""
__author__ = 'David Wagner'
__email__ = 'wagdav@gmail.com'
__version__ = '0.1.0'

__all__ = ['shot', 'tcvview']

from . mds import MDSConnection
from . geom import tcvview


def shot(shotnum=0, tree='tcv_shot', server='tcvdata.epfl.ch'):
    """
    Create an MDS connection to the TCV shot database

    Parameters
    ----------
    shotnum : int or MDSConnection instance
        Shot number or an open MDS connection
    tree : str, optional
        Name of the tree to open
    server : str, optional
        MDS database server
    """

    if isinstance(shotnum, MDSConnection):
        conn = shotnum
        return MDSConnection(conn.shot, conn.tree, conn.server)
    else:
        return MDSConnection(shotnum, tree, server)
