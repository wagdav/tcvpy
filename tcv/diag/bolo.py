"""
BOLO diagnostic

Written by Nicola Vianello
"""

import os
import numpy
import scipy
import xray
import tcv

class Bolo(object):
    """
    Load calibrated Bolometry cameras, checking also for good not saturated signals
    All methods implemented as staicmethods so that can be called directly
    """

    @staticmethod
    def fromshot(shot, offset=True):
        """
        Return the calibrated signal of the BOLO diagnostic with the possibility to choose given LoS.
        So far it mimic only one of the filter used
        Parameters
        ----------
        shot: Shot number which will be used
        offset: Boolean. If set it remove the offset before the shot
        Returns
        -------
        Calibrated BOLO signals

        Examples
        -------

        >>> import tcv
        >>> boloData = tcv.diag.Bolo.fromshot(50766)

        """

        # collect the raw data
        raw = conn.tdi(r'\base::bolo:source', dims=('time', 'los'))
        # collect the gains
        gains = conn.tdi(r'\base::bolo:gains')
        # collect the calibration
        calibration = conn.tdi(r'\base::bolo:calibration')
        # collect the etendue
        etend = conn.tdi(r'\base::bolo:geom_fact')
        # define the conversion factor
        convfact = 9.4*0.0015*0.004
        # collect tau
        tau = conn.tdi(r'\base::bolo:tau')
        # collect the geometry dictionary which will be used afterward. I will append it to the data as additional
        # dictionary
        geoDict = Bolo.geo(shot)
        # we define the point when the offset removing mechanism is switched off
        start = conn.tdi('timing("401")').values
        if start < 0 and offset:
            raw -= raw.where((raw.time <0) & (raw.time > 0.7*tstart)).mean(dim='time')
            print ' -- Offset removal -- '
        else:
            print ' -- Offset not removed  --'


    @staticmethod
    def geo(shot, los):
        """
        Returns a dictionary with the geometrical information regarding the passed LoS
        -------

        """
        conn = tcv.shot(shot)
        # angle of pinhole surface normal
        vangle_cp = numpy.repeat(scipy.pi, 8)
        vangle_cp[0] *= -1./2.
        vangle_cp[-1] *= 1/2.

        # radial position of the pinholes
        xpos = numpy.asarray([0.88, 1.235, 1.235, 1.235, 1.235, 1.235, 1.235,0.88])
        # vertical position of the pinholes
        ypos = numpy.asarray([.815, 0.455, 0.455, -.0025, -.0025, -0.46, -0.46, -0.815])
        # provide now the exact positions of the 64 LoS
        xdet = conn.tdi(r'\base::bolo:radial_pos').values
        ydet = conn.tdi(r'\base::bolo:z_pos').values
        # detector poloidal size and toroidal size
        detSize = numpy.asarray([.0015, 0.004])
        # aperture size (poloidal, toroidal)
        apeSize = numpy.asarray([ [0.0026*2, 0.0022*2, 0.0022*2, 0.0022*2, 0.0022*2, 0.0022*2, 0.0022*2, 0.0026*2],
                                 [0.01*2, 0.008*2, 0.008*2, 0.008*2, 0.008*2, 0.008*2, 0.008*2, 0.01*2]])


        out = dict(pinO_x = xpos, pinO_z = ypos, xdet = xdet, ydet = ydet, detsize = detSize, apsize = apeSize)
        return out

    @staticmethod
    def gottardifilt():
