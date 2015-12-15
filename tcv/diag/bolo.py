"""
BOLO diagnostic

Written by Nicola Vianello
"""

import numpy
import scipy
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
        offset: Boolean. If set it remove the offset before the shot. Default is true
        Returns
        -------
        Calibrated BOLO signals. Computation is done as a default using a savitzky-golay method

        Examples
        -------

        >>> import tcv
        >>> boloData = tcv.diag.Bolo.fromshot(50766)

        """
        conn = tcv.shot(shot)
        # collect the raw data
        raw = conn.tdi(r'\base::bolo:source', dims=('time', 'los'))
        # we lack the correct time bases so we load it
        time = conn.tdi(r'dim_of(\base::bolo:signals)').values
        raw.time.values = time
        # collect the gains
        gains = conn.tdi(r'\base::bolo:gains')
        # collect the calibration
        calibration = conn.tdi(r'\base::bolo:calibration')
        # collect the etendue
        etendue = conn.tdi(r'\base::bolo:geom_fact')
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
            raw -= raw.where(((raw.time <0) & (raw.time > 0.7*start))).mean(dim='time')
            print ' -- Offset removal -- '
        else:
            print ' -- Offset not removed  --'

        # now we compute the calibrated data
        sm, smd = Bolo.savitzkyfilt(raw, iwin = 45, pol = 4)
        dynVolt = sm + smd*tau.values.reshape(1,tau.shape[0])
        data = dynVolt / (gains.values.reshape(1,gains.shape[0])) * calibration.values.reshape(1,
                                                                                            calibration.shape[0]
                                                                                               )*etendue.values.reshape(1,etendue.shape[0])*convfact
        return dynVolt

    @staticmethod
    def geo(shot):
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
    def gottardifilt(data, **kwargs):
        """
        Perform an appropriate smoothing of the signal according to a given formula
        Parameters.
        Not working properly
        ----------
        data: xarray DataSet as obtained from a TDI call
        knoise: number of smoothing iterations
        ibave: Sample no. to calculate the initial value of the signal. Default is 80
        ieave: Sample no. to calculate the final value of the signal. Default is 10
        alevel: Suppression parameter. Default is 0.16
        tNoise: interval to calculate the derivative noise. Default: [-0.04, 0]

        Returns
        -------
        Smoothed array
        """

        # standard definition
        knoise = kwargs.get('knoise', 20)
        ibave  = kwargs.get('ibave', 80)
        ieave  = kwargs.get('ieave', 10)
        alevel = kwargs.get('alevel', 0.16)
        tNoise = kwargs.get('tNoise', [-0.04, 0])
        # sampling rate
        dt = (data.time.values.max()-data.time.values.min())/(data.time.size-1)
        time = data.time.values
        # determine the indices where the noise level is computed
        indNoise = ((time>= tNoise[0]) & (time<= tNoise[1]))
        # create a copy of the signal changin the first and last point
        dataC = data.values.transpose().copy()
        dataC[:, 0] = dataC[:, :ibave].mean()
        dataC[:, -1] = dataC[:, (dataC.shape[1]-ieave+1):-1].mean()
        # copy of the data to perform smoothing
        YS_left = dataC.copy()
        YS_right= dataC.copy()
        count=0
        while (count < knoise):
            diffNorm = numpy.abs(numpy.diff(dataC, axis = -1)/numpy.diff(time)/numpy.max(numpy.abs(numpy.diff(
                dataC[:,indNoise], axis = -1)/numpy.diff(time[indNoise]))[1:]))
            for k in numpy.arange(1, data.time.size-1,1):
                YS_left[:, k] = (YS_left[:, k-1] + dataC[:, k] + dataC[:, k+1])/3. + numpy.tanh(alevel*numpy.max([
                    diffNorm[:, k-1], diffNorm[:, k]]))*(dataC[:, k]-(YS_left[:, k-1] + dataC[:, k] + dataC[:, k+1]))/3
                NB = (time.size-1) - k
                YS_right[:, NB] = (YS_right[:, NB+1] + dataC[:, NB] + dataC[:,NB-1])/3. + numpy.tanh(
                    alevel*numpy.max([diffNorm[:, NB-1], diffNorm[:, NB]]))*(dataC[:, NB]-
                                                                             (YS_right[:, NB+1] +dataC[:, NB] +
                                                                              dataC[:,NB-1])/3.)
            dataC[:, 2:] = (YS_left[:, 2:]+ YS_right[:, 2:])/2.
            count += 1
        # redefine the time basis
        ts = 0.5*(time + numpy.roll(time,1))
        ts[0] = 0
        ts[-1] = ts[-2] + (ts[-2]-ts[-3])
        # define the output smoothed signal
        ys = 0.5*(dataC + numpy.roll(dataC, -1, axis=-1))
        ys[:,0] = 0
        ys[:, -1] = ys[:, -2]
        # define the output smoothed derivative
        ds = numpy.diff(dataC, axis = -1)/numpy.diff(time)
        ds[0] = 0

        return ts, ys, ds

    @staticmethod
    def savitzkyfilt(data, **kwargs):
        """
        Gives as output the computed smoothed signal and smoothed derivative with a Savitzky-Golay
        Smoothed filtering
        Parameters
        ----------
        data: this is the xray dataArray as computed from Connection method
        kwargs:
            iwin : window length (odd numbers). Default is 45
            pol  : polynomial order used for the smoothing. For the derivative it uses the pol+1

        Returns
        -------
            (xray data set for signal, xray data set for derivative)
        """
        from scipy import signal
        iwin = kwargs.get('iwin', 25)
        pol = kwargs.get('pol', 4)
        smoothed = data.copy()
        smoothed.values = scipy.signal.savgol_filter(smoothed, iwin, pol, axis = 0)
        smoothedDer = data.copy()
        dt = (data.time.values.max()-data.time.values.min())/(data.time.values.size-1)
        smoothedDer.values = scipy.signal.savgol_filter(smoothedDer, iwin, pol+1, axis=0, delta =dt)
        return (smoothed, smoothedDer)

