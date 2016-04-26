"""
BOLO diagnostic

Written by Nicola Vianello
"""

import numpy
import scipy
from scipy import signal
import tcv
import xray


class Bolo(object):
    """
    Load calibrated Bolometry cameras, checking also for good
    not saturated signals
    All methods implemented as staticmethods so that can be called directly
    """

    @staticmethod
    def fromshot(shot, offset=True, filter='gottardi'):
        """
        Return the calibrated signal of the BOLO diagnostic with
        the possibility to choose given LoS.
        So far it mimic only one of the filter used
        Parameters
        ----------
        shot: Shot number which will be used
        offset: Boolean. If set it remove the offset before the shot.
        Default is true
        filter: String indicating the type of string used to load the data.
        Available filters are 'gottardi' (local implementation, default)
        or 'savitzky' (not as good as gottardi)
        Returns
        -------
        Calibrated BOLO signals.
        Examples
        -------
        >>> import tcv
        >>> boloData = tcv.diag.Bolo.fromshot(50766)
        """
        conn = tcv.shot(shot)
        # collect the raw data
        raw = conn.tdi(r'\base::bolo:source', dims=('time', 'los'))
        # we lack the correct time bases so we load it
        # and substitue in the XRAY data set
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
        # collect the geometry dictionary which will be used afterward.
        # I will append it to the data as additional
        # dictionary
        geodict = Bolo.geo(shot)
        # we define the point when the offset removing
        # mechanism is switched off
        start = conn.tdi('timing("401")').values
        if start < 0 and offset:
            raw -= raw.where(((raw.time < 0) &
                              (raw.time > 0.7*start))).mean(dim='time')
            print ' -- Offset removal -- '
        else:
            print ' -- Offset not removed  --'

        # now we compute the calibrated data
        if filter == 'gottardi':
            ts, sm, smd = Bolo.gottardifilt(raw)
        elif filter == 'savitzky':
            sm, smd = Bolo.savitzkyfilt(raw)
        elif filter == 'bessel':
            sm, smd = Bolo.bessel(raw)
        else:
            print('Filter not implemented, using default gottardi')
            filter = 'gottardi'
            ts, sm, smd = Bolo.gottardifilt(raw)

        dynVolt = sm + smd*tau.values.reshape(1, tau.shape[0])
        data = dynVolt / (gains.values.reshape(1, gains.shape[0]) *
                          calibration.values.reshape(1, calibration.shape[0]) *
                          etendue.values.reshape(1, etendue.shape[0])*convfact)
        # transform data on xray data source and adding geo as a dictionary
        out = xray.DataArray(data, dims=('time', 'los'))
        if filter == 'gottardi':
            out.coords['time'] = ts
        else:
            out.coords['time'] = time
        out.coords['los'] = numpy.linspace(1, 64, 64, dtype='int')
        # adding the attributes
        out.attrs['shot'] = 'shot'
        out.attrs['units'] = 'W/m^2'
        for key in geodict.keys():
            out.attrs[key] = geodict[key]
        return out

    @staticmethod
    def geo(shot):
        """

        Parameters
        ----------
        shot: Shot number

        Returns
        -------
        Return a dictionary with all the geometrical
        information for the BOLO diagnostic

        """

        conn = tcv.shot(shot)
        # angle of pinhole surface normal
        vangle_cp = numpy.repeat(scipy.pi, 8)
        vangle_cp[0] *= -1./2.
        vangle_cp[-1] *= 1/2.

        # radial position of the pinholes
        xpos = numpy.asarray([0.88, 1.235, 1.235, 1.235,
                              1.235, 1.235, 1.235, 0.88])
        # vertical position of the pinholes
        ypos = numpy.asarray([.815, 0.455, 0.455, -.0025,
                              -.0025, -0.46, -0.46, -0.815])
        # provide now the exact positions of the 64 LoS
        xdet = conn.tdi(r'\base::bolo:radial_pos').values
        ydet = conn.tdi(r'\base::bolo:z_pos').values
        # detector poloidal size and toroidal size
        detsize = numpy.asarray([.0015, 0.004])
        # aperture size (poloidal, toroidal)
        apesize = numpy.asarray([[0.0026*2, 0.0022*2, 0.0022*2, 0.0022*2,
                                  0.0022*2, 0.0022*2, 0.0022*2, 0.0026*2],
                                 [0.01*2, 0.008*2, 0.008*2, 0.008*2,
                                  0.008*2, 0.008*2, 0.008*2, 0.01*2]])
        out = dict([('pinO_x', xpos), ('pinO_z', ypos), ('xdet', xdet),
                    ('ydet', ydet), ('detsize', detsize),
                    ('apsize', apesize)])
        return out

    @staticmethod
    def gottardifilt(data, **kwargs):
        """
        Perform an appropriate smoothing of the signal
        according to a given formula
        Parameters.
        ----------
        data: xarray DataSet as obtained from a TDI call
        knoise: number of smoothing iterations
        ibave: Sample no. to calculate the initial value of the signal.
        Default is 80
        ieave: Sample no. to calculate the final value of the signal.
        Default is 10
        alevel: Suppression parameter. Default is 0.16
        tNoise: interval to calculate the derivative noise. Default: [-0.04, 0]

        Returns
        -------
        Smoothed array and time derivative
        """

        # standard definition
        knoise = kwargs.get('knoise', 20)
        ibave = kwargs.get('ibave', 80)
        ieave = kwargs.get('ieave', 10)
        alevel = kwargs.get('alevel', 0.16)
        tNoise = kwargs.get('tNoise', [-0.04, 0])
        time = data.time.values
        # determine the indices where the noise level is computed
        indNoise = ((time >= tNoise[0]) & (time <= tNoise[1]))
        # create a copy of the signal changing the first and last point
        dataC = data.values.transpose().copy()
        dataC[:, 0] = dataC[:, :ibave].mean()
        dataC[:, -1] = dataC[:, (dataC.shape[1]-ieave+1):-1].mean()
        # copy of the data to perform smoothing
        ys_left = numpy.zeros((dataC.shape[0], dataC.shape[1]))
        ys_right = numpy.zeros((dataC.shape[0], dataC.shape[1]))
        ys_left[:, 0] = dataC[:, 0]
        ys_right[:, -1] = dataC[:, -1]
        count = 0
        while count < knoise:
            diffnorm = numpy.abs(numpy.diff(dataC, axis=-1) /
                                 numpy.diff(time) /
                                 numpy.max(numpy.abs(
                                     numpy.diff(dataC[:, indNoise],
                                                axis=-1) /
                                     numpy.diff(time[indNoise]))[1:]))

            for k in numpy.arange(1, time.size-1, 1):
                ys_left[:, k] = (ys_left[:, k-1] +
                                 dataC[:, k] + dataC[:, k+1])/3. + \
                                 numpy.tanh(alevel *
                                            numpy.max([diffnorm[:, k-1],
                                                       diffnorm[:, k]])) * \
                                (dataC[:, k] - (ys_left[:, k-1] +
                                                dataC[:, k] +
                                                dataC[:, k+1])/3.)
                NB = - k
                ys_right[:, NB] = (ys_right[:, NB+1] +
                                   dataC[:, NB] + dataC[:, NB-1])/3. + \
                                   numpy.tanh(alevel *
                                              numpy.max([diffnorm[:, NB-1],
                                                        diffnorm[:, NB]])) * \
                                  (dataC[:, NB] -
                                   (ys_right[:, NB+1] + dataC[:, NB] +
                                    dataC[:, NB-1])/3.)
            dataC[:, 1:] = (ys_left[:, 1:] + ys_right[:, 1:])/2.
            count += 1
        # redefine the time basis
        ts = numpy.zeros(dataC.shape[1])
        ts[:-1] = 0.5*(time[1:] + time[:-1])
        ts[-1] = ts[-2] + (ts[-2]-ts[-3])
        # define the output smoothed signal
        ys = 0.5*(dataC + numpy.roll(dataC, 1, axis=-1))
        ys[:, -1] = ys[:, -2]
        # define the output smoothed derivative
        ds = numpy.zeros((dataC.shape[0], dataC.shape[1]))
        ds[:, :-1] = numpy.diff(dataC, axis=-1)/numpy.diff(time)
        ds[:, -1] = 0

        return ts, ys.transpose(), ds.transpose()

    @staticmethod
    def savitzkyfilt(data, **kwargs):
        """
        Gives as output the computed smoothed signal and smoothed
        derivative with a Savitzky-Golay
        Smoothed filtering
        Parameters
        ----------
        data: this is the xray dataArray as computed from Connection method
        kwargs:
            iwin : window length (odd numbers). Default is 45
            pol  : polynomial order used for the smoothing.
        For the derivative it uses the pol+1

        Returns
        -------
            (xray data set for signal, xray data set for derivative)
        """
        iwin = kwargs.get('iwin', 21)
        pol = kwargs.get('pol', 4)
        g = [signal.savgol_coeffs(iwin, pol, i) for i in range(3)]
        smoothed = numpy.asarray([numpy.convolve(g[0],
                                                 data.values[:, m],
                                                 mode='same')
                                  for m in range(data.values.shape[1])])
        dt = numpy.mean(numpy.diff(data.time.values))
        smoothedder = -numpy.asarray([numpy.convolve(g[1],
                                                     data.values[:, m],
                                                     mode='same')
                                      for m in range(data.values.shape[1])])/dt
        return smoothed.transpose(), smoothedder.transpose()

    @staticmethod
    def bessel(data, **kwargs):
        """
        Implement a Bessel type analog filter
        ----------
        data: xarray DataSet as obtained from a TDI call


        Returns
        -------
        Smoothed array and time derivative
        """
        time = data.time.values
        dt = (time.max()-time.min())/(time.size-1)
        Ny = numpy.round(1./((time.max()-time.min())/(time.size-1)))/2
        # implement an appropriate Bessel analog filter
        fcutoff = kwargs.get('fcutoff', 30.)
        _Wn = fcutoff/Ny
        b, a = signal.bessel(4, _Wn)
        # create a copy of the signals
        sm = data.values.transpose().copy()
        smd = signal.filtfilt(b, a, numpy.gradient(sm, dt, axis=-1), axis=-1)
        return sm.transpose(), smd.transpose()
