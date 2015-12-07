"""
XTOMO diagnostics data

Written by Nicola Vianello
"""

import os

import numpy as np
import scipy.io
import xray

import tcv


class XtomoCamera(object):
    """
    Load single XtomoCamera with appropriate LoS chosen eventually diagnostic
    analysis. All the methods are defined as staticmethods so they can be
    called without instances
    """

    @staticmethod
    def fromshot(shot, camera, los=None):
        """
        Return the calibrated signal of the XtomoCamera LoS chosen.

        Parameters
        ----------
        shot : int or MDSConnection
            Shot number or connection instance
        camera : int
            Number of the XTOMO camera
        los : int or sequence of ints
            Optional argument with lines of sight (LoS) of the chosen camera.
            If None, it loads all the 20 channels

        Returns
        -------
        Calibrated signals XTOMO signals.

        Examples
        --------
        >>> import tcv
        >>> cam = tcv.diag.XtomoCamera.fromshot(50766, camera=1, los=[4, 5])
        """

        if los:
            los = np.atleast_1d(los)
        else:
            los = np.arange(20) + 1

        _Names = XtomoCamera.channels(shot, camera, los=los)
        _g, _a = XtomoCamera.gains(shot, camera, los=los)

        values = []
        with tcv.shot(shot) as conn:
            for _n in _Names:
                values.append(conn.tdi(_n, dims='time'))
        data = xray.concat(values, dim='los')
        data['los'] = los

        # Remove the offset before the shot
        data -= data.where(data.time < 0).mean(dim='time')

        # and now we normalize conveniently
        # FIXME: use xray's infrastructure to compute this
        data *= np.transpose(np.tile(_a, (data.values.shape[1], 1)) / np.tile(_g, (data.values.shape[1], 1)))

        data.attrs.update({'camera': camera})

        return data

    @staticmethod
    def channels(shot, camera, los=None):
        """
        Provide the names of the channel chosen in the init action

        Parameters
        ----------
        shot : int or MDSConnection
            Shot number or connection instance
        camera : int
            Number of the XTOMO camera
        los : int or sequence of ints
            Optional argument with lines of sight (LoS) of the chosen camera.
            If None, it loads all the 20 channels

        Returns
        -------
        Array of strings with the channel names
        """

        if los:
            los = np.atleast_1d(los)
        else:
            los = np.arange(20) + 1

        with tcv.shot(shot) as conn:
            names = conn.tdi(r'\base::xtomo:array_{:03}:source'.format(camera))

        return names[los - 1].values

    @staticmethod
    def gains(shot, camera, los=None):
        """
        Parameters
        ----------
            Same as XtomoCamera.fromshot()

        Returns
        -------
        The gains and the multiplication factor for the chosen camera and LoS
        """

        if los:
            los = np.atleast_1d(los)
        else:
            los = np.arange(20) + 1

        # to convert we must define an index which is not modified
        IndeX = los - 1

        # etendue
        catDefault = XtomoCamera.calibration_data(shot)
        angFact = catDefault['angfact'][:, camera-1]

        gAins = np.zeros(20)
        aOut = np.zeros(20)
        # remeber that we need to collect all the values of gains
        # and we decide to choose the only needed afterwards
        with tcv.shot(shot) as conn:
            for diods in range(20):
                out = conn.tdi(
                    '\\vsystem::tcv_publicdb_i["XTOMO_AMP:{:03}_{:03}"]'
                    .format(camera, diods + 1))
                gAins[diods] = 10**out.values
                aOut[diods] = angFact[diods]

        # now we need to reorder to take into account the ordering of the
        # diodes
        if shot <= 34800:
            index = np.asarray([
                np.arange(1, 180, 1),
                180 + [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
                       18, 17, 20, 19]])
        else:
            aZ = np.arange(1, 21, 1)
            bZ = np.arange(40, 20, -1)
            cZ = np.arange(60, 40, -1)
            dZ = np.arange(61, 101, 1)
            eZ = np.arange(120, 100, -1)
            fZ = np.arange(140, 120, -1)
            gZ = np.arange(141, 161, 1)
            hZ = np.arange(180, 160, -1)
            iZ = np.arange(200, 180, -1)

            # FIXME: what's happening here?
            index=np.append(aZ,np.append(bZ,np.append(cZ,np.append(dZ,
                                                                   np.append(eZ, np.append(fZ,
                                                                                          np.append(gZ,
                                                                                                    np.append(hZ,iZ))))))))
        index -= 1  # remember that matlab start from 1

        mask = (camera - 1) * 20 + np.arange(0, 20, 1)
        ia = np.argsort(index[np.arange(index.shape[0])[np.in1d(index, mask)]])
        gAins = gAins[ia]

        # we now need to choose only the gains for the given diods
        # just in case we have a single diods we reduce to a single element
        if np.size(los) != 20:
            gAins = gAins[IndeX]
            aOut = aOut[IndeX]

        return gAins, aOut

    @staticmethod
    def geo(shot, camera, **kwargs):
        """

        Parameters
        ----------
        Input:
            shot: the shot number
            camera: the camera
        kwargs:
            los = the line of sight
        Returns
        -------
            the x,y coordinates of the LoS for the chosen camera
        """
        # loading the Diods
        los = kwargs.get('los', np.arange(20) + 1)
        los = np.atleast_1d(los)

        index = los - 1  # convert los into indices
        catDefault = XtomoCamera.calibration_data(shot)
        xchord = catDefault['xchord'][:, (camera - 1) * 20 + index] / 100.
        ychord = catDefault['ychord'][:, (camera - 1) * 20 + index] / 100.

        return xchord, ychord

    @staticmethod
    def calibration_data(shot):
        base = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'xtomocalib')

        if shot <= 34800:
            return scipy.io.loadmat(os.path.join(base, 'cat_defaults2001.mat'))
        else:
            return scipy.io.loadmat(os.path.join(base, 'cat_defaults2008.mat'))
