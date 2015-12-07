"""
XTOMO diagnostics data

Written by Nicola Vianello
"""

import numpy as np
import scipy
import xray # this is needed as tdi save into an xray
import tcv  # this is the tcv main library component
import os   # for correctly handling the directory position
# to get the proper directory of the location of the package
_package_folder = os.path.dirname(os.path.realpath(__file__))
class XtomoCamera(object):

    """
    Python class for Loading a single XtomoCamera with appropriate LoS chosen
    eventually diagnostic analysis. All the methods are defined as classmethod so
    they can be called without instances
    Method:
        'fromshot' = just read the calibrated data eventually in the interval chosen
        'channels' = Return the names of the chosen channels as found in the MDSplus tree
        'gain' = save the gain of the channels
        'geo' = provide a TCV_polview with the chosen LoS plot
        'spectrogram' = compute the spectrogram of all the Diods chosen
    Autor:
        nicola vianello
    Date:
        03 December 2015

    """

    def __init__(self, data):
         # this is the only self defined in init
         self.shot   = data.attrs['shot']
         # given that you now open the corresponding tree through a connection
         self.camera = data.attrs['camera']
         self.los    = data.los.values # the plus 1 is coincident with the fact that
         # now to we need a -1 to ensure that the nDiods are correctly the indices
         self.trange = [data.time.values.min(), data.time.values.max()]
    @classmethod
    def fromshot(Cls, shot, camera, **kwargs):
        """
        Return the calibrated signal of the XtomoCamera LoS chosen. It is build as a `classmethod` so
        can be called without instance
        Parameters
        ----------
        Input:
            shot   = shot number
            camera = number of camera
        kwargs:
            los    = Optional argument with LoS of the chosen camera. If not set it loads all the 20 channels
            trange = trange. If not set it loads up to 2.2 s of discharge

        Returns
        -------
            calibrated signals as an xray data structure

        Examples
        ----------
        In [1]: import tcv
        In [2]: cm= tcv.diag.XtomoCamera.fromshot(shot,camera,los=los)

        """
        # define the appropriate default values
        los = kwargs.get('los',np.arange(20).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        trange = kwargs.get('trange',[-0.01,2.2])
        # first of all define the proper los
        _Names = Cls.channels(shot, camera, los=los)
        _g,_a  = Cls.gains(shot, camera, los=los)
        values=[]
        with tcv.shot(shot) as conn:
            for _n in _Names:
                values.append(conn.tdi(_n,dims='time'))
        data = xray.concat(values,dim='los')
        data['los']= los
        # we remove the offset before the shot
        data -= data.where(data.time<0).mean(dim='time')
        # we limit to the chosen time interval
        data = data[:,((data.time> trange[0]) & (data.time <= trange[1]))]
        # and now we normalize conveniently
        data *= np.transpose(np.tile(_a,(data.values.shape[1],1))/np.tile(_g,(data.values.shape[1],1)))
        # we add also to the attributes the number of the camera
        data.attrs.update({'camera':camera})
        # now we need the appropriate gains to provide the calibrated signal

        return data

    @classmethod
    def channels(Cls, shot, camera, **kwargs):
        """
        Provide the names of the channel chosen in the init action
        Parameters
        ----------
        Input:
            shot   = shot number
            camera = number of camera
        kwargs:
            los    = Optional. Number of diods

        Returns
        -------
         String array with the names
        """

        los = kwargs.get('los',np.arange(20).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        # convert los into indices
        index = los -1

        if (camera < 10):
            stringa = '0' + str(camera)
        else:
            stringa = str(camera)
        with tcv.shot(shot) as conn:
            _Names = conn.tdi(r'\base::xtomo:array_0' + stringa + ':source')

        if np.size(los) != 20:
            _Names=_Names[index]
        return _Names.values

    @classmethod
    def gains(Cls, shot, camera, **kwargs):
        """

        Parameters
        ----------
        Input:
            shot   = shot number
            camera = number of camera
        kwargs:
            los    = Optional. Number of diods

        Returns
        -------
        return the gains and the multiplication factor for the chosen camera and LoS
        """

        los = kwargs.get('los',np.arange(20).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        # to convert we must define an index which is not modified
        IndeX = los -1

        # calibration neeeds to be called at the init of the camera
        if shot <= 34800:
            catDefault = scipy.io.loadmat(_package_folder+'/xtomocalib/cat_defaults2001.mat')
        else:
            catDefault = scipy.io.loadmat(_package_folder+'/xtomocalib/cat_defaults2008.mat')

        # etendue
        angFact = catDefault['angfact'][:,camera-1]

        if (camera < 10):
            stringa = '00' + str(camera)
        else:
            stringa = '0' + str(camera)
        gAins = np.zeros(20)
        aOut = np.zeros(20)
        # remeber that we need to collect all the values of gains
        # and we decide to choose the only needed afterwards
        with tcv.shot(shot) as conn:
            for diods in range(20):
                if diods < 9:
                    strDiods = '00'+str(diods+1)
                else:
                    strDiods ='0'+str(diods+1)
                _str='\\vsystem::tcv_publicdb_i["XTOMO_AMP:'+stringa+'_'+strDiods+'"]'

                out = conn.tdi(_str)
                gAins[diods]=10**out.values
                aOut[diods]=angFact[diods]

        # now we need to reorder to take into account the ordering of the diodes
        if shot<= 34800:
            index=np.asarray([np.arange(1,180,1),180+[2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19]])
            index -= 1 # remember that matlab start from 1
        else:
            aZ=np.arange(1,21,1)
            bZ=np.arange(40,20,-1)
            cZ=np.arange(60,40,-1)
            dZ=np.arange(61,101,1)
            eZ=np.arange(120,100,-1)
            fZ=np.arange(140,120,-1)
            gZ=np.arange(141,161,1)
            hZ=np.arange(180,160,-1)
            iZ=np.arange(200,180,-1)
            index=np.append(aZ,np.append(bZ,np.append(cZ,np.append(dZ,
                                                                   np.append(eZ,np.append(fZ,
                                                                                          np.append(gZ,
                                                                                                    np.append(hZ,iZ))))))))
            index -= 1 # remember that matlab start from 1

        mask = (camera-1)*20+np.arange(0,20,1)
        ia = np.argsort(index[np.arange(index.shape[0])[np.in1d(index, mask)]])
        gAins=gAins[ia]

        # we now need to choose only the gains for the given diods
        # just in case we have a single diods we reduce to a single element
        if np.size(los)!=20:
            gAins = gAins[IndeX]
            aOut  = aOut[IndeX]

        return gAins,aOut

    @classmethod
    def geo(Cls, shot, camera, **kwargs):
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
        los = kwargs.get('los',np.arange(20).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        # convert los into indices
        index = los-1

        if shot <= 34800:
            catDefault = scipy.io.loadmat(_package_folder+'/xtomocalib/cat_defaults2001.mat')
        else:
            catDefault = scipy.io.loadmat(_package_folder+'/xtomocalib/cat_defaults2008.mat')


        # load the chords and the diods use
        xchord = catDefault['xchord'][: , (camera - 1) * 20 + index] / 100.
        ychord = catDefault['ychord'][: , (camera - 1) * 20 + index] / 100.

        return xchord, ychord
