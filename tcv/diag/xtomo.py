__author__  = 'Nicola Vianello'
__version__ = '0.1'
__date__    = '01.12.2015'


import numpy as np
import scipy
from scipy import io # this is needed to load the settings of xtomo
import matplotlib as mpl # this is need to get the appropriate gca
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
            plt    = Boolean. Default is False. If True it also plot the results
            save   = Boolean. If true together with plt it save the pdf of the plot
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
        plt = kwargs.get('plt',False)
        if plt == True:
            if np.size(los) > 4:
                fig, axarr = mpl.pyplot.subplots(figsize=(15.7, 9.45),
                                                 nrows= np.round(los.size/4),
                                                 ncols=4, sharex=True)
            else:
                fig, axarr = mpl.pyplot.subplots(figsize=(15.7, 5.45),
                                                 nrows=1,
                                                 ncols=los.size, sharex=True)
            if np.size(los) != 1:
                for i in range(data.shape[0]):
                    axarr.flat[i].plot(data.time, data.values[i, :])
                    axarr.flat[i].set_xlabel(r't[s]')
                    axarr.flat[i].set_title('# ' + str(shot) + ' cam ' + str(camera) + ' ph ' + str(los[i]), fontsize=10)
                    fig.tight_layout()
            else:
                axarr.plot(data.time, data.values[0, :])
                axarr.set_xlabel(r't[s]')
                axarr.set_title('# ' + str(shot) + ' cam ' + str(camera) + ' ph ' +
                            str(los + 1), fontsize=10)
            mpl.pylab.show()
            save = kwargs.get('save',False)
            if save == True:
                mpl.pylab.savefig(pwd+'/Signal_'+str(shot)+'.pdf',bbox_to_inches=True)
        return (data)

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
            plt = Boolean, default is False. If True it plot a poloidal cut with the LoS chosen
        Returns
        -------
            the x,y coordinates of the LoS for the chosen camera
        """
        plt = kwargs.get('plt', True)
        t0 = kwargs.get('t0', 0.6) # t0 or middle point
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

        if plt == True:
            tcv.tcvview(shot,t0)
            ax=mpl.pylab.gca()
            ax.plot(xchord,ychord,'k--')
            for l in range(los.size):
                ax.text(xchord[1,l]-0.03,ychord[1,l]-0.03,str(index[l]+1),color='green')
        return (xchord,ychord)

    @classmethod
    def spectrogram(Cls, shot, camera, nfft=1024, ftStep=2,
                    ftPad=5, ftWidth=0.4, wGauss=True, **kwargs):
        """
        It compute the spectrogam for the chosen signals in the chosen camera.

        Parameters
        ----------
        Input:
            shot   = shot number
            camera = number of camera
        kwargs:
            los    = Optional argument with LoS of the chosen camera. If not set it loads all the 20 channels
            trange = trange. If not set it loads up to 2.2 s of discharge
            wGauss: boolean, Default is True.  For the application of a gaussian window to the spectrogam.
                    If false it uses standard window for mpl.mlab.specgram
            nfft: Number of point for the spectrogram. Default is 1024
            ftWidth: FWHM of gaussian window [in ms]. It is used with the gaussian window. Default is 0.4 ms
            ftStep: FT overlap (default is 2 -> ft_width / 2) This keyword is accepted for all the window
            ftPad: FT 0 - padding (as multiple of ftWidth).  Again valid for all the accepted window
            plt : Boolean, default is false. If true it also plot the spectrogram
            save: Boolean, default is false. If you set to true, together with the true for plt it also save the pdf
                  of the figure
        Returns
        -------
        The spectrogram, frequency base, time base of the spectrogram

        Examples
        -------
        [1]: from tcv.diag import xtomo
        [2]: sp, fr, tsp = xtomo.XtomoCamera.spectrogram(50882,2,los=[9,10,11],wGauss=True,nfft=2048,plt=True)


        """
        # the los
        los = kwargs.get('los',np.arange(20).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        # the time range
        trange = kwargs.get('trange',[-0.01,2.2])
        # the default values for the spectrogram
        # default values
        #nfft    = kwargs.get('nfft', 1024)
        ftStep  = kwargs.get('ftStep', 2)
        ftPad   = kwargs.get('ftPad', 5)
        ftWidth = kwargs.get('ftWidth', 0.4)
        # define the keyword for the application of a gaussian window
        # default is true
        wGauss = kwargs.get('wGauss', True)
        # read the signals and compute the sampling frequency
        data = Cls.fromshot(shot,camera,los=los,trange=trange)
        _t = data.time.values
        dt = (_t.max()-_t.min())/(_t.size-1)
        Fs = np.round(1./dt)
        nS = _t.size
        # now build the appropriate gaussian window we use the same algorithm of M.Sertoli
        # generate an array of power of 2
        pow2 = np.power(2, np.arange(12) + 1)
        ftWidth = pow2[np.argmin(np.abs(ftWidth * 1e-3 / dt - pow2))]
        ftStep = ftWidth / ftStep
        ftPad = np.round(2 ** np.round(np.log2(ftWidth * ftPad)))
        from scipy import signal
        sigma = ftWidth / (2 * np.sqrt(2 * np.log(2)))
        w = scipy.signal.gaussian(ftPad, sigma)
        if data.shape[0] == 1:
            # with gaussian windowing
            if wGauss == True:
                sp,fr,tsp = mpl.mlab.specgram(data.values[0,:],Fs = Fs, NFFT = ftPad.astype('int'),
                                              window = w, scale_by_freq = True, noverlap = ftStep.astype('int'))
                tsp += _t.min()
            # standard hanning windowing
            else:
                spc, fr, tsp = mpl.mlab.specgram(data.values[0,:], Fs = Fs, NFFT = ftPad.astype('int'), scale_by_freq = True,
                                                 pad_to = 2 * ftPad.astype('int'), noverlap = ftStep.astype('int'))

        else:
            # gaussian windowing when we have more than one los
            if wGauss == True:
                _du, fr, tsp = mpl.mlab.specgram(data.values[0,:], Fs = Fs, NFFT = ftPad.astype('int'), window = w,
                                                     scale_by_freq = True,  noverlap = ftStep.astype('int'))
                sp = np.zeros((_du.shape[0],_du.shape[1],los.size))
                sp[:,:,0]= _du
                tsp += _t.min()
                for i in range(data.shape[0]-1):
                    _du,fr,tsp = mpl.mlab.specgram(data.values[i+1,:], Fs = Fs, NFFT = ftPad.astype('int'), window = w,
                                                     scale_by_freq = True,  noverlap = ftStep.astype('int'))
                    sp[:,:,i+1] = _du
            else:
                _du, fr, tsp = mpl.mlab.specgram(data.values[0,:], Fs = Fs, NFFT = ftPad.astype('int'),
                                                     scale_by_freq = True,  noverlap = ftStep.astype('int'))
                sp = np.zeros((_du.shape[0],_du.shape[1],los.size))
                sp[:,:,0]= _du
                tsp += _t.min()
                for i in range(data.shape[0]-1):
                    _du,fr,tsp = mpl.mlab.specgram(data.values[i+1,:], Fs = Fs, NFFT = ftPad.astype('int'),
                                                     scale_by_freq = True,  noverlap = ftStep.astype('int'))
                    sp[:,:,i+1] = _du

        # now in case of true we plot the spectrogam
        plt = kwargs.get('plt',False)
        if plt == True:
            if mpl.__version__ >= 1.5:
                cmap = mpl.cm.viridis
            else:
                cmap = mpl.cm.Spectral
            if np.size(los) >4:
                fig, axarr = mpl.pyplot.subplots(figsize = (15.7,9.45),
                                                 nrows = np.round(los.size/4),
                                                 ncols = 4, sharex = True, sharey = True)
            else:
                nrows = np.int(np.round(np.size(los)/2.))
                fig, axarr = mpl.pyplot.subplots(figsize = (15.7,9.45), nrows = 1,
                                                  ncols = los.size, sharex = True, sharey = True)
            if np.size(los) != 1:
                for i in range(sp.shape[2]):
                     axarr.flat[i].imshow(np.log10(sp[:,:,i]),aspect='auto',origin='lower',
                                            extent=(tsp.min(),tsp.max(),fr.min()/1e3,fr.max()/1e3),cmap=cmap)

                     axarr.flat[i].set_xlabel(r't[s]')
                     axarr.flat[i].set_ylabel(r'f [kHz]')
                     axarr.flat[i].set_title('# '+str(shot)+' cam '+str(camera)+' ph '+str(los[i]),fontsize=10)
                     fig.tight_layout()
            else:
                axarr.imshow(np.log10(sp),aspect='auto',origin='lower',
                             extent=(tsp.min(),tsp.max(),fr.min()/1e3,fr.max()/1e3), cmap=cmap)
                axarr.set_xlabel(r't[s]')
                axarr.set_title('# '+str(shot)+' cam '+str(camera)+' ph '+
                                str(los ),fontsize=10)

            # introduce the option for saving a pdf file for the plot in the current directory
            save = kwargs.get('save',False)
            if save == True:
                mpl.pylab.savefig(pwd+'/Spectrogram_'+str(shot)+'.pdf',bbox_to_inches=True)

        return sp,fr,tsp

