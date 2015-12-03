# -*- coding: utf-8 -*-
__author__  = 'Nicola Vianello'
__version__ = '0.1'
__date__    = '03.12.2015'



import numpy as np
from scipy import signal # this is needed for the computation of the spectrogam
from scipy import io # this is needed to load the settings of xtomo
import matplotlib as mpl # this is need to get the appropriate gca
import xray # this is needed as tdi save into an xray
import tcv  # this is the tcv main library component
import os   # for correctly handling the directory position
from itertools import izip
# for definition of the
diag_path = os.path.dirname(os.path.realpath(__file__))
diag_path += '/dmpxcalib'

class Top(object):
    """
    Class to load and analyze the data contained in the top chamber of DMPX.
    All the methods described as classmethod and so they can be called directly
    Methods:
        read: Read the data calibrated in the same way as mpxdata matlab routine.
              It can read also single chords, accordingly to the call to Top
        gains: Return the gains used for the calibration of the chors
        geo : return the geometrical position of the chords
        spectrogram: It create and eventually plot the spectrograms(s)

    """

    @classmethod
    def channels(Cls, shot, **kwargs):
        """

        Parameters
        ----------
        Input:
            shot: Shot number
        kwargs
            los: Eventually the los we would like to look at
        Returns
        -------
            the string array containing the channels, already sorted out according to shot number
        """
        los = kwargs.get('los', np.arange(64).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        indexLos = los-1 # we convert from LoS to index

        # we provide here all the definition we need
        if shot < 24087  or shot > 24725:
            dtNe1 =r'\atlas::dt100_northeast_001:'
            dtNe2 =r'\atlas::dt100_northeast_002:'
        else:
            dtNe1 =r'\atlas::dt100_southwest_001:'
            dtNe2 =r'\atlas::dt100_southwest_002:'

        # we define the table corresponding to correct cabling of the signals
        # cabling for the shots > 26555
        if shot >= 26555:
            cd2Cords = np.arange(1,65,2)
            cd2Chans = np.arange(1,33,1)
            cd1Cords = np.arange(2,66,2)
            cd1Chans = np.append(np.arange(16,0,-1),np.arange(32,16,-1))
        elif shot>20643 and shot< 26555:
            cd2Cords = np.arange(1,65,2)
            cd2Chans = np.arange(1,33,1)
            cd1Cords = np.arange(2,66,2)
            cd1Chans = np.arange(32,0,-1)
        else:
            raise Exception('This programs does not load shots before 20643')

        # now we define the appropriate board according to the shot number
        # and the presence of fast signals
        cd2Cards = np.repeat(dtNe2,cd2Cords.size)
        cd1Cards = np.repeat(dtNe1,cd1Cords.size)
        # now we collect all together the signals and sort from HFS to LFS
        allCords = np.append(cd1Cords,cd2Cords)
        allChans = np.append(cd1Chans,cd2Chans)
        allCards = np.append(cd1Cards,cd2Cards)
        # sort the index from HFS to LFS
        allChans = allChans[np.argsort(allCords)]
        allCards = allCards[np.argsort(allCords)]
        # now select the chosen diods
        _chosenChans = allChans[indexLos]
        _chosenCards = allCards[indexLos]

        return _chosenCards,_chosenChans

    @classmethod
    def fromshot(Cls, shot, **kwargs):
        """

        Parameters
        ----------
        Input:
            shot = shot number
        kwargs
            los = line of sights
            trange = trange of interest
            plt = boolean. Default is false. If True it plots the data

        Returns
        -------
        An xray.DataArray containing the data, time basis and all the information in dictionary

        Examples
        -------
        In [1]: from tcv.diag.dmpx import Top
        In [2]: data = Top.fromshot(50730,los=32)
        """
        # define the defauls LoS if not given
        los = kwargs.get('los', np.arange(64).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        # define the default trange if not given
        trange = kwargs.get('trange',[-0.01,2.2])
        # first of all choose the right channels and cards
        _chosenCards, _chosenChans = Cls.channels(shot,los=los)

        # we need to repeat this because we do not know apriori which are the dtNe1 and
        # dtNe2
        if shot < 24087  or shot > 24725:
            dtNe1 =r'\atlas::dt100_northeast_001:'
            dtNe2 =r'\atlas::dt100_northeast_002:'
        else:
            dtNe1 =r'\atlas::dt100_southwest_001:'
            dtNe2 =r'\atlas::dt100_southwest_002:'

        # now define the appropriate channels
        # test the dtacq triggering mode
        conn = tcv.shot(shot)
        mode1 = conn.tdi(dtNe1+'MODE').values
        mode2 = conn.tdi(dtNe2+'MODE').values
        mode3 = 4
        # now a raise exception if wrong timing occurs
        mOde = mode1*mode2*mode3
        if shot >= 24087 and mOde != 64:
            print('Random temporal gap (5 to 25 us) between the two or three MPX acquisition cards.')
            print('Random temporal gap (0.1 to 0.2ms) between DTACQ and TCV.')
            raise Warning('DTACQ not in mode 4')

        # load the time basis
        # we check if fast nodes are collected for shot below 34988 and eventually load fast data
        if shot <= 34988:
            try:
                if _chosenChans[0]<10:
                    _str = _chosenCards[0]+'selected:channel_00'+str(_chosenChans[0])
                else:
                    _str = _chosenCards[0]+'selected:channel_0'+str(_chosenChans[0])
                conn.tdi(_str)
                fast = True
                print('Loading high frequency for old shot')
            except:
                fast = False
                print('Loading low frequency for old shot')
        else:
            fast = False
            print('Loading fast data after big opening')

        values = []
        # read the raw data
        for Cards,Chans,Cords in izip(_chosenCards, _chosenChans,los):
            if Chans < 10:
                if fast == True:
                    values.append(conn.tdi(Cards+'selected:channel_00'+str(Chans),dims='time'))
                else:
                    values.append(conn.tdi(Cards+'channel_00'+str(Chans),dims='time'))
            else:
                if fast == True:
                    values.append(conn.tdi(Cards+'selected:channel_0'+str(Chans),dims='time'))
                else:
                    values.append(conn.tdi(Cards+'channel_0'+str(Chans),dims='time'))

            print('reading Cord '+str(Cords)+' Channel '+ str(Chans)+' on Board '+ str(Cards[ -2:-1]))

        # now we create the xray
        data = xray.concat(values, dim = 'los')
        data['los'] = los
        #correct for bad timing
        if shot > 19923 and shot < 20939:
            data.time = (data.time +0.04)*0.9697-0.04
        # correct for missing shots
        if shot>=25102 and shot<25933 and los.__contains__(36):
            data.value[np.argmin(np.abs(los-36)),:]=0
            print('Channel 36 was missing for this shot')
        elif shot>=27127 and shot<=28124:
            #missing channels, problem with cable 2, repaired by DF in Dec 2004
            missing = np.asarray([3,64,62,60,58,56])
            for fault in missing:
                if los.__contains__(fault):
                    data.values[np.argmin(np.abs(los-fault)),:]=0
                    print('Channel '+str(fault)+' missing for this shot')
            if shot>=27185 and los.__contains__(44): #one more channel missing !...
                    data.values[np.argmin(np.abs(los-44)),:]=0
                    print('Channel 44 missing for this shot')
        if shot>=28219 and shot<31446:
            missing = np.asarray([19,21])
            for fault in missing:
                if los.__contains__(fault):
                    data.values[np.argmin(np.abs(los-fault)),:]=0
                    print('Channel '+str(fault)+' missing for this shot')
        # chose calibrate the signals
        # read the gain
        _calib, _gains = Cls.gains(shot,los=los)
        data.values *= (_calib / _gains).reshape(los.size,1)

        # limit to the correct time bases
        data[:,((data.time> trange[0]) & (data.time <= trange[1]))]

        # eventually plot the signals
        plt = kwargs.get('plt', False)
        if plt:
            if np.size(los) > 4:
                fig, axarr = mpl.pyplot.subplots(figsize=(15.7, 9.45), ncols=4,
                                                 nrows=np.round(np.size(los) / 4), sharex=True)
            else:
                fig, axarr = mpl.pyplot.subplots(figsize=(15.7, 5.45), nrows=1, ncols=np.size(los),
                                                 sharex=True)

            if np.size(los) != 1:
                for i in range(data.shape[0]):
                    axarr.flat[i].plot(data.time.values, data.values[i,:])
                    axarr.flat[i].set_xlabel(r't[s]')
                    axarr.flat[i].set_title('# ' + str(shot) + ' DMPX Chords ' + str(los[i]),
                                            fontsize=10)
                fig.tight_layout()
            else:
                axarr.plot(data.time.values, data.values[0,:])
                axarr.set_xlabel(r't[s]')
                axarr.set_title('# ' + str(shot) + ' DMPX Chords ' +
                                str(los), fontsize=10)
            mpl.pylab.show()



        return data

    @classmethod
    def gains(Cls,shot,**kwargs):

        """
        Provide the proper calibration factor for the signals so that can obtain directly the calibration used
        call: Calibration, Gain = Top.gains()

        """
        los = kwargs.get('los', np.arange(64).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        indexLos = los-1 # we convert from LoS to index


        conn = tcv.shot(shot)

        #After #26575, the real voltage is indicated in the Vista window, before it was the reference voltage
        mm = 500 if shot<26765 else 1
        voltage = conn.tdi(r'\VSYSTEM::TCV_PUBLICDB_R["ILOT:DAC_V_01"]').values*mm
        if shot == 32035:
            voltage=2000
        # we first load all the calibration and then choose the correct one according to the ordering
        gainC=np.zeros(64)
        gainR=np.zeros(64)

        I=np.where(shot >= np.r_[20030,23323,26555,27127,29921,30759,31446])[0][-1]
        if I == 0:
            print('Detector gain dependence on the high voltage value not included in the signal calibration')
            calib = io.loadmat(diag_path+'/mpx_calib_first.mat')
            calib_coeff_t=np.squeeze(calib['calib_coeff'])
            gainC[:] = 1
        if I == 1:
            print('Detector gain dependence on the high voltage value not included in the signal calibration')
            calib = io.loadmat(diag_path+'/mpx_calib_sept02.mat')
            calib_coeff_t=np.squeeze(calib['calib_coeff'])
            gainC[:]=1
        if I == 2:
            print('Detector gain dependence on the high voltage value not included in the signal calibration')
            print('There were leaks in the top detector wire chamber for 26554<shot<27128')
            print('Calibration is not very meaningful')
            calib = io.loadmat(diag_path+'/mpx_calib_may04.mat')
            calib_coeff_t=np.np.squeeze(calib['calib_coeff'])
            gainC[:]=1
        if I == 3:
            print('Same gain dependence on the high voltage value taken for each channel')
            calib = io.loadmat(diag_path+'/mpx_calib_july04.mat')
            calib_coeff = np.squeeze(calib['calib_coeff'])
            R = np.squeeze(calib['R'])
            calib_coeff_t = calib_coeff
            calib = io.loadmat(diag_path+'/mpx_calib_may05.mat')
            C = np.squeeze(calib['C'])
            V = np.squeeze(calib['V'])
            C=np.mean(C[:,:64],1) # Use the same gain for each channel
            gainC[:] = np.exp(np.interp(voltage,V,np.log(C)))
            gainR[:]=R
        if I == 4:
            print('Same gain dependence on the high voltage value taken for each channel')
            calib = io.loadmat(diag_path+'/mpx_calib_may05.mat')
            calib_coeff = np.np.squeeze(calib['calib_coeff'])
            C = np.squeeze(calib['C'])
            V = np.squeeze(calib['V'])
            calib_coeff_t=calib_coeff
            C=np.mean(C[:,:64],1) # Use the same gain for each channel
            gainC[:] =np.exp(np.interp(voltage, V,np.log(C)))
            R = io.loadmat(diag_path+'/mpx_calib_july04.mat')['R'] #use the previous relative calibration
            gainR[:]=R
        if I == 5:
            #
            # In this case, the different behaviour of the wires is contained in the matrix of gains.
            # The calibration coefficients are in a vector: one value per wire, same value for all tensions.
            #
            print('Gain dependence on the high voltage value calibrated for each channel')
            print('Leaks in the bottom detector, no relative calibration of the two detectors')
            calib = io.loadmat(diag_path+'/mpx_calib_oct05.mat')
            calib_coeff = np.np.squeeze(calib['calib_coeff'])
            C = np.squeeze(calib['C'])
            V = np.squeeze(calib['V'])
            calib_coeff_t=calib_coeff
            # Interpolation to get the proper gains wrt to the high tension value
            gainC[:]=[np.interp(voltage, V,np.log(C[:,jj])) for jj in range(64)]
            gainR[:]=np.NaN
        if I == 6:
            #
            # In this case, the different behaviour of the wires is contained in the matrix of calibration coefficients.
            # The gains are in a vector: one value per tension, same value for all wires.
            #
            print('Gain dependence on the high voltage value calibrated for each channel')
            #load(sprintf('#mpx_calib_dec05_bis.mat',[mpxpath 'calibration_used/']),'calib_coeff_top','C_top_av','V_top','R')
            calib = io.loadmat(diag_path+'/mpx_calib_dec05_bis.mat')
            calib_coeff_top = np.squeeze(calib['calib_coeff_top'])
            C_top_av = np.squeeze(calib['C_top_av'])
            V_top = np.squeeze(calib['V_top'])
            R = np.squeeze(calib['R'])
            calib_coeff_t = []
            for jj in range(64): # Interpolation to get the proper calibration coefficient wrt the high tension value
                calib_coeff_t.append(np.interp(voltage, V_top,calib_coeff_top[:,jj]))
            gainC[:]=np.exp(np.interp(voltage, V_top,np.log(C_top_av)))
            gainR =R
        # now we can order accordingly to the index for shot > 26555
        # the ordering of the calibration is awful

        II=np.zeros(64,dtype=int)
        if shot >= 26555:
            II[0:63:2] = np.r_[32:64]
            II[1:64:2]=np.r_[15:-1:-1,31:15:-1]
            calib_coeff_t = np.asarray(calib_coeff_t)[II]
            gainC = gainC[II]
        # limit the output to the chosen diods
        cOut = calib_coeff_t[indexLos]
        gOut = gainC[indexLos]

        return cOut, gOut



    @classmethod
    def geo(Cls, shot, **kwargs):
        """

        Parameters
        ----------
        Input:
            shot: shot number
        kwargs
            los : Line of Sight of interest. If not given it consider all of them
            plt : Boolean, default is False. If True it provide a poloidal view with the chords
            t0 : time instant for the equilibrium plot.
        Returns
        -------
            The radial and vertical coordinates of the LoS
        Examples
        -------
        In [1]: from tcv.diag.dmpx import Top
        In [2]: x, y = Top.geo(50766,los=9,t0=6)
        """

        los = kwargs.get('los', np.arange(64).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')

        plt = kwargs.get('plt', False)
        t0 = kwargs.get('t0',0.6)
        # these are the coordinates saved in a mat files
        _geoF = io.loadmat(diag_path + '/dmpx_los.mat')
        xc = _geoF['xchordt'][: , los-1]
        yc = _geoF['ychordt'][: , los-1]

        if plt == True:
            # and now add the LOS of the DMPX
            mpl.pylab.show()
            tcv.tcvview(shot,t0)
            ax=mpl.pylab.gca()
            ax.plot(xc, yc, '--', color = '#ff6600', linewidth = 1.5)
            for l in range(los.size):
                ax.text(xc[1,l]-0.03,yc[1,l]-0.03,str(los[l]),color='green')

        return xc, yc

    @classmethod
    def spectrogram(Cls, shot, nfft=1024, ftStep=2,
                    ftPad=5, ftWidth=0.4, wGauss=True, **kwargs):

        """
        It compute the spectrogam for the chosen signals in the chosen camera.

        Parameters
        ----------
        Input:
            shot   : shot number
        kwargs:
            los    : Optional argument with LoS of the chosen camera. If not set it loads all the 20 channels
            trange : trange. If not set it loads up to 2.2 s of discharge
            wGauss : boolean, Default is True.  For the application of a gaussian window to the spectrogam.
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
        [1]: from tcv.diag.dmpx import Top
        [2]: sp, fr, tsp = Top.spectrogram(50882,2,los=[9,10,11],wGauss=True,nfft=2048,plt=True)

        """

        los = kwargs.get('los', np.arange(64).astype('int')+1)
        if type(los) != np.ndarray:
            if np.size(los)== 1:
                los = np.asarray([los], dtype='int')
            else:
                los = np.asarray(los,dtype='int')
        # the time range
        trange = kwargs.get('trange',[-0.01,2.2])
        # the default values for the spectrogram
        # default values
        ftStep  = kwargs.get('ftStep', 2)
        ftPad   = kwargs.get('ftPad', 5)
        ftWidth = kwargs.get('ftWidth', 0.4)
        # define the keyword for the application of a gaussian window
        # default is true
        wGauss = kwargs.get('wGauss', True)
        # read the signals and compute the sampling frequency
        data = Cls.fromshot(shot,los=los,trange=trange)
        _t = data.time.values
        dt = (_t.max()-_t.min())/(_t.size-1)
        Fs = np.round(1./dt)

        # now build the appropriate gaussian window we use the same algorithm of M.Sertoli
        # generate an array of power of 2
        pow2 = np.power(2, np.arange(12) + 1)
        ftWidth = pow2[np.argmin(np.abs(ftWidth * 1e-3 / dt - pow2))]
        ftStep = ftWidth / ftStep
        ftPad = np.round(2 ** np.round(np.log2(ftWidth * ftPad)))
        sigma = ftWidth / (2 * np.sqrt(2 * np.log(2)))
        w = signal.gaussian(ftPad, sigma)
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
                     axarr.flat[i].set_title('# '+str(shot)+' DMPX LoS '+str(los[i]),fontsize=10)
                     fig.tight_layout()
            else:
                axarr.imshow(np.log10(sp),aspect='auto',origin='lower',
                             extent=(tsp.min(),tsp.max(),fr.min()/1e3,fr.max()/1e3), cmap=cmap)
                axarr.set_xlabel(r't[s]')
                axarr.set_title('# '+str(shot)+' DMPX LoS '+
                                str(los),fontsize=10)

            # introduce the option for saving a pdf file for the plot in the current directory
            save = kwargs.get('save',False)
            if save == True:
                mpl.pylab.savefig(pwd+'/SpectrogramDmpx_'+str(shot)+'.pdf',bbox_to_inches=True)

        return sp,fr,tsp
