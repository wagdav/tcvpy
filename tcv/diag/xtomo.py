__author__  = 'Nicola Vianello'
__version__ = '0.1'
__date__    = '01.12.2015'


import numpy as np
import scipy as sp
from scipy import io # this is needed to load the settings of xtomo
import matplotlib as mpl # this is need to get the appropriate gca
import xray # this is needed as tdi save into an xray
import tcv  # this is the tcv main library component
import os
# to get the proper directory of the location of the package
_package_folder = os.path.dirname(os.path.realpath(__file__))
class XtomoCamera(object):

    """
    Python class for Loading a single XtomoCamera with appropriate LoS chosen
    eventually diagnostic analysis. As an input it just receive
    the shot number.
    Method:
        'fromshot' = just read the calibrated data eventually in the interval chosen
        'channels' = Return the names of the chosen channels as found in the MDSplus tree
        'plot' = plot all the Chosen LoS
        'gain' = save the gain of the channels
        'geo' = provide a TCV_polview with the chosen LoS plot
        'spectrogram' = compute the spectrogram of all the Diods chosen
        'specplot' = plot the spectrogram of all the diods

    """

    def __init__(self, shot, camera, **kwargs):
        # this is the only self defined in init
        self.shot   = shot
        # given that you now open the corresponding tree through a connection
        self.conn   = tcv.shot(self.shot,tree='tcv_shot',server='tcvdata.epfl.ch')
        self.camera = camera
        self.los    = kwargs.get('los',np.arange(20).astype('int')+1) # the plus 1 is coincident with the fact that
        if isinstance(self.los ,int):
            self.los  = np.asarray([self.los ],dtype='int')
        else:
            self.los  = np.asarray(self.los ,dtype='int')
        # calibration neeeds to be called at the init of the camera
        if self.shot <= 34800:
            self.catDefault = sp.io.loadmat(_package_folder+'/xtomocalib/cat_defaults2001.mat')
        else:
            self.catDefault = sp.io.loadmat(_package_folder+'/xtomocalib/cat_defaults2008.mat')

        self.angFact = self.catDefault['angfact']
        # choose those pertaining to the chosen camera
        self.angFact = self.angFact[:,self.camera-1]
        # now to we need a -1 to ensure that the nDiods are correctly the indices
        self.los -= 1
        self.trange=kwargs.get('trange',[-0.01,2.2])

    @classmethod
    def fromshot(cls, shot,camera,los=None,trange=None):
        """
        Return the calibrated signal of the XtomoCamera LoS chosen in the init action
        Parameters
        ----------
            None

        Returns
        -------
            calibrated signals as an xray data structure

        Examples
        ----------
        In [1]: import tcv
        In [2]: cm= tcv.diag.XtomoCamera(shot,camera,los=los)
        In [3]: data = cm.fromshot()

        """
        # define the LoS
        if los == None:
            los = np.arange(20).astype('int')+1
        else:
            los = np.asarray(los).astype('int')

        # define the trange
        if trange = None:
            trange = [-0.01,2.2]


        # first of all define the proper los
        _Names = self.channels()
        _g,_a = self.gains()
        values=[]
        for _n in _Names:
            values.append(self.conn.tdi(_n,dims='time'))
        data = xray.concat(values,dim='los')
        data['los']=los
        # we remove the offset before the shot
        data -= data.where(data.time<0).mean(dim='time')
        # we limit to the chosen time interval
        data = data[:,((data.time> trange[0]) & (data.time <= trange[1]))]
        # and now we normalize conveniently
        data *= np.transpose(np.tile(_a,(data.values.shape[1],1))/np.tile(_g,(data.values.shape[1],1)))
        # we add also to the attributes the number of the camera
        data.attrs.update({'camera':self.camera})
        # now we need the appropriate gains to provide the calibrated signal
        return data

    def channels(self):
        """
        Provide the names of the channel chosen in the init action
        Parameters
        ----------
        None

        Returns
        -------
         String array with the names
        """

        if (self.camera < 10):
            stringa = '0' + str(self.camera)
        else:
            stringa = str(self.camera)

        _Names = self.conn.tdi(r'\base::xtomo:array_0' + stringa + ':source')

        if np.size(self.los) != 20:
            _Names=_Names[self.los]
        return _Names.values

    def gains(self, **kwargs):
        """

        Parameters
        ----------
        kwargs

        Returns
        -------
        return the gains and the moltiplication factor for the chosen camera and LoS
        """
        if (self.camera < 10):
            stringa = '00' + str(self.camera)
        else:
            stringa = '0' + str(self.camera)
        gAins = np.zeros(20)
        aOut = np.zeros(20)
        # remeber that we need to collect all the values of gains
        # and we decide to choose the only needed afterwards
        for diods in range(20):
            if diods < 9:
                strDiods = '00'+str(diods+1)
            else:
                strDiods ='0'+str(diods+1)
            _str='\\vsystem::tcv_publicdb_i["XTOMO_AMP:'+stringa+'_'+strDiods+'"]'
            out = self.conn.tdi(_str)
            gAins[diods]=10**out.values
            aOut[diods]=self.angFact[diods]

        # now we need to reorder to take into account the ordering of the diodes
        if self.shot<= 34800:
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

        mask = (self.camera-1)*20+np.arange(0,20,1)
        ia = np.argsort(index[np.arange(index.shape[0])[np.in1d(index, mask)]])
        gAins=gAins[ia]


        # we now need to choose only the gains for the given diods
        # just in case we have a single diods we reduce to a single element
        if np.size(self.los)!=20:
            gAins = gAins[self.los]
            aOut  = aOut[self.los]

        return gAins,aOut

    def geo(self, **kwargs):
        plt = kwargs.get('plt', True)
        t0 = kwargs.get('t0', np.mean(self.trange)) # t0 or middle point
        # load the chords and the diods use
        xchord = self.catDefault['xchord'][: , (self.camera - 1) * 20 + self.los] / 100.
        ychord = self.catDefault['ychord'][: , (self.camera - 1) * 20 + self.los] / 100.
        if plt == True:
            tcv.tcvview(self.shot,t0)
            ax=mpl.pylab.gca()
            ax.plot(xchor,ychord,'k--',linewidth=2)
        return (xchord,ychord)

    def plot(self):
        # this is just to plot the data of the camera
        # being 20 plot
        ch = self.fromshot()

        if np.size(self.los) == 20:
            fig, axarr = mpl.pyplot.subplots(figsize = (15.7,9.45),
                                             nrows = 4,
                                             ncols = 5, sharex = True)
        elif np.size(self.los) > 1 and np.size(self.los)<20:
           fig, axarr = mpl.pyplot.subplots(figsize = (15.7,9.45),
                                         nrows = np.round(np.size(self.los)/2),
                                           ncols = 2, sharex = True)
        else:
            fig,axarr = mpl.pyplot.subplots(figsize=plotutils.cm2inch(6,6),
                                            nrows=1,ncols=1)
        if np.size(self.los) != 1:
           for i in range(ch.shape[0]):
                 axarr.flat[i].plot(ch.time, ch.values[i,:])
                 axarr.flat[i].set_xlabel(r't[s]')
                 axarr.flat[i].set_title('# '+str(self.shot)+' cam '+str(self.camera)+' ph '+str(self.los[i] + 1),fontsize=10)
                 fig.tight_layout()
        else:
            axarr.plot(ch.time, ch.values[0,:])
            axarr.set_xlabel(r't[s]')
            axarr.set_title('# '+str(self.shot)+' cam '+str(self.camera)+' ph '+
                str(self.los + 1),fontsize=10)
        mpl.pylab.show()


    def spectrogram(self,**kwargs):
        import matplotlib as mpl
        _s = self.read()
        _t= _s.time
        dt = (_t.max()-_t.min())/(_t.size-1)
        Fs = np.round(1./dt)
        NFFT= kwargs.get('NFFT',2048)
        # do the first and check for the number of points
        if _s.shape[0] == 1:
            sp,fr,tsp = mpl.mlab.specgram(_s.values[0,:],Fs = Fs, NFFT = NFFT, detrend = 'linear',
                                          scale_by_freq = True, pad_to = 3*NFFT)
            tsp += _t.min()
        else:
            _du,fr,tsp = mpl.mlab.specgram(_s.values[0,:], Fs = Fs, NFFT = NFFT,
                                           detrend = 'linear', scale_by_freq = True, pad_to = 3*NFFT)
            sp = np.zeros((_du.shape[0],_du.shape[1],_s.shape[1]))
            sp[:,:,0]= _du
            tsp += _t.min()
            for i in range(_s.shape[0]-1):
                _du,fr,tsp = mpl.mlab.specgram(_s[i+1,:], Fs = Fs, NFFT = NFFT,
                                               detrend = 'linear', scale_by_freq = True, pad_to = 3*NFFT)
                sp[:,:,i+1] = _du
        if self.los.size == 1:
            sp = sp[:,:,0]
        return sp,fr,tsp

    def specplot(self,**kwargs):
        NFFT = kwargs.get('NFFT',2048)
        save = kwargs.get('save',False)
        sp,fr,tsp = self.spectrogram(NFFT=NFFT)
        if sys.platform == 'darwin':
            libC = '/Users/vianello/Documents/Fisica/pytholib/'
        else:
            libC= '/home/vianello/pythonlib/'
        sys.path.append(libC+'submodules/palettable')
        from palettable.colorbrewer.sequential import Oranges_3
       # now you can start the plotting as done for plo
        if np.size(self.los) == 20:
            fig, axarr = mpl.pyplot.subplots(figsize = (15.7,9.45),
                                             nrows = 4,
                                             ncols = 5, sharex = True, sharey = True)
        elif np.size(self.los) > 1 and np.size(self.los)<20:
             nrows = np.int(np.round(np.size(self.los)/2.))
             fig, axarr = mpl.pyplot.subplots(figsize = (15.7,9.45),
                                              nrows = nrows,
                                              ncols = 2, sharex = True, sharey = True)
        else:
            fig,axarr = mpl.pyplot.subplots(figsize= (6,6),
                                            nrows=1,ncols=1)
        if np.size(self.los) != 1:
           for i in range(sp.shape[2]):
                 axarr.flat[i].imshow(np.log10(sp[:,:,i]),aspect='auto',origin='lower',
                                        extent=(tsp.min(),tsp.max(),fr.min()/1e3,fr.max()/1e3),cmap=Oranges_3.mpl_colormap)

                 axarr.flat[i].set_xlabel(r't[s]')
                 axarr.flat[i].set_ylabel(r'f [kHz]')
                 axarr.flat[i].set_title('# '+str(self.shot)+' cam '+str(self.camera)+' ph '+str(self.los[i] + 1),fontsize=10)
                 fig.tight_layout()
        else:
            axarr.imshow(np.log10(sp),aspect='auto',origin='lower',
                                        extent=(tsp.min(),tsp.max(),fr.min()/1e3,fr.max()/1e3),
                         cmap=Oranges_3.mpl_colormap)
            axarr.set_xlabel(r't[s]')
            axarr.set_title('# '+str(self.shot)+' cam '+str(self.camera)+' ph '+
                str(self.los + 1),fontsize=10)

        # introduce the option for saving a pdf file for the plot in the current directory
        if save == True:
            _dir = pwd
            mpl.pylab.savefig(pwd+'/Spectrogram_'+str(self.shot)+'.pdf',bbox_to_inches=True)






