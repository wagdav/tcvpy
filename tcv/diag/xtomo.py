# THIS is a class for the XTOMO diagnostic in order to read the data and perform the
# analysis. It strongly relies on MDSplus in order to read the data. It assumes that the path
# are correctly set in mdsplus.conf file
__author__  = 'Nicola Vianello'
__version__ = '0.1'
__date__    = '01.12.2015'

import MDSplus
import numpy as np
import scipy as sp
from scipy import io
import matplotlib as mpl
import xray # this is needed as tdi save into an xray
import tcv  # this is the tcv main library component
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
        self.angFact = self.angFact[:,self.nCamera-1]
        # now to we need a -1 to ensure that the nDiods are correctly the indices
        self.los -= 1
        self.trange=kwargs.get('trange',[-0.01,2.2])


    def fromshot(self, **kwargs):
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
        In [2]: cm= tcv.diag.XtomoCamera(shot,nCamera,los=los)
        In [3]: data = cm.fromshot()

        """

        # first of all define the proper los
        _Names = self.channels()
        _g,_a = self.gains()
        _values=[]
        for _n in _Names:
            values.append(self.conn.tdi(_n,dims='time'))
        data = xray.concat(values,dim='los')
        # we remove the offset before the shot
        data -= data.where(data.time<0).mean(dim='time')
        # we limit to the chosen time interval
        data =data.sel(time=(time>self.trange[0]) & (time <= self.trange[1]))
        # and now we normalize conveniently
        data *= np.tile(_a,(data.values.shape[0],1))/np.tile(_g,(data.values.shape[0],1))

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

        if (self.nCamera < 10):
            stringa = '0' + str(self.nCamera)
        else:
            stringa = str(self.nCamera)

        _Names = self.conn.tdi(r'\base::xtomo:array_0' + stringa + ':source')

        if np.size(self.los) != 20:
            _Names=_Names[self.nDiods]
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
        if (self.nCamera < 10):
            stringa = '00' + str(self.nCamera)
        else:
            stringa = '0' + str(self.nCamera)
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

        #

        mask = (self.nCamera-1)*20+np.arange(0,20,1)
        ia = np.argsort(index[np.arange(index.shape[0])[np.in1d(index, mask)]])
        gAins=gAins[ia]


        # we now need to choose only the gains for the given diods
        # just in case we have a single diods we reduce to a single element
        if np.size(self.los)!=20:
            gAins = gAins[self.los]
            aOut  = aOut[self.los]

        return gAins,aOut

    def geo(self, **kwargs):




class camera(XTOMO):
    def __init__(self, shot, nCamera,**kwargs):
        XTOMO.__init__(self,shot)
        self.nCamera = nCamera
        self.nDiods = kwargs.get('nDiods',np.arange(20).astype('int')+1) # the plus 1 is coincident with the fact that
        if isinstance(self.nDiods,int):
            self.nDiods = np.asarray([self.nDiods],dtype='int')
        else:
            self.nDiods = np.asarray(self.nDiods,dtype='int')


    def read(self,**kwargs):
        if (self.nCamera < 10):
            stringa = '0' + str(self.nCamera)
        else:
            stringa = str(self.nCamera)
        _Names = self.tcv.getNode('\\base::xtomo:array_0' + stringa + ':source').getData().data()

        if np.size(self.nDiods) != 20:
            _Names=_Names[self.nDiods]
        if np.size(self.nDiods) == 1:
#            if sys.platform != 'darwin':
            t = self.tcv.getNode(_Names).getDimensionAt().data()
            # else:
            #     c = mds.Connection('tcvdata.epfl.ch')
            #     c.openTree('tcv_shot', self.shot)
            #     t = c.get('dim_of(' + _Names + ')').data()
            #     c.closeTree('tcv_shot', self.shot)
        else:
            # if sys.platform != 'darwin':
            t = self.tcv.getNode(_Names[0]).getDimensionAt().data()
            # else:
            #     c = mds.Connection('tcvdata.epfl.ch')
            #     c.openTree('tcv_shot', self.shot)
            #     t = c.get('dim_of(' + _Names[0] + ')').data()
            #     c.closeTree('tcv_shot', self.shot)

        _nDiods = _Names.size
        # read the time bases Unfortunately I've to do this with a Connection


        _camera = np.zeros((t.size, _nDiods))
        # now read the data
        for name, i in zip(_Names, range(_Names.size)):
            dummy = self.tcv.getNode(name).getData().data()
            dummy -= dummy[(t <= -0.005)].mean()
            _camera[: , i] = dummy


        # read the gains
        gAin,aOut = self.gains()
        _camera=_camera*np.tile(aOut,(np.shape(_camera)[0],1))/np.tile(gAin,(np.shape(_camera)[0],1))
         # limit to the chosen time window
        ii=((t>=self.trange[0]) & (t<=self.trange[1]))
        _camera = _camera[ii,:]
        t=t[ii]

        return _camera, t, _Names

    def gains(self,**kwargs):
        """
        Obtaining the gains of the cameras
        """
        if (self.nCamera < 10):
            stringa = '00' + str(self.nCamera)
        else:
            stringa = '0' + str(self.nCamera)
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
            out = mds.Data.compile(_str).evaluate().data()
            gAins[diods]=10**out
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

        #

        mask = (self.nCamera-1)*20+np.arange(0,20,1)
        print('size of mask '+str(mask.size))
        ia = np.argsort(index[np.arange(index.shape[0])[np.in1d(index, mask)]])
        print('size of mask '+str(mask.size))
        gAins=gAins[ia]


        # we now need to choose only the gains for the given diods
        # just in case we have a single diods we reduce to a single element
        if np.size(self.nDiods)!=20:
            gAins = gAins[self.nDiods]
            aOut  = aOut[self.nDiods]

        return gAins,aOut

    def geo(self,**kwargs):

        plt = kwargs.get('plt', False)
        t0 = kwargs.get('t0', np.mean(self.trange)) # t0 or middle point
        # load the chords and the diods use
        xchord = self.catDefault['xchord'][: , (self.nCamera - 1) * 20 + self.nDiods] / 100.
        ychord = self.catDefault['ychord'][: , (self.nCamera - 1) * 20 + self.nDiods] / 100.
        if plt == True:
            import eqtools
            eq = eqtools.TCVLIUQETree(self.shot)
            # get time time base of the equilibrium
            teq = eq.getTimeBase()
            # ind of the t0
            in0 = np.argmin(np.abs(teq - t0))
            psiRZ = eq.getFluxGrid()[in0,: ,: ]
            rGrid = eq.getRGrid()
            zGrid = eq.getZGrid()
            tilesP, vesselP = eq.getMachineCrossSectionPatch()
            fig = mpl.pyplot.figure( figsize = (6, 8))
            ax=fig.add_subplot(111, aspect='equal')
            ax.set_title(r'Shot # ' + str(self.shot) + ' @ ' + str(t0))
            ax.contour(rGrid, zGrid, psiRZ,  20, colors = 'k')
            ax.plot([xchord[0,: ], xchord[1,: ]] , [ychord[0,: ], ychord[1,: ]], 'r--')
            ax.add_patch(tilesP)
            ax.add_patch(vesselP)
            ax.set_xlim([0.5, 1.2])
            ax.set_ylim([ - 0.8, 0.8])
        return (xchord, ychord)


    def plot(self):
        # this is just to plot the data of the camera
        # being 20 plot
        ch, t, _Nm = self.read()
        if np.size(self.nDiods) == 20:
            fig, axarr = mpl.pyplot.subplots(figsize = plotutils.cm2inch(40, 24),
                                             nrows = 4,
                                             ncols = 5, sharex = True)
        elif np.size(self.nDiods) > 1 and np.size(self.nDiods)<20:
           fig, axarr = mpl.pyplot.subplots(figsize = plotutils.cm2inch(40, 24),
                                         nrows = np.round(np.size(self.nDiods)/2),
                                           ncols = 2, sharex = True)
        else:
            fig,axarr = mpl.pyplot.subplots(figsize=plotutils.cm2inch(14,14),
                                            nrows=1,ncols=1)
        if np.size(self.nDiods) != 1:
           for i in range(ch.shape[1]):
                 axarr.flat[i].plot(t, ch[: , i])
                 axarr.flat[i].set_xlabel(r't[s]')
                 axarr.flat[i].set_title('# '+str(self.shot)+' cam '+str(self.nCamera)+' ph '+str(self.nDiods[i] + 1),fontsize=10)
                 fig.tight_layout()
        else:
            axarr.plot(t, ch)
            axarr.set_xlabel(r't[s]')
            axarr.set_title('# '+str(self.shot)+' cam '+str(self.nCamera)+' ph '+
                str(self.nDiods + 1),fontsize=10)


        mpl.pylab.show()


    def spectrogram(self,**kwargs):
        import matplotlib as mpl
        _s,_t,_n = self.read()
        dt = (_t.max()-_t.min())/(_t.size-1)
        Fs = np.round(1./dt)
        NFFT= kwargs.get('NFFT',2048)
        # do the first and check for the number of points
        if _s.ndim == 1:
            sp,fr,tsp = mpl.mlab.specgram(_s,Fs = Fs, NFFT = NFFT, detrend = 'linear',
                                          scale_by_freq = True, pad_to = 3*NFFT)
            tsp += _t.min()
        else:
            _du,fr,tsp = mpl.mlab.specgram(_s[:,0], Fs = Fs, NFFT = NFFT,
                                           detrend = 'linear', scale_by_freq = True, pad_to = 3*NFFT)
            sp = np.zeros((_du.shape[0],_du.shape[1],_s.shape[1]))
            sp[:,:,0]= _du
            tsp += _t.min()
            for i in range(_s.shape[1]-1):
                _du,fr,tsp = mpl.mlab.specgram(_s[:,i+1], Fs = Fs, NFFT = NFFT,
                                               detrend = 'linear', scale_by_freq = True, pad_to = 3*NFFT)
                sp[:,:,i+1] = _du
        if self.nDiods.size == 1:
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
        if np.size(self.nDiods) == 20:
            fig, axarr = mpl.pyplot.subplots(figsize = plotutils.cm2inch(40, 24),
                                             nrows = 4,
                                             ncols = 5, sharex = True, sharey = True)
        elif np.size(self.nDiods) > 1 and np.size(self.nDiods)<20:
             nrows = np.int(np.round(np.size(self.nDiods)/2.))
             fig, axarr = mpl.pyplot.subplots(figsize = plotutils.cm2inch(40, 24),
                                              nrows = nrows,
                                              ncols = 2, sharex = True, sharey = True)
        else:
            fig,axarr = mpl.pyplot.subplots(figsize=plotutils.cm2inch(14,14),
                                            nrows=1,ncols=1)
        if np.size(self.nDiods) != 1:
           for i in range(sp.shape[2]):
                 axarr.flat[i].imshow(np.log10(sp[:,:,i]),aspect='auto',origin='lower',
                                        extent=(tsp.min(),tsp.max(),fr.min()/1e3,fr.max()/1e3),cmap=Oranges_3.mpl_colormap)

                 axarr.flat[i].set_xlabel(r't[s]')
                 axarr.flat[i].set_ylabel(r'f [kHz]')
                 axarr.flat[i].set_title('# '+str(self.shot)+' cam '+str(self.nCamera)+' ph '+str(self.nDiods[i] + 1),fontsize=10)
                 fig.tight_layout()
        else:
            axarr.imshow(np.log10(sp),aspect='auto',origin='lower',
                                        extent=(tsp.min(),tsp.max(),fr.min()/1e3,fr.max()/1e3),
                         cmap=Oranges_3.mpl_colormap)
            axarr.set_xlabel(r't[s]')
            axarr.set_title('# '+str(self.shot)+' cam '+str(self.nCamera)+' ph '+
                str(self.nDiods + 1),fontsize=10)

        # introduce the option for saving a pdf file for the plot in the current directory
        if save == True:
            _dir = pwd
            mpl.pylab.savefig(pwd+'/Spectrogram_'+str(self.shot)+'.pdf',bbox_to_inches=True)






