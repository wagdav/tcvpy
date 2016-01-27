"""
ECE - Electron Cyclotron Emission
"""
import logging

import numpy as np
import xray

import tcv


log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Lfs(object):
    """
    Class to load and analyze ECE data from the LFS ECE channes.
    """

    DEFAULT_FREQUENCIES = np.asarray([
        66.15, 67.6, 69.06, 70.51, 71.97, 73.42, 74.88, 76.33, 77.79, 79.24,
        80.69, 82.15, 85.48, 86.93, 88.39, 89.84, 91.3, 92.75, 94.21, 95.66,
        97.12, 98.57, 100.02, 101.48  # GHz
    ])

    @staticmethod
    def fromshot(shotnum, los=None):
        """ Read the ECE LFS data from the specified shot """

        with tcv.shot(shotnum) as conn:
            try:
                frequency = conn.tdi(r'\results::ece_lfs:rf_freqs')
            except:  # FIXME: catch more specific exception
                frequency = Lfs.DEFAULT_FREQUENCIES
        type(frequency)
        if los:
            # remember that we use the los as index for channels
            los = np.atleast_1d(los)-1
        else:
            los = np.arange(frequency.size)

        values = []
        used_los = []
        with tcv.shot(shotnum) as conn:
            for i, channel in enumerate(Lfs.channels(conn.shot)):
                if i in los:
                    values.append(conn.tdi(channel, dims='time'))
                    used_los.append(i+1)

        data = xray.concat(values, dim='los')
        data.coords['los'] = used_los
        # TODO: add frequency coordinate

        # Normalize to mean value
        mean = data.where(data.time < 0).mean(dim='time')
        data = (data - mean) / mean

        # Fill-in data attributes
        with tcv.shot(shotnum) as conn:
            data.attrs['z_antenna'] = Lfs.zpos(conn)

        return data

    @staticmethod
    def channels(shot):
        """
        Return the names of the data acquisition channels where the data is
        found.
        """

        if shot > 50237:
            base = r'\results::ece_lfs:channel_'
            numbers = np.arange(24) + 1
        else:
            base = r'\atlas::dt100_northeast_002:channel_'
            numbers = np.asarray([28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                                  17, 11, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        return [base + format(i, '03') for i in numbers]

    @staticmethod
    def zpos(shotnum):
        """
        Routine to check which Z position are the LoS for the current shot
        """

        with tcv.shot(shotnum) as conn:
            Zant = str(conn.tdi(r'\results::ece_lfs:z_antenna').values)

            if Zant[: 5] == 'Error':
                pb5 = str(conn.tdi(r'\vsystem::tcv_publicdb_b["ILOT_NO:B5"]')
                          .values)
                pb6 = str(conn.tdi(r'\vsystem::tcv_publicdb_b["ILOT_NO:B6"]')
                          .values)
                if pb5[:3] == 'OFF' and pb6[:3] == 'OFF':
                    Zant = 0.21  # already transformed in floating
                    log.info('LoS at 21 cm')
                elif pb5[: 3] == 'OFF' and pb6[: 3] == 'ON ':
                    Zant = 10
                    log.info('3rd LoS was used')
                elif pb5[: 3] == 'ON ' and pb6[: 3] == 'OFF':
                    Zant = 0.
                    log.info('LoS at 0 cm')
                elif pb5[: 3] == 'ON ' and pb6[: 3] == 'ON ':
                    Zant = np.nan
                    log.info('The LFS ECE radiometer was disconnected')
            else:
                Zant = np.float(Zant[: 2]) * 1e-2
                log.info('LoS at %4.1f cm', Zant * 100)

        return Zant
