import numpy as np
import os, platform
import astropy.units as u
from astropy import constants as const
import ctypes
from numpy.ctypeslib import ndpointer
from scipy import interpolate
import warnings
from PyQt5.QtCore import QThread, pyqtSignal, QRunnable,QObject
from pathlib import Path
import glob
warnings.simplefilter("default")


def find_and_load_mw_library():
    """
    Find and load the MWTransferArr library with multiple fallback options.

    Priority order:
    1. Newly compiled library named MWTransferArr_compiled* (from pip install)
    2. Pre-compiled binaries in binaries directory

    Returns:
        Tuple[str, ctypes.CDLL]: (library_path, loaded_library)

    Raises:
        RuntimeError: If no working library can be found
    """

    # Get system info
    system = platform.system()
    machine = platform.machine()

    # Get the package directory
    package_dir = Path(__file__).parent.parent
    binaries_dir = package_dir / "binaries"

    # Priority 1: Look for compiled library with our specific name
    # The compiled library will be named MWTransferArr_compiled with platform-specific suffix
    compiled_pattern = str(binaries_dir/ "MWTransferArr_compiled*")
    compiled_libs = glob.glob(compiled_pattern)
    #print(f'!!!!!!{compiled_pattern} and {compiled_libs}')

    if compiled_libs:
        # Try the first match
        lib_path = compiled_libs[0]
        return lib_path
        # try:
        #     lib = ctypes.CDLL(lib_path)
        #     # Verify it has the expected function
        #     try:
        #         _ = lib.pyGET_MW
        #         print(f"Successfully loaded compiled MW library from: {lib_path}")
        #         return lib_path, lib
        #     except AttributeError:
        #         try:
        #             _ = lib.PyGET_MW
        #             print(f"Successfully loaded compiled MW library from: {lib_path}")
        #             return lib_path, lib
        #         except AttributeError:
        #             print(f"Warning: Compiled library doesn't have pyGET_MW function")
        # except OSError as e:
        #     print(f"Warning: Failed to load compiled library: {e}")

    # Priority 2: Fall back to pre-compiled binaries

    if system == 'Darwin':  # macOS
        if machine == 'arm64':
            precompiled_name = "MWTransferArr_arm64.so"
        else:
            precompiled_name = "MWTransferArr.so"
    elif system == 'Linux':
        precompiled_name = "MWTransferArr.so"
    elif system == 'Windows':
        precompiled_name = "MWTransferArr64.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    precompiled_path = binaries_dir / precompiled_name

    if precompiled_path.exists():
        return precompiled_path
        # try:
        #     lib = ctypes.CDLL(str(precompiled_path))
        #     # Verify function exists
        #     try:
        #         _ = lib.pyGET_MW
        #     except AttributeError:
        #         _ = lib.PyGET_MW  # Try alternative name
        #
        #     print(f"Using pre-compiled library from: {precompiled_path}")
        #     warnings.warn(
        #         "Using pre-compiled binary. For better performance, consider reinstalling:\n"
        #         "  pip install --force-reinstall --no-binary :all: pygsfit",
        #         UserWarning
        #     )
        #     return str(precompiled_path), lib
        #
        # except (OSError, AttributeError) as e:
        #     print(f"Failed to load pre-compiled library: {e}")

    # If we get here, nothing worked
    error_msg = (
        "Could not find or load MWTransferArr library.\n\n"
        f"Looked for:\n"
        f"  1. Compiled library: {package_dir}/MWTransferArr_compiled*\n"
        f"  2. Pre-compiled binary: {precompiled_path}\n\n"
    )

    if system == 'Darwin':
        error_msg += (
            "Solutions:\n"
            "  1. Install OpenMP: brew install libomp\n"
            "  2. Reinstall to compile: pip install --force-reinstall --no-binary :all: pygsfit\n"
        )
    elif system == 'Linux':
        error_msg += (
            "Solutions:\n"
            "  1. Install OpenMP: sudo apt-get install libomp-dev\n"
            "  2. Reinstall to compile: pip install --force-reinstall --no-binary :all: pygsfit\n"
        )

    raise RuntimeError(error_msg)

def initGET_MW(libname, load_GRFF = False):
    """
    Python wrapper for fast gyrosynchrotron codes.
    Identical to GScodes.py in https://github.com/kuznetsov-radio/gyrosynchrotron
    This is for the single thread version
    @param libname: path for locating compiled shared library
    @return: An executable for calling the GS codes in single thread

    The pyGET_MW in GRFF that calculate gyroresonance and free-free emission : https://github.com/kuznetsov-radio/GRFF
    has the same calling manner. So, this function can be used to call the function as well when GRFF lib is provided.
    For the single thread version
    """
    _intp = ndpointer(dtype=ctypes.c_int32, flags='F')
    _doublep = ndpointer(dtype=ctypes.c_double, flags='F')

    libc_mw = ctypes.CDLL(libname)
    if not load_GRFF:
        mwfunc = libc_mw.pyGET_MW
    else:
        mwfunc = libc_mw.PyGET_MW
    mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype = ctypes.c_int

    return mwfunc

def sfu2tb(frequency, flux, area=None, size=None, square=True, reverse=False, verbose=False):
    """
        frequency: single element or array, in Hz
        flux: single element or array of flux, in sfu; if reverse, it is brightness temperature in K
        area: area in arcsec^2
        size: Two-dimensional width of the radio source, [major, minor], in arcsec.
              Ignored if both area and size are provided
        reverse: if True, convert brightness temperature in K to flux in sfu integrated uniformly withing the size
    """

    c = const.c.cgs
    k_B = const.k_B.cgs
    sfu = u.jansky * 1e4

    if (not 'area' in vars()) and (not 'size' in vars()):
        print('Neither area nor size is provided. Abort...')

    if not hasattr(frequency, 'unit'):
        # assume frequency is in Hz
        frequency = frequency * u.Hz

    if not hasattr(flux, 'unit'):
        # assume flux is in sfu
        if reverse:
            flux = flux * u.K
        else:
            flux = flux * sfu

    if area is not None:
        if not hasattr(area, 'unit'):
            # assume area is in arcsec^2
            area = area * u.arcsec ** 2

    if size is not None and (area is None):
        if type(size) != list:
            size = [size]

        if len(size) > 2:
            print('size needs to have 1 or 2 elements.')
        elif len(size) < 2:
            if verbose:
                print('Only one element is provided for source size. Assume symmetric source')
            if not hasattr(size[0], 'unit'):
                # assume size in arcsec
                size[0] = size[0] * u.arcsec
            # define half size
            a = b = size[0] / 2.
        else:
            if not hasattr(size[0], 'unit'):
                # assume size in arcsec
                size[0] = size[0] * u.arcsec
            if not hasattr(size[1], 'unit'):
                # assume size in arcsec
                size[1] = size[1] * u.arcsec
            # define half size
            a = size[0] / 2.
            b = size[1] / 2.
        if square:
            if verbose:
                print('Assume square-shaped source.')
            area = 4. * a * b
        else:
            if verbose:
                print('Assume elliptical-shaped source.')
            area = np.pi * a * b

    sr = area.to(u.radian ** 2)
    factor = c ** 2. / (2. * k_B * frequency ** 2. * sr)

    if reverse:
        # returned value is flux in sfu
        if verbose:
            print('converting input brightness temperature in K to flux density in sfu.')
        return (flux / factor).to(sfu, equivalencies=u.dimensionless_angles())
    else:
        # returned value is brightness temperature in K
        if verbose:
            print('converting input flux density in sfu to brightness temperature in K.')
        return (flux * factor).to(u.K, equivalencies=u.dimensionless_angles())

def ff_emission(em, T=1.e7, Z=1., mu=1.e10):
    T = T * u.k
    mu = mu * u.Hz
    esu = const.e.esu
    k_B = const.k_B.cgs
    m_e = const.m_e.cgs
    c = const.c.cgs
    bmax = (3 * k_B * T * u.k / m_e) ** 0.5 / 2.0 / np.pi / (mu * u.Hz)
    bmin = Z * esu ** 2 / 3. / k_B / T
    lnbb = np.log((bmax / bmin).value)
    ka_mu = 1. / mu ** 2 / T ** 1.5 * (
            Z ** 2 * esu ** 6 / c / np.sqrt(2. * np.pi * (m_e * k_B) ** 3)) * np.pi ** 2 / 4.0 * lnbb

    opc = ka_mu * em
    return T.value * (1 - np.exp(-opc.value))


class GSCostFunctions:
    def SinglePowerLawMinimizerOneSrc(fit_params, freqghz, spec=None, spec_err=None,
                                      spec_in_tb=True, pgplot_widget=None, show_plot=False, debug=False,
                                      elec_dist_index=None, verbose=False):
        """
        params: parameters defined by lmfit.Paramters()
        freqghz: frequencies in GHz
        spec: input spectrum, can be brightness temperature in K, or flux density in sfu
        spec_err: uncertainties of spectrum in K or sfu
        spec_in_tb: if True, input is brightness temperature in K, otherwise is flux density in sfu
        calc_flux: Default (False) is to return brightness temperature.
                    True if return the calculated flux density. Note one needs to provide src_area/src_size for this
                        option. Otherwise assumes src_size = 2 arcsec (typical EOVSA pixel size).
        elec_dist_index: place holder at this moment.
        @rtype: 1. If no tb/tb_err or flux/flux_err is provided, return the calculated
                    brightness temperature or flux for each input frequency.
                2. If tb/tb_err or flux/flux_err are provided, return the
                    (scaled) residual for each input frequency
        """

        # if platform.system() == 'Linux' or platform.system() == 'Darwin':
        #     cur_lib_flie = '../binaries/MWTransferArr.so'
        #     if platform.machine() == 'arm64':
        #         cur_lib_flie = '../binaries/MWTransferArr_arm64.so'
        #     libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                            cur_lib_flie)
        # if platform.system() == 'Windows': ##TODO: not yet tested on Windows platform
        #     libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                            '../binaries/MWTransferArr64.dll')
        libname = find_and_load_mw_library()
        GET_MW = initGET_MW(libname)  # load the library

        asec2cm = 0.725e8
        if 'area_asec2' in fit_params.keys():
            src_area = float(fit_params['area_asec2'].value)  # source area in arcsec^2
        else:
            src_area = 4.  # arcsec^2. default area for bright temperature spectral fitting. Will be divided out.
            if not spec_in_tb:
                print('=======Warning: no source area is provided for flux density calculation. '
                      'Use area = 4 arcsec^2 as the place default (1 EOVSA pixel).======')
        src_area_cm2 = src_area * asec2cm ** 2.  # source area in cm^2
        depth_cm = float(fit_params['depth_asec'].value) * asec2cm  # total source depth in cm
        Bmag = float(fit_params['Bx100G'].value) * 100.  # magnetic field strength in G
        Tth = float(fit_params['T_MK'].value) * 1e6  # thermal temperature in K
        nth = 10. ** float(fit_params['log_nth'].value)  # thermal density
        nrl = 10. ** float(fit_params['log_nnth'].value)  # total nonthermal density above E_min
        delta = float(fit_params['delta'].value)  # powerlaw index
        theta = float(fit_params['theta'].value)  # viewing angle in degrees
        Emin = float(fit_params['Emin_keV'].value) / 1e3  # low energy cutoff of nonthermal electrons in MeV
        Emax = float(fit_params['Emax_MeV'].value)  # high energy cutoff of nonthermal electrons in MeV
        if debug:
            # debug against previous codes
            print('depth, Bmag, Tth, nth/1e10, lognrl, delta, theta, Emin, Emax: '
                  '{0:.1f}, {1:.1f}, {2:.1f}, {3:.1f}, {4:.1f}, '
                  '{5:.1f}, {6:.1f}, {7:.2f}, {8:.1f}'.format(depth_cm / 0.725e8, Bmag, Tth / 1e6, nth / 1e10,
                                                              np.log10(nrl), delta, theta, Emin, Emax))
        # E_hi = 0.1
        # nrl = nrlh * (Emin ** (1. - delta) - Emax * (1. - delta)) / (E_hi ** (1. - delta) - Emax ** (1. - delta))

        Nf = 100  # number of frequencies
        NSteps = 1  # number of nodes along the line-of-sight

        Lparms = np.zeros(11, dtype='int32')  # array of dimensions etc.
        Lparms[0] = NSteps
        Lparms[1] = Nf

        Rparms = np.zeros(5, dtype='double')  # array of global floating-point parameters
        Rparms[0] = src_area_cm2  # Area, cm^2
        Rparms[1] = 0.8e9  # starting frequency to calculate spectrum, Hz
        Rparms[2] = 0.02  # logarithmic step in frequency
        Rparms[3] = 0  # f^C
        Rparms[4] = 0  # f^WH

        ParmLocal = np.zeros(24, dtype='double')  # array of voxel parameters - for a single voxel
        ParmLocal[0] = depth_cm / NSteps  # voxel depth, cm
        ParmLocal[1] = Tth  # T_0, K
        ParmLocal[2] = nth  # n_0 - thermal electron density, cm^{-3}
        ParmLocal[3] = Bmag  # B - magnetic field, G
        ParmLocal[6] = 3  # distribution over energy (PLW is chosen, 3)
        ParmLocal[7] = nrl  # n_b - nonthermal electron density, cm^{-3}
        ParmLocal[9] = Emin  # E_min, MeV
        ParmLocal[10] = Emax  # E_max, MeV
        ParmLocal[12] = delta  # \delta_1
        ParmLocal[14] = 0  # distribution over pitch-angle (isotropic is chosen)
        ParmLocal[15] = 90  # loss-cone boundary, degrees
        ParmLocal[16] = 0.2  # \Delta\mu

        Parms = np.zeros((24, NSteps), dtype='double', order='F')  # 2D array of input parameters - for multiple voxels
        for i in range(NSteps):
            Parms[:, i] = ParmLocal  # most of the parameters are the same in all voxels
            Parms[4, i] = theta

        RL = np.zeros((7, Nf), dtype='double', order='F')  # input/output array
        dummy = np.array(0, dtype='double')

        # calculating the emission for array distribution (array -> on)
        res = GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)
        if True:
            # retrieving the results
            f = RL[0]
            I_L = RL[5]
            I_R = RL[6]
            all_zeros = not RL.any()
            if not all_zeros:
                flux_model = I_L + I_R
                flux_model = np.nan_to_num(flux_model) + 1e-11
                logf = np.log10(f)
                logflux_model = np.log10(flux_model)
                logfreqghz = np.log10(freqghz)
                interpfunc = interpolate.interp1d(logf, logflux_model, kind='linear')
                logmflux = interpfunc(logfreqghz)
                mflux = 10. ** logmflux
                mtb = sfu2tb(np.array(freqghz) * 1.e9, mflux, area=src_area).value

                if pgplot_widget:
                    ##todo: figure out a way to update the main widget
                    import pyqtgraph as pg
                    all_items = pgplot_widget.getPlotItem().listDataItems()
                    if len(all_items) > 0:
                        pgplot_widget.removeItem(all_items[-1])
                    spec_fitplot = pgplot_widget.plot(x=np.log10(freqghz), y=np.log10(mtb),
                                                             pen=dict(color=pg.mkColor(0), width=3),
                                                             symbol=None, symbolBrush=None)
                    pgplot_widget.addItem(spec_fitplot)

                if show_plot:
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    ax1.plot(freqghz, mflux, 'k')
                    #ax1.set_xlim([1, 20])
                    ax1.set_xlabel('Frequency (GHz)')
                    ax1.set_ylabel('Flux (sfu)')
                    ax1.set_title('Flux Spectrum')
                    ax1.set_xscale('log')
                    ax1.set_yscale('log')
                    ax2.plot(freqghz, mtb, 'k')
                    #ax2.set_xlim([1, 20])
                    ax2.set_xlabel('Frequency (GHz)')
                    ax2.set_ylabel('Brightness Temperature (K)')
                    ax2.set_title('Brightness Temperature Spectrum')
                    ax2.set_xscale('log')
                    ax2.set_yscale('log')
                    ax2.legend()
                    plt.show()
            else:
                print("Calculation error! Assign an unrealistically huge number")
                mflux = np.ones_like(freqghz) * 1e4
                mtb = sfu2tb(np.array(freqghz) * 1.e9, mflux, area=src_area).value
        else:
            print("Calculation error! Assign an unrealistically huge number")
            mflux = np.ones_like(freqghz) * 1e9
            mtb = sfu2tb(np.array(freqghz) * 1.e9, mflux, area=src_area).value

        # Return values
        if spec_in_tb:
            if spec is None:
                # nothing is provided, return the model spectrum
                return mtb
            if spec_err is None:
                # no uncertainty provided, return absolute residual
                return mtb - spec
            # Return scaled residual
            return (mtb - spec) / spec_err
        else:
            if spec is None:
                # nothing is provided, return the model spectrum
                return mflux
            if spec_err is None:
                # no uncertainty provided, return absolute residual
                return mflux - spec
            # Return scaled residual
            return (mflux - spec) / spec_err

    def GRFFMinimizerOneSrc(fit_params, freqghz, spec=None, spec_err=None,
                                      spec_in_tb=True, pgplot_widget=None, show_plot=False, mechanism_index=0, debug=False, verbose=False):
        """
        # emission mechanism flag (1: gyroresonance is off; 2: free-free is off;
        # 4: contribution of neutrals is off;  8: even if DEM and/or DDM are present, the free-free and
        # gyroresonance emissions are computed using the isothermal approximation with the electron concentration
        params: parameters defined by lmfit.Paramters()
        freqghz: frequencies in GHz
        spec: input spectrum, can be brightness temperature in K, or flux density in sfu
        spec_err: uncertainties of spectrum in K or sfu
        spec_in_tb: if True, input is brightness temperature in K, otherwise is flux density in sfu
        calc_flux: Default (False) is to return brightness temperature.
                    True if return the calculated flux density. Note one needs to provide src_area/src_size for this
                        option. Otherwise assumes src_size = 2 arcsec (typical EOVSA pixel size).
        @rtype: 1. If no tb/tb_err or flux/flux_err is provided, return the calculated
                    brightness temperature or flux for each input frequency.
                2. If tb/tb_err or flux/flux_err are provided, return the
                    (scaled) residual for each input frequency
        """

        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            cur_lib_flie = '../binaries/GRFF_DEM_Transfer.so'
            if platform.machine() == 'arm64':
                cur_lib_flie = '../binaries/GRFF_DEM_Transfer_mac_arm64.so'
            libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   cur_lib_flie)
        if platform.system() == 'Windows': ##TODO: not yet tested on Windows platform
            libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   '../binaries/GRFF_DEM_Transfer_64.dll')
        GET_MW = initGET_MW(libname, load_GRFF=True)  # load the library

        asec2cm = 0.725e8
        if 'area_asec2' in fit_params.keys():
            src_area = float(fit_params['area_asec2'].value)  # source area in arcsec^2
        else:
            src_area = 4.  # arcsec^2. default area for bright temperature spectral fitting. Will be divided out.
            if not spec_in_tb:
                print('=======Warning: no source area is provided for flux density calculation. '
                      'Use area = 4 arcsec^2 as the place default (1 EOVSA pixel).======')

        src_area_cm2 = src_area * asec2cm ** 2.  # source area in cm^2
        depth_cm = float(fit_params['depth_asec'].value) * asec2cm  # total source depth in cm
        Bmag = float(fit_params['Bx100G'].value) * 100.  # magnetic field strength in G
        Tth = float(fit_params['T_MK'].value) * 1e6  # thermal temperature in K
        nth = 10. ** float(fit_params['log_nth'].value)  # thermal density
        theta = float(fit_params['theta'].value)  # viewing angle in degrees
        #EmiMech_lkuptab = {'thermal f-f + gyrores':0, 'thermal f-f ':1}
        #EmissionMechanism = EmiMech_lkuptab[ele_dist]
        if debug:
            # debug against previous codes
            print('depth, Bmag, Tth, nth/1e10, lognrl, delta, theta, Emin, Emax: '
                  '{0:.1f}, {1:.1f}, {2:.1f}, {3:.1f},'
                  '{6:.1f}'.format(depth_cm / 0.725e8, Bmag, Tth / 1e6, nth / 1e10, theta))
        # E_hi = 0.1
        # nrl = nrlh * (Emin ** (1. - delta) - Emax * (1. - delta)) / (E_hi ** (1. - delta) - Emax ** (1. - delta))

        Nf = 100  # number of frequencies
        #assume voxel with the same depth of dx and dy
        NSteps = np.round(depth_cm/(0.725e8*2)).astype(int) # number of pixel along z axis
        #NSteps = 1  # number of nodes along the line-of-sight

        Lparms = np.zeros(5, dtype='int32')  # array of dimensions etc.
        Lparms[0] = NSteps
        Lparms[1] = Nf
        Lparms[2] = 0 #number of temperatures in the T_arr array; must be ≥ 2 – otherwise DEM/DEM are ignored;
        Lparms[3] = 1 # DEM_key – global DEM on/off key. (1 is off while 0 is on)
        Lparms[4] = 1 # DDM_key – global DDM on/off key. (1 is off while 0 is on)
        #Lparms[5] = len(fzeta_arr)
        #Lparms[6] = len(Tzeta_arr)
        #Lparms[7] = 1

        Rparms = np.zeros(3, dtype='double')  # array of global floating-point parameters
        Rparms[0] = src_area_cm2  # Area, cm^2
        Rparms[1] = 0.8e9  # starting frequency to calculate spectrum, Hz
        Rparms[2] = 0.02  # logarithmic step in frequency

        ParmLocal = np.zeros(15, dtype='double')  # array of voxel parameters - for a single voxel
        ParmLocal[0] = depth_cm / NSteps  # voxel depth, cm
        ParmLocal[1] = Tth  # T_0, K
        ParmLocal[2] = nth  # n_0 - thermal electron density, cm^{-3}
        ParmLocal[3] = Bmag  # B - magnetic field, G

        ParmLocal[4] = theta  # viewing angle, degrees
        ParmLocal[5] = 0  # azimuthal angle, degrees
        ParmLocal[6] = mechanism_index  # emission mechanism flag (1: gyroresonance is off; 2: free-free is off;
        # 4: contribution of neutrals is off;  8: even if DEM and/or DDM are present, the free-free and
        # gyroresonance emissions are computed using the isothermal approximation with the electron concentration
        # and temperature derived from the DDM or DEM (from the DDM, if both are specified).)
        ParmLocal[7] = 30  # maximum harmonic number
        ParmLocal[8] = 0  # proton concentration, cm^{-3} (not used in this example)
        ParmLocal[9] = 0  # neutral hydrogen concentration, cm^{-3}
        ParmLocal[10] = 0  # neutral helium concentration, cm^{-3}
        ParmLocal[11] = 1  # local DEM on/off key (off)
        ParmLocal[12] = 1  # local DDM on/off key (off)
        ParmLocal[13] = 0  # element abundance code (coronal, following Feldman 1992)
        ParmLocal[14] = 0  # reserved


        Parms = np.zeros((15, NSteps), dtype='double', order='F')  # 2D array of input parameters - for multiple voxels
        decay_ratio = 0.5 #todo: edit the ratio/decay function in GUI
        for i in range(NSteps):
            Parms[:, i] = ParmLocal
            Parms[0, i] = ParmLocal[0] / NSteps
            current_decay = max(0, 1.0 - (i*decay_ratio))
            Parms[3, i] = ParmLocal[3]*current_decay
            Parms[2, i] = ParmLocal[2]*current_decay

        RL = np.zeros((7, Nf), dtype='double', order='F')  # input/output array
        dummy = np.array(0, dtype='double')

        # calculating the emission for array distribution (array -> on)
        res = GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)
        if True:
            # retrieving the results
            f = RL[0]
            I_L = RL[5]
            I_R = RL[6]
            all_zeros = not RL.any()
            if not all_zeros:
                flux_model = I_L + I_R
                flux_model = np.nan_to_num(flux_model) + 1e-11
                logf = np.log10(f)
                logflux_model = np.log10(flux_model)
                logfreqghz = np.log10(freqghz)
                interpfunc = interpolate.interp1d(logf, logflux_model, kind='linear')
                logmflux = interpfunc(logfreqghz)
                mflux = 10. ** logmflux
                mtb = sfu2tb(np.array(freqghz) * 1.e9, mflux, area=src_area).value

                if pgplot_widget:
                    ##todo: figure out a way to update the main widget
                    import pyqtgraph as pg
                    all_items = pgplot_widget.getPlotItem().listDataItems()
                    if len(all_items) > 0:
                        pgplot_widget.removeItem(all_items[-1])
                    spec_fitplot = pgplot_widget.plot(x=np.log10(freqghz), y=np.log10(mtb),
                                                             pen=dict(color=pg.mkColor(0), width=3),
                                                             symbol=None, symbolBrush=None)
                    pgplot_widget.addItem(spec_fitplot)

                if show_plot:
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    ax1.plot(freqghz, mflux, 'k')
                    #ax1.set_xlim([1, 20])
                    ax1.set_xlabel('Frequency (GHz)')
                    ax1.set_ylabel('Flux (sfu)')
                    ax1.set_title('Flux Spectrum')
                    ax1.set_xscale('log')
                    ax1.set_yscale('log')
                    ax2.plot(freqghz, mtb, 'k')
                    #ax2.set_xlim([1, 20])
                    ax2.set_xlabel('Frequency (GHz)')
                    ax2.set_ylabel('Brightness Temperature (K)')
                    ax2.set_title('Brightness Temperature Spectrum')
                    ax2.set_xscale('log')
                    ax2.set_yscale('log')
                    ax2.legend()
                    plt.show()
            else:
                print("Calculation error! Assign an unrealistically huge number")
                mflux = np.ones_like(freqghz) * 1e4
                mtb = sfu2tb(np.array(freqghz) * 1.e9, mflux, area=src_area).value
        else:
            print("Calculation error! Assign an unrealistically huge number")
            mflux = np.ones_like(freqghz) * 1e9
            mtb = sfu2tb(np.array(freqghz) * 1.e9, mflux, area=src_area).value

        # Return values
        if spec_in_tb:
            if spec is None:
                # nothing is provided, return the model spectrum
                return mtb
            if spec_err is None:
                # no uncertainty provided, return absolute residual
                return mtb - spec
            # Return scaled residual
            return (mtb - spec) / spec_err
        else:
            if spec is None:
                # nothing is provided, return the model spectrum
                return mflux
            if spec_err is None:
                # no uncertainty provided, return absolute residual
                return mflux - spec
            # Return scaled residual
            return (mflux - spec) / spec_err

class FitThread(QThread):
    finished = pyqtSignal(object)
    def __init__(self, main_window, roi_index):
        super().__init__()
        self.main_window = main_window
        self.current_roi_idx = roi_index
        self.finished.connect(self.deleteLater)
    def run(self):
        self.main_window.current_roi_idx = self.current_roi_idx
        self.main_window.calc_roi_spec(None)
        self.main_window.update_fitmask()
        result = self.main_window.do_spec_fit()
        self.finished.emit(result)

class FitTaskSignals(QObject):
    # Signal to indicate task completion
    completed = pyqtSignal()

class FitTask(QRunnable):
    def __init__(self, main_window, roi_index):
        super().__init__()
        self.main_window = main_window
        self.current_roi_idx = roi_index
        #self.signals = FitTaskSignals()
        #self.signals = pyqtSignal()

    def run(self):
        #self.main_window.current_roi_idx = self.current_roi_idx
        self.main_window.do_spec_fit(local_roi_idx=self.current_roi_idx)
        #self.signals.completed.emit()
        #self.completed.emit()


def array_lmfit_convert_param(params, keyword, par_value=None, to_array=True):
    """
    Convert parameter attributes between lmfit.Parameter object and numpy array.

    :param params: lmfit.Parameters object containing the fitting parameters.
    :param keyword: The name of the parameter to convert or update.
    :param array: A numpy array containing the new value, minimum, and maximum of the parameter.
                  This parameter is ignored if to_array is True.
    :param to_array: Boolean flag determining the operation mode. If True, converts parameter
                     attributes to a numpy array. If False, updates parameter attributes from the array.
    :return: A numpy array containing the value, minimum, and maximum of the parameter if to_array is True.
             None if to_array is False and the operation is to update the parameter from the array.
    """
    if to_array:
        # Convert parameter to array
        if keyword in params:
            param = params[keyword]
            return np.array([param.value, param.min, param.max])
        else:
            raise ValueError(f"Parameter '{keyword}' not found in the provided lmfit.Parameters object.")
    else:
        # Update parameter from array
        if keyword in params:
            if par_value is not None:
                if 'log' in keyword:
                    params[keyword].value = np.log10(par_value)
                elif keyword == 'Bx100G':
                    params[keyword].value = par_value/100.
                else:
                    params[keyword].value = par_value
            else:
                raise ValueError("The provided array must be a three-element numpy array.")
        else:
            raise ValueError(f"Parameter '{keyword}' not found in the provided lmfit.Parameters object.")

def pyWrapper_Fit_Spectrum_Kl(ninput, rinput, parguess, freq, spec_in):
    """
    A python wrapper to call  Dr.Fleishman's Fortran code: fit_Spectrum_Kl.for/fit_Spectrum_Kl.so, all the input should be
    numpy array with dtype='float64', order='F', txt files can be created by calling get_Table().
    :param ninput: np.array([7, 0, 1, 30, 1, 1], dtype='int32'), 30 here will be replace by n_freq later. See Long_input.txt
    :param rinput: see real_input.txt, for example: np.array([0.17, 1e-6, 1.0, 4.0, 8.0, 0.015], dtype='float64')
    :param parguess: Input parameters ([guess, min, max]*15), see Parms_input.txt
    :param freq: freqs in GHz, example:    freq = np.array([3.42, 3.92, 4.42,.......], dtype='float64', order='F')
    :param spec_in: spectrum/uncertainty to be fitted, (1, n_freq, 4), spectrum:[0,:,0], uncertainty:[0,:,2]
    :return:fitted spectrum, parameters and corresponding uncerntainties.
    """
    # find the lib
    bin_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'binaries')
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        cur_lib_flie = os.path.join(bin_folder,'fit_Spectrum_Kl.so')
        if platform.machine() == 'arm64':
            cur_lib_flie = os.path.join(bin_folder,'fit_Spectrum_Kl_arm64.so')
        libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               cur_lib_flie)
    if platform.system() == 'Windows':  ##TODO: not yet tested on Windows platform
        cur_lib_flie = os.path.join(bin_folder, 'fit_Spectrum_Kl.dll')
        libname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               cur_lib_flie)
    #orgnize the input args
    n_freq = len(freq)
    ninput[3] = np.int32(n_freq)
    #create the output holder
    spec_out = np.zeros((1, n_freq, 2), dtype='float64', order='F')
    aparms = np.zeros((1, 8), dtype='float64', order='F')
    eparms = np.zeros((1, 8), dtype='float64', order='F')

    #prepare the pointer of the i/o array
    inp_arrays = [ninput, rinput, parguess, freq, spec_in, aparms, eparms, spec_out]
    ct_pointers = [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for arr in inp_arrays]
    ninput_ct, rinput_ct, parguess_ct, freq_ct, spec_in_ct, aparms_ct, eparms_ct, spec_out_ct = ct_pointers
    argv = (ctypes.POINTER(ctypes.c_double) * 8)(ninput_ct, rinput_ct, parguess_ct, freq_ct, spec_in_ct, aparms_ct,
                                                 eparms_ct, spec_out_ct)

    libc_mw = ctypes.CDLL(cur_lib_flie)
    mwfunc = libc_mw.get_mw_fit_

    res = mwfunc(ctypes.c_longlong(8), argv)
    return (spec_out, aparms, eparms)

class dummyMiniResClass:
    #pretent to be lmfit.minimizerResult for 'Dr.Fleishman' method
    def __init__(self, params):
        self.params = params