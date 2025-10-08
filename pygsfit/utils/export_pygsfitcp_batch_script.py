# export_cp_batch_script.py
# -*- coding: utf-8 -*-
"""
Create a CP-style batch script from the current pygsfit.App state.

Usage (inside pygsfit.App):
    from export_cp_batch_script import export_cp_batch_script_from_app
    export_cp_batch_script_from_app(self)
"""

import os
import json
import numpy as np

try:
    # Qt types are only needed at runtime inside pygsfit
    from PyQt5.QtWidgets import QFileDialog
except Exception:
    QFileDialog = None  # allows importing this file without Qt

# --------------------------- internal helpers ---------------------------

def _num(x, default=None):
    """Safe float conversion."""
    try:
        return float(x)
    except Exception:
        return default

def _roi_rect_world(roi):
    """
    Return ROI rect in world coords [ [x0,y0], [x1,y1] ].
    Assumes pg.RectROI (or compatible) and that the image view uses world axes.
    """
    # PyQtGraph ROI returns pos() and size() in plot coordinates
    p = roi.pos()
    s = roi.size()
    x0, y0 = _num(p[0]), _num(p[1])
    w,  h  = _num(s[0]), _num(s[1])
    if None in (x0, y0, w, h):
        raise RuntimeError("ROI does not return numeric world coordinates.")
    return [[x0, y0], [x0 + w, y0 + h]]

def _normalize_box_to_unit(box_xy, x_full, y_full):
    """
    Convert a world box [[x0,y0],[x1,y1]] into unit fractions [[fx0,fy0],[fx1,fy1]]
    relative to full extents x_full=(x0,x1), y_full=(y0,y1).
    """
    (X0, X1), (Y0, Y1) = x_full, y_full
    (x0, y0), (x1, y1) = box_xy
    dx = (X1 - X0) if X1 != X0 else 1.0
    dy = (Y1 - Y0) if Y1 != Y0 else 1.0
    fx0 = (x0 - X0) / dx
    fx1 = (x1 - X0) / dx
    fy0 = (y0 - Y0) / dy
    fy1 = (y1 - Y0) / dy
    # clamp to [0,1] for safety
    clamp = lambda v: max(0.0, min(1.0, v))
    return [[clamp(fx0), clamp(fy0)], [clamp(fx1), clamp(fy1)]]

def _lmfit_param_triplet(params, name, *, log10=False, scale=1.0, default):
    """
    Extract (guess, min, max) from lmfit.Parameters.
    - If log10=True, interpret value/min/max as log10(value) and convert back.
    - 'scale' divides the result (e.g., 1e7 for n_nth, 1e9 for n_th).
    - If the parameter doesn't exist, return the provided default triplet.
    """
    if params is None or name not in params:
        return default
    p = params[name]
    def _val(v):
        if v is None:
            return None
        return (10.0 ** v) if log10 else float(v)
    g = _val(p.value)
    lo = _val(p.min) if p.min is not None else g
    hi = _val(p.max) if p.max is not None else g
    # rescale (e.g. convert cm^-3 to "×1e7" units)
    g = g / scale if g is not None else None
    lo = lo / scale if lo is not None else None
    hi = hi / scale if hi is not None else None
    # Fill Nones with guess to keep CP happy
    g = g if g is not None else (default[0] if default else 1.0)
    lo = lo if lo is not None else g
    hi = hi if hi is not None else g
    return [g, lo, hi]

def _build_cp_initial_parguess(app):
    """
    Map pygsfit params -> CP parameter table in this order:
    [ n_nth(×1e7), Bx100G, theta(deg), n_th(×1e9), delta, E_max(MeV), T_e(MK) ]
    Returns a 2D Python list ready for np.array(..., dtype=np.float64).
    """
    P = getattr(app, 'fit_params', None)

    # Sensible defaults if some params are missing
    #            guess,    min,    max
    d_nnth  = [  10.0,  0.0001, 2000.0]  # ×1e7  (=> 1e8 cm^-3)
    d_Bx100 = [   2.0,   0.01,   100.0]  # B [×100 G]
    d_theta = [  45.0,    0.01,   89.9]  # deg
    d_nth   = [  10.0,   0.01,   600.0]  # ×1e9
    d_delta = [   4.0,    1.0,    30.0]
    d_emax  = [  10.0,   0.05,   100.0]  # MeV
    d_Te    = [   5.0,    0.1,   100.0]  # MK

    # Pull from lmfit where available
    nnth  = _lmfit_param_triplet(P, 'log_nnth', log10=True,  scale=1e7, default=d_nnth)
    bx100 = _lmfit_param_triplet(P, 'Bx100G',   log10=False, scale=1.0, default=d_Bx100)
    theta = _lmfit_param_triplet(P, 'theta',    log10=False, scale=1.0, default=d_theta)
    nth   = _lmfit_param_triplet(P, 'log_nth',  log10=True,  scale=1e9, default=d_nth)
    delta = _lmfit_param_triplet(P, 'delta',    log10=False, scale=1.0, default=d_delta)
    emax  = _lmfit_param_triplet(P, 'Emax_MeV', log10=False, scale=1.0, default=d_emax)
    Te    = _lmfit_param_triplet(P, 'T_MK',     log10=False, scale=1.0, default=d_Te)

    parguess = [nnth, bx100, theta, nth, delta, emax, Te]
    return parguess

# --------------------------- main entry point ---------------------------

def export_cp_batch_script_from_app(app):
    """
    Create a CP-style batch script from the running pygsfit.App state.

    What it does:
      * Prompts for an output directory (Qt dialog).
      * Derives inp_fov from the FIRST ROI (world arcsec).
      * Derives background_xyrange from the background ROI (unit fractions); falls back to default if absent.
      * Reads fit_freq_bound (GHz) and writes start/end in Hz for CP code.
      * Maps your lmfit Parameters to CP initial_parguess order with unit conversions.
      * Writes a ready-to-run Python script that loops over your mapcube list and calls pygsfit_cp.

    The generated script depends on package `pygsfit_cp`
    (`pygsfit_cp_src`, `pygsfit_cp_utils`, and `results_convertor`).
    """
    if not getattr(app, 'has_eovsamap', False):
        raise RuntimeError("EOVSA map is not loaded in pygsfit.App (has_eovsamap=False).")

    # 1) Collect the mapcube list
    if getattr(app, 'eoimg_time_seq', None):
        mapcubes = list(app.eoimg_time_seq)
    else:
        mapcubes = [app.eoimg_fname] if getattr(app, 'eoimg_fname', None) else []
    if not mapcubes:
        raise RuntimeError("No EOVSA spectral image FITS files found.")

    # 2) FIRST ROI -> inp_fov (world arcsec)
    roi0 = None
    if getattr(app, 'rois', None) and len(app.rois) and len(app.rois[0]):
        roi0 = app.rois[0][0]
    elif getattr(app, 'grid_rois', None) and len(app.grid_rois) and len(app.grid_rois[0]):
        roi0 = app.grid_rois[0][0]
    if roi0 is None:
        raise RuntimeError("No ROI found. Draw at least one ROI to define inp_fov.")
    inp_fov = _roi_rect_world(roi0)  # [[x0,y0],[x1,y1]] in arcsec

    # 3) Background box -> background_xyrange (unit fractions)
    if getattr(app, 'has_bkg', False) and getattr(app, 'bkg_roi', None) is not None:
        bkg_box_world = _roi_rect_world(app.bkg_roi)
        x_full = (float(app.x0), float(app.x1))
        y_full = (float(app.y0), float(app.y1))
        background_xyrange = _normalize_box_to_unit(bkg_box_world, x_full, y_full)
    else:
        # default used in your CP source
        background_xyrange = [[0.15, 0.25], [0.75, 0.85]]

    # 4) Frequency range from GUI (GHz -> Hz)
    fit_fmin_ghz, fit_fmax_ghz = getattr(app, 'fit_freq_bound', [1.0, 18.0])
    start_freq_hz = float(fit_fmin_ghz) * 1e9
    end_freq_hz   = float(fit_fmax_ghz) * 1e9

    # 5) Initial parameter guesses/bounds for CP
    parguess_list = _build_cp_initial_parguess(app)
    nparms = len(parguess_list)

    # 6) Default ninput / rinput (will set ninput[0] == nparms in the script)
    #    Notes:
    #      ninput = [Nparms, Angular_mode, Npix, Nfreq(placeholder), fitting_mode, stokes]
    #      rinput = [beam_w, beam_amp, ?, ?, ?, ?]  (keep your CP defaults)
    ninput_repr = f"[{nparms}, 0, 1, 30, 1, 1]"
    rinput_repr = "[0.17, 1e-6, 1.0, 4.0, 8.0, 0.015]"

    # 7) Other CP controls
    rms_factor = 1.0
    integrated_threshold_sfu = 1.0  # conservative threshold; change in GUI later if you prefer

    # 8) Ask for the CP output directory (Qt)
    out_dir = None
    if QFileDialog is not None:
        out_dir = QFileDialog.getExistingDirectory(
            app, "Select CP output directory", os.path.expanduser("~")
        )
    if not out_dir:
        # user cancel → use HOME/pygsfit_cp_output
        out_dir = os.path.join(os.path.expanduser("~"), "pygsfit_cp_output")

    # 9) Ask where to save the batch script
    script_path = None
    if QFileDialog is not None:
        fname, _ = QFileDialog.getSaveFileName(
            app,
            "Save CP Batch Script",
            os.path.join(out_dir, "pygsfit_cp_batch.py"),
            "Python Files (*.py)"
        )
        script_path = fname if fname else None
    if not script_path:
        script_path = os.path.join(out_dir, "pygsfit_cp_batch.py")

    # 10) Compose script text
    # We keep it minimal and robust; it mirrors your exported_notebook_cells.py flow.
    lines = []
    ap = lambda s="": lines.append(s + "\n")

    ap("# -*- coding: utf-8 -*-")
    ap("# Auto-generated by pygsfit → CP exporter")
    ap("import os")
    ap("import sys")
    ap("import numpy as np")
    ap("import pkg_resources")
    ap("from pygsfit_cp.pygsfit_cp_src import pygsfit_cp_class")
    ap("from pygsfit_cp.pygsfit_cp_utils import get_fixed_fov_mask, combine_hdf5_files")
    ap("import pygsfit_cp.results_convertor as rc")
    ap("")
    ap("def main():")
    ap("    package_path = pkg_resources.resource_filename('pygsfit_cp', '')")
    #ap(f"    out_dir = r'''{out_dir}'''")
    ap(f"    out_dir = f'{out_dir}'")
    ap("    os.makedirs(out_dir, exist_ok=True)")
    # mapcubes list
    ap("    eovsa_map_cubes = []")
    for f in mapcubes:
        #ap(f"    eovsa_map_cubes.append(r'''{f}''')")
        ap(f"    eovsa_map_cubes.append(f'{f}')")
    ap("    if not eovsa_map_cubes:")
    ap("        print('No mapcube files found.'); sys.exit(1)")
    # inp_fov
    ap(f"    inp_fov = {json.dumps(inp_fov)}  # world arcsec [[x0,y0],[x1,y1]] from first ROI")
    ap("    cur_fixed_mask = get_fixed_fov_mask(eovsa_map_cubes[0], inp_fov)")
    ap("")
    # loop
    ap("    map_files_by_frame = []")
    ap("    fitting_files_by_frame = []")
    ap("    for ceomap in eovsa_map_cubes:")
    ap("        gs = pygsfit_cp_class(filename=ceomap, out_dir=out_dir)")
    ap("        gs.pix_mask = cur_fixed_mask")
    ap(f"        gs.background_xyrange = {json.dumps(background_xyrange)}")
    ap(f"        gs.rms_factor = {rms_factor}")
    ap("        gs.update_rms()")
    #ap(f"        gs.integrated_threshold_sfu = {integrated_threshold_sfu}")
    ap(f"        gs.set_integrated_threshold_sfu({integrated_threshold_sfu})")
    ap("        gs.update_flux_threshold_mask()")
    ap(f"        gs.start_freq = {start_freq_hz:.6g}")  # Hz
    ap(f"        gs.end_freq   = {end_freq_hz:.6g}")    # Hz
    ap(f"        gs.ninput = np.array({ninput_repr}, dtype='int32')")
    ap(f"        gs.rinput = np.array({rinput_repr}, dtype='float64')")
    ap("        # *guess, *min, *max  per row")
    ap(f"        gs.initial_parguess = np.array({json.dumps(parguess_list)}, dtype=np.float64)")
    ap("        res = gs.do_fit(mode='batch')")
    ap("        # Prefer the class-provided output filename if available")
    ap("        if isinstance(res, dict) and 'out_file' in res:")
    ap("            cur_save_file = res['out_file']")
    ap("        else:")
    ap("            base = os.path.splitext(os.path.basename(ceomap))[0]")
    ap("            cur_save_file = os.path.join(out_dir, f\"{base}.h5\")")
    ap("        cur_map_file = rc.create_params_map(cur_save_file)")
    ap("        map_files_by_frame.append(cur_map_file)")
    ap("        fitting_files_by_frame.append(cur_save_file)")
    ap("")
    ap("    # Combine per-frame HDF5 into a single file")
    ap("    fitting_res_file = os.path.join(out_dir, 'fitting_results.h5')")
    ap("    combine_hdf5_files(file_list=fitting_files_by_frame, output_file=fitting_res_file)")
    ap("    print('All results saved to:', fitting_res_file)")
    ap("    print('Map files:', map_files_by_frame)")
    ap("")
    ap("if __name__ == '__main__':")
    ap("    main()")
    ap("")

    # 11) Write the script
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 12) Tell the caller where it went
    print(f"CP batch script written to: {script_path}")
    return script_path
