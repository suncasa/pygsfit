import copy
import os
import sys

import astropy
import astropy.units as u
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pyqtgraph as pg
import sunpy
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from astropy import wcs
from astropy.io import fits
from astropy.time import Time, TimeDelta
from matplotlib import patches
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sunpy import map as smap
import h5py
import threading
from tqdm import *
import multiprocessing
import tempfile
import glob
# import re
# import time
# from astropy.coordinates import SkyCoord
# import datetime
# import dill


filedir = os.path.dirname(os.path.realpath(__file__))
print(filedir)
sys.path.append(filedir)
from utils import gstools, roiutils, ndfits
from utils.roiutils import PolyLineROIX, Grid_Dialog
import warnings

warnings.filterwarnings("ignore")
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(imageAxisOrder='row-major')

SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fit_param_text = {'Bx100G': 'B [\u00d7100 G]',
                  'log_nnth': 'log(n<sub>nth</sub>) [cm<sup>-3</sup>]',
                  'delta': '\u03b4',
                  'Emin_keV': 'E<sub>min</sub> [keV]',
                  'Emax_MeV': 'E<sub>max</sub> [MeV]',
                  'theta': '\u03b8 [degree]',
                  'log_nth': 'log(n<sub>th</sub>) [cm<sup>-3</sup>]',
                  'T_MK': 'T [MK]',
                  'depth_asec': 'depth [arcsec]',
                  'area_asec2': 'area [arcsec<sup>2</sup>]'}


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.eoimg_fname = '<Select or enter a valid EOVSA image fits file name>'
        self.eodspec_fname = '<Select or enter a valid EOVSA spectrogram fits file name>'
        self.aiafname = '<Select or enter a valid AIA fits file name>'
        self.eoimg_time_seq = None
        self.cur_frame_idx = 0
        self.data_in_seq = False
        # self.eoimg_fitsentry = QLineEdit()
        # self.eodspec_fitsentry = QLineEdit()
        self.title = 'pygsfit'
        self.left = 0
        self.top = 0
        self.width = 1600
        self.height = 1000
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self._main = QWidget()
        self.setCentralWidget(self._main)
        self.fit_method = 'nelder'
        self.fit_params = lmfit.Parameters()
        self.fit_params.add_many(('Bx100G', 2., True, 0.1, 100., None, None),
                                 ('log_nnth', 5., True, 3., 11, None, None),
                                 ('delta', 4., True, 1., 30., None, None),
                                 ('Emin_keV', 10., False, 1., 100., None, None),
                                 ('Emax_MeV', 10., False, 0.05, 100., None, None),
                                 ('theta', 45., True, 0.01, 89.9, None, None),
                                 ('log_nth', 10, True, 4., 13., None, None),
                                 ('T_MK', 1., False, 0.1, 100, None, None),
                                 ('depth_asec', 5., False, 1., 100., None, None))
        self.fit_params_nvarys = 5
        self.fit_kws = {'maxiter': 2000, 'xatol': 0.01, 'fatol': 0.01}
        self.fit_function = gstools.GSCostFunctions.SinglePowerLawMinimizerOneSrc
        self.threadpool = QThreadPool()
        self.fit_threads = []
        self.has_eovsamap = False
        self.has_dspec = False
        self.has_stokes = False
        self.has_aiamap = False
        self.has_bkg = False
        self.has_rois = False
        self.has_grid_rois = False
        self.fbar = None
        self.data_freq_bound = [1.0, 18.0]  # Initial Frequency Bound of the instrument
        self.tb_spec_bound = [1e3, 5e9]  # Bound of brightness temperature; the lower bound is set to the fit limit
        self.flx_spec_bound = [1e-4, 1e5]  # Bound of flux density; the lower bound is set to the fit limit
        self.fit_freq_bound = [1.0, 18.0]
        self.roi_freq_bound = [1.0, 18.0]
        self.spec_frac_err = 0.1  # fractional error of the intensity (assumed to be due to flux calibration error)
        self.spec_rmsplots = []
        self.spec_dataplots = []
        self.spec_dataplots_tofit = []
        self.ele_dist = 'powerlaw'
        self.roi_grid_size = 2
        self.rois = [[]]
        self.grid_rois = []
        self.vis_roi = True
        self.roi_group_idx = 0
        self.nroi_current_group = 0
        self.current_roi_idx = 0
        self.number_grid_rois = 0
        self.pixelized_grid_rois = None
        self.distSpecCanvasSet = {}
        self.pol_select_idx = 0
        self.spec_in_tb = True
        self.is_calibrated_tp = True
        self.qlookimg_axs = None
        self.qlookdspec_ax = None
        #for signal slots:
        #self.update_roi_index.connect(self.set_roi_index)
        # some controls for qlookplot
        self.opencontour = True
        self.clevels = np.array([0.7])
        self.calpha = 1.
        self.pgcmap = self._create_pgcmap(cmap='viridis', ncolorstop=6)
        self.savedata = False  ### Surajit edit
        self.update_gui = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_tasks_completion)
        self.timer.setInterval(2000)
        # initialize the window
        # self.initUI()
        self.initUItab_explorer()
        # ## quick input for debug --------------
        # self.eoimg_file_select()
        # ## quick input for debug --------------
        #self.eoimg_fname = '/Users/walterwei/Downloads/20220511/slf_final_XX_t19_allbd.fits'
        #self.eoimg_file_select_return()

    def _create_pgcmap(self, cmap='viridis', ncolorstop=6):
        """This is to create the cmap for pyqtgraph's ImageView Widgets"""
        from matplotlib import cm
        mpl_cmap = cm.get_cmap(cmap, ncolorstop)
        mpl_colors = mpl_cmap(np.linspace(0, 1, ncolorstop)) * 255
        colors = []
        for s in range(ncolorstop):
            colors.append((int(mpl_colors[s, 0]), int(mpl_colors[s, 1]), int(mpl_colors[s, 2])))

        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, ncolorstop), color=colors)
        return cmap

    def _createToolBars(self):
        # Using a title
        # fileToolBar = self.addToolBar("File")
        iconsize = QSize(50, 50)
        selectToolBar = self.addToolBar("ADD")
        self.addSelectAction = QAction(QIcon("{}/resources/add-button.svg".format(filedir)), "&Add ROI", self)
        self.addSelectAction.setText('Add ROI')
        self.addSelectAction.triggered.connect(self.add_new_roi)
        selectToolBar.addAction(self.addSelectAction)

        ## define ROI toolbar
        roiToolBar = self.addToolBar("ROI")

        self.rectButton = QToolButton(self, text="RectROI", checkable=True)
        self.rectButton.setIcon(QIcon("{}/resources/roi-rect.svg".format(filedir)))
        self.rectButton.setChecked(True)
        self.rectButton.setIconSize(iconsize)
        self.rectButton.setToolTip('Rectangle ROI tool')
        self.rectButton.setStyleSheet("QToolButton::hover"
                                      "{"
                                      "background-color : #555555;"
                                      "}")

        self.elpsButton = QToolButton(self, text="EllipseROI", checkable=True)
        self.elpsButton.setIcon(QIcon("{}/resources/roi-ellipse.svg".format(filedir)))
        self.elpsButton.setIconSize(iconsize)
        self.elpsButton.setToolTip('Ellipse ROI tool')
        self.elpsButton.setStyleSheet("QToolButton::hover"
                                      "{"
                                      "background-color : #555555;"
                                      "}")

        self.polygonButton = QToolButton(self, text="PolygonROI", checkable=True)
        self.polygonButton.setIcon(QIcon("{}/resources/roi-polygon.svg".format(filedir)))
        self.polygonButton.setIconSize(iconsize)
        self.polygonButton.setToolTip('Polygon ROI tool')
        self.polygonButton.setStyleSheet("QToolButton::hover"
                                         "{"
                                         "background-color : #555555;"
                                         "}")
        ## define Slice toolbar
        sliceToolBar = self.addToolBar("sliceROI")

        self.lineButton = QToolButton(self, text="LineSegmentROI", checkable=True)
        self.lineButton.setIcon(QIcon("{}/resources/slice-line.svg".format(filedir)))
        self.lineButton.setIconSize(iconsize)
        self.lineButton.setToolTip('Line slice ROI tool')
        self.lineButton.setStyleSheet("QToolButton::hover"
                                      "{"
                                      "background-color : #555555;"
                                      "}")

        self.polyLineButton = QToolButton(self, text="PolyLineROI", checkable=True)
        self.polyLineButton.setIcon(QIcon("{}/resources/slice-polyline.svg".format(filedir)))
        self.polyLineButton.setIconSize(iconsize)
        self.polyLineButton.setToolTip('PolyLine slice ROI tool')
        self.polyLineButton.setStyleSheet("QToolButton::hover"
                                          "{"
                                          "background-color : #555555;"
                                          "}")

        # self.multiRectButton = QToolButton(self, text="MultiRectROI", checkable=True)
        # self.multiRectButton.setIcon(QIcon("{}/resources/slice-line.svg".format(filedir)))
        # self.multiRectButton.setIconSize(iconsize)
        # self.multiRectButton.setToolTip('PolyLine slice tool')
        # self.multiRectButton.setStyleSheet("QToolButton::hover"
        #                                    "{"
        #                                    "background-color : #555555;"
        #                                    "}")

        self.toolBarButtonGroup = QButtonGroup(self, exclusive=True)
        for button in [
            self.rectButton,
            self.elpsButton,
            self.polygonButton
        ]:
            roiToolBar.addWidget(button)
            self.toolBarButtonGroup.addButton(button)
        # for button in [self.lineButton, self.polyLineButton, self.multiRectButton]:
        for button in [self.lineButton, self.polyLineButton]:
            sliceToolBar.addWidget(button)
            self.toolBarButtonGroup.addButton(button)

        self.add2slice = QCheckBox('Add ROI to slice?')
        self.add2slice.setChecked(False)
        self.add2slice.setToolTip(
            'If True, the selected ROI will be added to the last slice ROI. If there no slice ROI exists, use the sliceROI tools on the right to create one.')
        roiToolBar.addWidget(self.add2slice)
        # self.add2slice.toggled.connect(self.is_calibrated_tp_state)

        # self.addSelectAction.set(True)
        # self.rectAction = QAction(QIcon("{}/resources/roi-rect.svg".format(filedir)), "&Rectangle", self)
        # self.rectAction.setCheckable(True)
        # self.circAction = QAction(QIcon("{}/resources/roi-circ.svg".format(filedir)), "&Circle", self)
        # self.circAction.setCheckable(True)
        # self.polylineAction = QAction(QIcon("{}/resources/roi-polyline.svg".format(filedir)), "&PolyLine", self)
        # self.polylineAction.setCheckable(True)
        # roiToolBar.addAction(self.rectAction)
        # roiToolBar.addAction(self.circAction)
        # roiToolBar.addAction(self.polylineAction)
        # # # Using a QToolBar object
        # # editToolBar = QToolBar("Edit", self)
        # # self.addToolBar(editToolBar)
        # # # Using a QToolBar object and a toolbar area
        # # helpToolBar = QToolBar("Image", self)
        # # self.addToolBar(Qt.TopToolBarArea, helpToolBar)

    def _createMenuBar(self):
        menubar = self.menuBar()
        ## PYQT5 does not support native menubar on MacOS
        menubar.setNativeMenuBar(False)
        # fileMenu = QMenu("&File", self)
        # menubar.addMenu(fileMenu)
        actionFile = menubar.addMenu("File")

        action_loadEOVSAimage = QAction("Load EOVSA Image", self)
        action_loadEOVSAimage.triggered.connect(self.eoimg_file_select)
        actionFile.addAction(action_loadEOVSAimage)
        action_loadAIA = QAction("Load AIA", self)
        action_loadAIA.triggered.connect(self.aiafile_select)
        actionFile.addAction(action_loadAIA)
        action_loadEOVSAspectrogram = QAction("Load EOVSA Spectrogram", self)
        action_loadEOVSAspectrogram.triggered.connect(self.eodspec_file_select)
        actionFile.addAction(action_loadEOVSAspectrogram)
        action_exportBatchScript = QAction("Export Batch Script", self)
        action_exportBatchScript.triggered.connect(self.export_batch_script)
        actionFile.addAction(action_exportBatchScript)
        # actionFile.addAction("Open AIA")
        # actionFile.addAction("Open EOVSA Spectrogram")
        actionFile.addSeparator()
        actionFile.addAction("Quit")

        # Creating menus using a title
        editMenu = menubar.addMenu("&Edit")
        paramsMenu = menubar.addMenu("&Params")
        action_RestoInit = QAction("Result to Initial", self)
        action_RestoInit.triggered.connect(self.update_init_guess)
        paramsMenu.addAction(action_RestoInit)
        # helpMenu = menuBar.addMenu(" &Help")
        # self.menu_layout.addWidget(menuBar)

    # def initUI(self):
    #     self.statusBar = QStatusBar()
    #     self.setStatusBar(self.statusBar)
    #     self.progressBar = QProgressBar()
    #     self.progressBar.setGeometry(10, 10, 200, 15)
    #
    #     layout = QVBoxLayout()
    #     # Initialize tab screen
    #     self.tabs = QTabWidget()
    #     tab_explorer = QWidget()
    #     tab_fit = QWidget()
    #     tab_analyzer = QWidget()
    #
    #     # Add tabs
    #     self.tabs.addTab(tab_explorer, "Explorer")
    #     # self.tabs.addTab(tab_fit, "Fit")
    #     self.tabs.addTab(tab_analyzer, "Analyzer")
    #
    #     # Each tab's user interface is complex, so this splits them into separate functions.
    #     self.initUItab_explorer()
    #     # self.initUItab_fit()
    #     self.initUItab_analyzer()
    #
    #     # self.tabs.currentChanged.connect(self.tabChanged)
    #
    #     # Add tabs to widget
    #     layout.addWidget(self.tabs)
    #     self._main.setLayout(layout)
    #
    #     self.show()

    # Explorer Tab User Interface
    #
    def initUItab_explorer(self):
        # Create main layout (a Vertical Layout)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(10, 10, 200, 15)
        self._createMenuBar()
        self._createToolBars()
        main_layout = QHBoxLayout()

        # Creat Data Display and Fit Tab
        data_layout = QVBoxLayout()
        fit_layout = QVBoxLayout()

        ###### The Following is for datalayout ######
        # Upper box of the data layout has two hboxes: top box and quicklook plot box
        data_layout_upperbox = QVBoxLayout()
        data_layout.addLayout(data_layout_upperbox)

        # lowerbox
        data_layout_lowerbox = QGridLayout()  # lower box has two hboxes: left for multi-panel display and right for spectrum
        pgimgbox = QVBoxLayout()  # left box of lower box for pg ImageView and associated buttons

        # Status box
        pg_img_status_box = QVBoxLayout()
        self.pg_img_mouse_pos_widget = QLabel("")
        pg_img_status_box.addWidget(self.pg_img_mouse_pos_widget)
        self.pg_img_roi_info_widget = QLabel("")
        # pg_img_status_box.addWidget(self.pg_img_roi_info_widget)
        self.pg_img_bkg_roi_info_widget = QLabel("")
        pg_img_status_box.addWidget(self.pg_img_bkg_roi_info_widget)

        # Add plotting area for multi-panel EOVSA images
        self.pg_img_plot = pg.PlotItem(labels={'bottom': ('Solar X [arcsec]', ''), 'left': ('Solar Y [arcsec]', '')})
        self.pg_img_canvas = pg.ImageView(name='EOVSA Explorer', view=self.pg_img_plot)
        #self.pg_img_canvas_wrapper = CustomImageViewer(wrapped_image_view = self.pg_img_canvas)

        pgimgbox.addWidget(self.pg_img_canvas)
        #pgimgbox.addWidget(self.pg_img_canvas_wrapper)
        # pgimgbox.addLayout(pgbuttonbox)
        pgimgbox.addLayout(pg_img_status_box)
        data_layout_lowerbox.addLayout(pgimgbox, 0, 0)
        data_layout_lowerbox.setColumnStretch(0, 2)
        self.pg_img_canvas.sigTimeChanged.connect(self.update_fbar)

        # Add eoimg seq selection slider (will only show up when more SIMILAR fits files are detect)
        eoimg_seq_slider_label_layout = QHBoxLayout()
        self.eoimg_seq_slider = QSlider(Qt.Horizontal)
        eoimg_seq_slider_label_layout.addWidget(self.eoimg_seq_slider)  # Add the slider to the horizontal layout
        self.eoimg_seq_label = QLabel("Time: ")
        eoimg_seq_slider_label_layout.addWidget(self.eoimg_seq_label)  # Add the label to the horizontal layout
        self.eoimg_seq_slider_label_widget = QWidget()
        self.eoimg_seq_slider_label_widget.setLayout(eoimg_seq_slider_label_layout)
        self.eoimg_seq_slider.setMinimum(0)
        self.eoimg_seq_slider.setMaximum(1)
        self.eoimg_seq_slider.setTickPosition(QSlider.TicksBelow)
        self.eoimg_seq_slider.setValue(0)
        self.eoimg_seq_slider_label_widget.setVisible(False)  # Initially hide the widget
        self.eoimg_seq_slider.valueChanged.connect(self.time_slider_slided)
        self.eoimg_seq_slider.sliderReleased.connect(self.time_slider_released)
        pgimgbox.insertWidget(0, self.eoimg_seq_slider_label_widget)

        #Display the selectde time
        self.text_box_time = QLabel()
        self.text_box_time.setText('')
        eoimg_seq_slider_label_layout.addWidget(self.text_box_time)


        # right box for spectral plots
        self.specplotarea = QVBoxLayout()
        # self.speccanvas = FigureCanvas(Figure(figsize=(4, 6)))
        self.speccanvas = pg.PlotWidget()
        # self.spectoolbar = NavigationToolbar(self.speccanvas, self)
        # self.specplotarea.addWidget(self.spectoolbar)
        self.specplotarea.addWidget(self.speccanvas)
        # self.spec_axs = self.speccanvas.figure.subplots(nrows=1, ncols=1)
        data_layout_lowerbox.addLayout(self.specplotarea, 0, 1)
        data_layout_lowerbox.setColumnStretch(1, 1)

        # add a toggle between Tb and flux density
        specplotmode_box = QVBoxLayout()
        tb_flx_button_group = QButtonGroup(self)
        self.plot_tb_button = QRadioButton("Brightness Temperature", self)
        self.plot_tb_button.toggled.connect(self.tb_flx_btnstate)
        self.plot_flx_button = QRadioButton("Flux Density", self)
        self.plot_flx_button.toggled.connect(self.tb_flx_btnstate)
        tb_flx_button_group.addButton(self.plot_tb_button)
        tb_flx_button_group.addButton(self.plot_flx_button)
        self.plot_tb_button.setChecked(True)
        # self.plot_flx_button.setChecked(False)
        specplotmode_box.addWidget(self.plot_tb_button)
        specplotmode_box.addWidget(self.plot_flx_button)
        self.specplotarea.addLayout(specplotmode_box)

        data_layout.addLayout(data_layout_lowerbox)

        data_layout_lowerbox2 = QVBoxLayout()
        # Create a button to toggle the qlookimg box.
        qlookimglabel = QLabel("solar Image")
        self.qlookimgbutton = QToolButton()
        self.qlookimgbutton.setArrowType(Qt.RightArrow)
        self.qlookimgbutton.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.qlookimgbutton.setFixedSize(20, 20)
        self.qlookimgbutton.setCheckable(True)
        self.qlookimgbutton.toggled.connect(self.showqlookimg)

        # Create a button to toggle the qlookspec box.
        qlookdspeclabel = QLabel("Spectrogram")
        self.qlookdspecbutton = QToolButton()
        self.qlookdspecbutton.setArrowType(Qt.RightArrow)
        self.qlookdspecbutton.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.qlookdspecbutton.setFixedSize(20, 20)
        self.qlookdspecbutton.setCheckable(True)
        self.qlookdspecbutton.setChecked(False)
        self.qlookdspecbutton.toggled.connect(self.showqlookdspec)

        self.is_calibrated_tp_button = QCheckBox('Is Calibrated TP?')
        self.is_calibrated_tp_button.setChecked(self.is_calibrated_tp)
        self.is_calibrated_tp_button.toggled.connect(self.is_calibrated_tp_state)

        ### Surajit: use a button to toogle for saving the fitted spectrum and the data underneath it.
        self.savedata_button = QCheckBox('Save data?')
        self.savedata_button.setChecked(self.savedata)
        self.savedata_button.toggled.connect(self.savedata_state)

        qlookbuttonbox = QHBoxLayout()
        qlookbuttonbox_l = QHBoxLayout()
        qlookbuttonbox_l.addWidget(self.qlookimgbutton)
        qlookbuttonbox_l.addWidget(qlookimglabel)
        qlookbuttonbox_r = QHBoxLayout()
        qlookbuttonbox_r.addWidget(self.qlookdspecbutton)
        qlookbuttonbox_r.addWidget(qlookdspeclabel)
        qlookbuttonbox_r.addWidget(self.is_calibrated_tp_button)
        qlookbuttonbox_r.addWidget(self.savedata_button)  ## Surajit
        qlookbuttonbox.addLayout(qlookbuttonbox_l)
        qlookbuttonbox.addLayout(qlookbuttonbox_r)
        qlookbuttonbox.setStretch(0, 1)
        qlookbuttonbox.setStretch(1, 2)
        data_layout_lowerbox2.addLayout(qlookbuttonbox)

        # Bottom of the upper box in data layout: quick look plotting area
        qlookarea = QHBoxLayout()

        self.qlookimgbox = QVBoxLayout()
        self.qlookdspecbox = QVBoxLayout()
        self.qlookdummybox = QVBoxLayout()

        dummy_spacer = QLabel('')
        self.qlookdummybox.addWidget(dummy_spacer)
        qlookarea.addLayout(self.qlookimgbox)
        qlookarea.addLayout(self.qlookdspecbox)
        qlookarea.addLayout(self.qlookdummybox)

        data_layout_lowerbox2.addLayout(qlookarea)
        data_layout.addLayout(data_layout_lowerbox2)
        # data_layout.setRowStretch(0, 1.)

        # data_layout.addLayout(data_layout_bottombox)
        main_layout.addLayout(data_layout)

        ####### The following is for fit layout on the right of the main window ######
        # Group 1: ROI Definition Group
        roi_definition_group = QGroupBox("ROI Definition")
        roi_definition_group_box = QVBoxLayout()
        roi_definition_group.setLayout(roi_definition_group_box)

        # ADD ROI Region for Obtaining Spectra
        roi_button_box = QHBoxLayout()
        self.add_roi_button = QPushButton("Add New ROI")
        self.add_roi_button.setStyleSheet("background-color : lightgrey")
        roi_button_box.addWidget(self.add_roi_button)
        self.add_roi_button.clicked.connect(self.add_new_roi)

        self.add_to_roigroup_button = QToolButton(self)
        self.add_to_roigroup_button.setText("0")
        self.add_to_roigroup_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.add_to_roigroup_button.setMenu(QMenu(self.add_to_roigroup_button))
        self.add_to_roigroup_widget = QListWidget()
        self.add_to_roigroup_widget.addItems([str(i) for i in range(len(self.rois)+1)])
        #self.add_to_roigroup_widget.addItems(['0', '1', '2'])  ##todo: update ROI group # on the fly
        self.add_to_roigroup_widget.itemClicked.connect(self.add_to_roigroup_selection)
        action = QWidgetAction(self.add_to_roigroup_button)
        action.setDefaultWidget(self.add_to_roigroup_widget)
        self.add_to_roigroup_button.menu().addAction(action)
        roi_button_box.addWidget(QLabel("to Group"))
        roi_button_box.addWidget(self.add_to_roigroup_button)

        self.roi_freq_lowbound_selector = QDoubleSpinBox()
        self.roi_freq_lowbound_selector.setDecimals(1)
        self.roi_freq_lowbound_selector.setRange(self.data_freq_bound[0], self.data_freq_bound[1])
        self.roi_freq_lowbound_selector.setSingleStep(0.1)
        self.roi_freq_lowbound_selector.setValue(self.data_freq_bound[0])
        self.roi_freq_lowbound_selector.valueChanged.connect(self.roi_freq_lowbound_valuechange)
        roi_button_box.addWidget(QLabel("Min Freq (GHz)"))
        roi_button_box.addWidget(self.roi_freq_lowbound_selector)

        self.roi_freq_hibound_selector = QDoubleSpinBox()
        self.roi_freq_hibound_selector.setDecimals(1)
        self.roi_freq_hibound_selector.setRange(self.data_freq_bound[0], self.data_freq_bound[1])
        self.roi_freq_hibound_selector.setSingleStep(0.1)
        self.roi_freq_hibound_selector.setValue(self.data_freq_bound[1])
        self.roi_freq_hibound_selector.valueChanged.connect(self.roi_freq_hibound_valuechange)
        roi_button_box.addWidget(QLabel("Max Freq (GHz)"))
        roi_button_box.addWidget(self.roi_freq_hibound_selector)

        # ADD presets selection box
        self.add_manual_rois_button = QToolButton()
        self.add_manual_rois_button.setText('Define ROIs')
        self.add_manual_rois_button.clicked.connect(self.add_manually_defined_rois)
        self.add_manual_rois_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.group_roi_op_menu = QMenu()
        self.group_roi_op_menu.addAction('Save Group', self.group_roi_op_selector)
        self.group_roi_op_menu.addAction('Load Group', self.group_roi_op_selector)
        self.add_manual_rois_button.setMenu(self.group_roi_op_menu)
        roi_button_box.addWidget(self.add_manual_rois_button)
        roi_definition_group_box.addLayout(roi_button_box)

        roi_grid_box = QHBoxLayout()
        ## todo: add this button to draw a box and define a group of ROIs with a given size
        self.add_roi_grid_button = QPushButton("Add ROI Grid")
        self.add_roi_grid_button.setStyleSheet("background-color : lightgrey")
        roi_grid_box.addWidget(self.add_roi_grid_button)
        self.add_roi_grid_button.clicked.connect(self.open_grid_window)
        # Create a QLabel for the spinbox
        ncpu_label = QLabel("nThreads")
        roi_grid_box.addWidget(ncpu_label)  # Add label to the layout

        # Create a QSpinBox
        self.ncpu_spinbox = QSpinBox()
        self.ncpu_spinbox.setMinimum(1)  # Assuming you want to set a minimum value
        self.ncpu_spinbox.setMaximum(48)  # And a maximum value
        self.ncpu_spinbox.setRange(1, 48)
        self.ncpu_spinbox.setValue(multiprocessing.cpu_count()-2)
        roi_grid_box.addWidget(self.ncpu_spinbox)
        # self.roi_grid_size_selector = QSpinBox()
        # self.roi_grid_size_selector.setRange(1, 1000)
        # self.roi_grid_size_selector.setSingleStep(1)
        # self.roi_grid_size_selector.setValue(2)
        #self.roi_grid_size_selector.valueChanged.connect(self.roi_grid_size_valuechange)
        #roi_grid_box.addWidget(QLabel("Grid Size (pix)"))
        #roi_grid_box.addWidget(self.roi_grid_size_selector)
        roi_definition_group_box.addLayout(roi_grid_box)
        ngrid_rois_label = QLabel("nGridRois")
        roi_grid_box.addWidget(ngrid_rois_label)  # Add label to the layout
        self.grid_roi_number_lcd = QLCDNumber()
        self.grid_roi_number_lcd.setObjectName("gridRoiNumber")
        self.grid_roi_number_lcd.setStyleSheet("QLCDNumber { background-color: grey; color: green; }")
        self.grid_roi_number_lcd.display(self.number_grid_rois)
        roi_grid_box.addWidget(self.grid_roi_number_lcd)
        roi_definition_group_box.addLayout(roi_grid_box)


        fit_layout.addWidget(roi_definition_group)
        # Group 2: ROI Selection Group
        roi_selection_group = QGroupBox("ROI Selection")
        roi_selection_group_box = QVBoxLayout()
        roi_selection_group.setLayout(roi_selection_group_box)

        roi_selection_button_box = QHBoxLayout()

        # ROI GROUP Selection Button
        self.roigroup_selection_button = QToolButton(self)
        self.roigroup_selection_button.setText('0')
        self.roigroup_selection_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.roigroup_selection_button.setMenu(QMenu(self.roigroup_selection_button))
        self.roigroup_selection_widget = QListWidget()
        self.roigroup_selection_widget.addItems([str(i) for i in range(len(self.rois))])
        #self.roigroup_selection_widget.addItems(['0', '1', '2'])  ##todo: update ROI group list on the fly
        self.roigroup_selection_widget.itemClicked.connect(self.roigroup_selection)
        action = QWidgetAction(self.roigroup_selection_button)
        action.setDefaultWidget(self.roigroup_selection_widget)
        self.roigroup_selection_button.menu().addAction(action)
        roi_selection_button_box.addWidget(QLabel("Select ROI Group"))
        roi_selection_button_box.addWidget(self.roigroup_selection_button)

        # ROI Selection Button
        self.roi_selection_button = QToolButton(self)
        self.roi_selection_button.setText(str(self.current_roi_idx))
        self.roi_selection_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.roi_selection_button.setMenu(QMenu(self.roi_selection_button))
        self.roi_selection_widget = QListWidget()
        self.roi_selection_widget.addItems([str(i) for i in range(self.nroi_current_group)])
        self.roi_selection_widget.itemClicked.connect(self.roi_selection_action)
        action = QWidgetAction(self.roi_selection_button)
        action.setDefaultWidget(self.roi_selection_widget)
        self.roi_selection_button.menu().addAction(action)
        roi_selection_button_box.addWidget(QLabel("Select ROI"))
        roi_selection_button_box.addWidget(self.roi_selection_button)
        # Slider for ROI selection
        # self.roi_select_slider = QSlider(Qt.Horizontal)
        # self.roi_select_slider.setMinimum(0)
        # self.roi_select_slider.setMaximum(1)
        # self.roi_select_slider.setSingleStep(1)
        # self.roi_select_slider.setTickPosition(QSlider.TicksBelow)
        # self.roi_select_slider.setTickInterval(1)
        # self.roi_select_slider.valueChanged.connect(self.roi_slider_valuechange)
        # roi_select_box.addWidget(QLabel("ROI ID #"))
        # roi_select_box.addWidget(self.roi_select_slider)
        roi_selection_group_box.addLayout(roi_selection_button_box)

        # Text Box for ROI information
        self.roi_info = QTextEdit()
        self.roi_info.setReadOnly(True)
        self.roi_info.setMinimumHeight(50)
        self.roi_info.setMaximumHeight(100)
        # self.roi_info.setMinimumWidth(200)
        # self.roi_info.setMaximumWidth(200)
        # roi_group_box.addWidget(QLabel("ROI Information"))
        roi_selection_group_box.addWidget(self.roi_info)

        # Layout for computing and applying total power calibration factor
        tpcal_correction_box = QHBoxLayout()
        # Button for combining flux from a ROI group
        combine_flux_button = QPushButton("Combine Flux of Current ROI Group")
        combine_flux_button.clicked.connect(self.combine_roi_group_flux)
        tpcal_correction_box.addWidget(combine_flux_button)

        # Button for calculating total power calibration factor
        calc_tpcal_factor_button = QPushButton("Compute TP Cal factor")
        calc_tpcal_factor_button.clicked.connect(self.calc_tpcal_factor)
        tpcal_correction_box.addWidget(calc_tpcal_factor_button)

        # Button for applying total power calibration factor
        self.apply_tpcal_factor_button = QRadioButton("Apply?")
        self.apply_tpcal_factor_button.clicked.connect(self.apply_tpcal_factor)
        tpcal_correction_box.addWidget(self.apply_tpcal_factor_button)

        roi_selection_group_box.addLayout(tpcal_correction_box)

        ## Add buttons for doing spectral fit
        fit_button_box = QHBoxLayout()
        # Background selection button
        # self.bkg_selection_button = QPushButton("Select Background")
        # self.bkg_selection_button.setStyleSheet("background-color : lightgrey")
        # self.bkg_selection_button.setCheckable(True)
        # self.bkg_selection_button.clicked.connect(self.bkg_rgn_select)
        # fit_button_box.addWidget(self.bkg_selection_button)

        # Do Fit Button
        do_spec_fit_button = QPushButton("Fit Selected ROI")
        do_spec_fit_button.clicked.connect(self.do_spec_fit)
        fit_button_box.addWidget(do_spec_fit_button)

        # Add fit button box to roi group
        roi_selection_group_box.addLayout(fit_button_box)

        fit_layout.addWidget(roi_selection_group)

        ####### Group 2: Fit Settings Group
        fit_setting_group = QGroupBox("Fit Settings")
        fit_setting_box = QVBoxLayout()
        fit_setting_group.setLayout(fit_setting_box)
        spec_adjust_box = QHBoxLayout()

        # Group 2: Lower Frequency Bounds for fit
        self.freq_lowbound_selector = QDoubleSpinBox()
        self.freq_lowbound_selector.setDecimals(1)
        self.freq_lowbound_selector.setRange(self.data_freq_bound[0], self.data_freq_bound[1])
        self.freq_lowbound_selector.setSingleStep(0.1)
        self.freq_lowbound_selector.setValue(self.data_freq_bound[0])
        self.freq_lowbound_selector.valueChanged.connect(self.freq_lowbound_valuechange)
        spec_adjust_box.addWidget(QLabel("Min Freq (GHz)"))
        spec_adjust_box.addWidget(self.freq_lowbound_selector)

        # Group 2: Upper Frequency Bounds for fit
        self.freq_hibound_selector = QDoubleSpinBox()
        self.freq_hibound_selector.setDecimals(1)
        self.freq_hibound_selector.setRange(self.data_freq_bound[0], self.data_freq_bound[1])
        self.freq_hibound_selector.setSingleStep(0.1)
        self.freq_hibound_selector.setValue(self.data_freq_bound[1])
        self.freq_hibound_selector.valueChanged.connect(self.freq_hibound_valuechange)
        spec_adjust_box.addWidget(QLabel("Max Freq (GHz)"))
        spec_adjust_box.addWidget(self.freq_hibound_selector)

        # Group 2: Fractional intensity error for fit
        self.spec_frac_err_selector = QDoubleSpinBox()
        self.spec_frac_err_selector.setDecimals(2)
        self.spec_frac_err_selector.setRange(0.0, 1.0)
        self.spec_frac_err_selector.setSingleStep(0.02)
        self.spec_frac_err_selector.setValue(self.spec_frac_err)
        self.spec_frac_err_selector.valueChanged.connect(self.spec_frac_err_valuechange)
        spec_adjust_box.addWidget(QLabel("Frac. Intensity Error"))
        spec_adjust_box.addWidget(self.spec_frac_err_selector)

        fit_setting_box.addLayout(spec_adjust_box)

        # Group 2: Fit Method
        self.fit_method_box = QHBoxLayout()
        self.fit_method_selector_widget = QComboBox()
        self.fit_method_selector_widget.addItems(["nelder", "basinhopping", "differential_evolution", "mcmc"])
        self.fit_method_selector_widget.currentIndexChanged.connect(self.fit_method_selector)
        self.fit_method_box.addWidget(QLabel("Fit Method"))
        self.fit_method_box.addWidget(self.fit_method_selector_widget)

        # Group 2: Fit Method Dependent Keywords
        self.fit_kws_box = QHBoxLayout()
        self.update_fit_kws_widgets()

        # Group 2: Electron Function
        ele_function_box = QHBoxLayout()
        self.ele_function_selector_widget = QComboBox()
        self.ele_function_selector_widget.addItems(
            ["powerlaw", "double_Powerlaw", "thermal f-f + gyrores", "thermal f-f"])
        self.ele_function_selector_widget.currentIndexChanged.connect(self.ele_function_selector)
        ele_function_box.addWidget(QLabel("Electron Dist. Function"))
        ele_function_box.addWidget(self.ele_function_selector_widget)

        fit_setting_box.addLayout(self.fit_method_box)
        fit_setting_box.addLayout(self.fit_kws_box)
        fit_setting_box.addLayout(ele_function_box)
        fit_layout.addWidget(fit_setting_group)

        ##### Group 3: Fit Parameters Group
        fit_param_group = QGroupBox("Fit Parameters")
        self.fit_param_box = QGridLayout()
        fit_param_group.setLayout(self.fit_param_box)
        self.update_fit_param_widgets()

        fit_layout.addWidget(fit_param_group)
        main_layout.addLayout(fit_layout)

        self._main.setLayout(main_layout)
        self.show()

    # Fit Tab User Interface
    #
    # def initUItab_fit(self):
    #    # Create main layout (a Vertical Layout)
    #    mainlayout = QVBoxLayout()
    #    mainlayout.addWidget(QLabel("This is the tab for analyzing fit results"))
    #    self.tabs.widget(1).setLayout(mainlayout)

    # Analyzer Tab User Interface
    #
    def initUItab_analyzer(self):
        # Create main layout (a Vertical Layout)
        mainlayout = QVBoxLayout()
        mainlayout.addWidget(QLabel("This is the tab for analyzing fit results"))
        self.tabs.widget(1).setLayout(mainlayout)

    def eoimg_file_select(self):
        """ Handle Browse button for EOVSA FITS file"""
        # self.fname = QFileDialog.getExistingDirectory(self, 'Select FITS File', './', QFileDialog.ShowDirsOnly)
        ## quick input for debug -------------
        tmp_eoimg_fname_seq, _file_filter = QFileDialog.getOpenFileNames(self, 'Select EOVSA Spectral Image FITS File(s)',
                                                                     './', 'FITS Images (*.fits *.fit *.ft *.fts)')
        self.data_in_seq = len(tmp_eoimg_fname_seq) > 1
        self.eoimg_time_seq = tmp_eoimg_fname_seq
        self.eoimg_fname = self.eoimg_time_seq[self.cur_frame_idx]
        self.eoimg_seq_slider_label_widget.setVisible(self.data_in_seq)
        self.cur_frame_idx = 0
        self.eoimg_files_seq_select_return()
        print('{} files are selected and loaded as a img sequence.'.format(len(tmp_eoimg_fname_seq)))
        
        # self.eoimg_fname = 'EOVSA_20210507T190135.000000.outim.image.allbd.fits'
        ## quick input for debug -------------
        # self.eoimg_fitsentry.setText(self.eoimg_fname)


    def eoimg_files_seq_select_return(self):
        ''' Called when load eoimg in a time sequence.
            Trys to read FITS header and return header info (no data read at this time)
        '''

        # print('self.qlookimg_axs is None:',self.qlookimg_axs is None)
        if self.qlookimg_axs is not None:
            for ax in self.qlookimg_axs:
                ax.cla()
        #     self.showqlookimg(showplt=True)
        # self.eoimg_fname = self.eoimg_fitsentry.text()
        self.eoimg_fitsdata = None
        self.has_eovsamap = False

        self.pg_img_canvas.clear()
        if self.has_bkg:
            self.pg_img_canvas.removeItem(self.bkg_roi)
            self.has_bkg = False
        if self.has_rois:
            for roi_group in self.rois:
                for roi in roi_group:
                    self.pg_img_canvas.removeItem(roi)
            self.rois = [[]]
            self.has_rois = False

        try:
        #if True:
            meta, data = ndfits.read(self.eoimg_fname)
            if meta['naxis'] < 3:
                print('Input fits file must have at least 3 dimensions. Abort..')
            elif meta['naxis'] == 3:
                print('Input fits file does not have stokes axis. Assume Stokes I.')
                data = np.expand_dims(data, axis=0)
                self.pol_axis = 0
                self.pol_names = ['I']
                self.has_stokes = False
            elif meta['naxis'] == 4:
                print('Input fits file has stokes axis, located at index {0:d} of the data cube'.
                      format(meta['pol_axis']))
                self.has_stokes = True
                self.pol_axis = meta['pol_axis']
                self.pol_names = meta['pol_names']
            self.meta = meta
            self.data = data
            self.cfreqs = meta['ref_cfreqs'] / 1e9  # convert to GHz
            self.freqdelts = meta['ref_freqdelts'] / 1e9  # convert to GHz
            self.freq_dist = lambda fq: (fq - self.cfreqs[0]) / (self.cfreqs[-1] - self.cfreqs[0])
            self.x0, self.x1 = (np.array([1, meta['nx']]) - meta['header']['CRPIX1']) * meta['header']['CDELT1'] + \
                               meta['header']['CRVAL1']
            self.y0, self.y1 = (np.array([1, meta['ny']]) - meta['header']['CRPIX2']) * meta['header']['CDELT2'] + \
                               meta['header']['CRVAL2']
            self.dx, self.dy = self.meta['header']['CDELT1'], self.meta['header']['CDELT2']
            self.xcen, self.ycen = [(self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0]
            self.xsiz, self.ysiz = [self.meta['nx'] * self.dx,
                                    self.meta['ny'] * self.dy]
            self.mapx, self.mapy = np.linspace(self.x0, self.x1, meta['nx']), np.linspace(self.y0, self.y1, meta['ny'])
            self.tp_cal_factor = np.ones_like(self.cfreqs)

            self.has_eovsamap = True
            with fits.open(self.eoimg_fname, mode='readonly') as wcs_hdul:
                self.eo_wcs = wcs.WCS(wcs_hdul[0].header)
            # self.infoEdit.setPlainText(repr(rheader))
            #extend a time axis to the data:
            self.eoimg_date_seq = []
            self.eoimg_exptime_seq = []
            new_4d_data = np.repeat(np.expand_dims(self.data, axis=1), repeats=len(self.eoimg_time_seq), axis=1)
            for cf_idx, c_eo_file in enumerate(self.eoimg_time_seq):
                cmeta, cdata = ndfits.read(c_eo_file)
                self.eoimg_date_seq.append(cmeta['refmap'].date)
                self.eoimg_exptime_seq.append(TimeDelta(cmeta['refmap'].meta['exptime'], format='sec'))
                new_4d_data[self.pol_select_idx, cf_idx,:,:,:] = cdata[self.pol_select_idx, :,:,:]
            self.data = new_4d_data
            self.eoimg_seq_slider.setMaximum(len(self.eoimg_time_seq)-1)
            self.eoimg_seq_slider.setValue(self.cur_frame_idx)
            self.text_box_time.setText(self.eoimg_date_seq[self.cur_frame_idx].iso[11:])
        except:
        #else:
            self.statusBar.showMessage('Filename is not a valid FITS file', 2000)
            self.eoimg_fname = '<Select or enter a valid fits filename>'
            self.eoimg_time_seq = []
            self.cur_frame_idx = 0
            # self.eoimg_fitsentry.setText(self.eoimg_fname)
            # self.infoEdit.setPlainText('')

        if not self.qlookimgbutton.isChecked():
            self.qlookimgbutton.setChecked(True)
        else:
            self.showqlookimg()
        # Clean up all existing plots

        #self.plot_qlookmap()
        self.init_pgspecplot()
        self.plot_pg_eovsamap()
        if self.has_dspec:
            self.plot_dspec()

    def time_slider_slided(self):
        self.text_box_time.setText(self.eoimg_date_seq[self.eoimg_seq_slider.value()].iso[11:])

    def time_slider_released(self):
        self.cur_frame_idx = self.eoimg_seq_slider.value()
        self.plot_eoimgs_trange_on_dspec()
        # if not self.qlookimgbutton.isChecked():
        #     self.qlookimgbutton.setChecked(True)
        # else:
        #     self.showqlookimg()
        self.init_pgspecplot()
        self.plot_pg_eovsamap()
        self.calc_roi_spec(None)
        self.eoimg_fname = self.eoimg_time_seq[self.cur_frame_idx]

    def eodspec_file_select(self):
        """ Handle Browse button for EOVSA FITS file"""
        # self.fname = QFileDialog.getExistingDirectory(self, 'Select FITS File', './', QFileDialog.ShowDirsOnly)
        self.eodspec_fname, _file_filter = QFileDialog.getOpenFileName(self, 'Select EOVSA Dynamic Spectrum FITS File',
                                                                       './', 'FITS Images (*.fits *.fit *.ft *.fts)')
        # self.eodspec_fitsentry.setText(self.eodspec_fname)
        self.eodspec_file_select_return()

    def eodspec_file_select_return(self):
        try:
            hdulist = fits.open(self.eodspec_fname)
            dspec = hdulist[0].data
            header = hdulist[0].header
            observatory = header['telescop']
            pol = header['polariza']
            fghz = np.array(astropy.table.Table(hdulist[1].data)['sfreq'])
            tim_ = astropy.table.Table(hdulist[2].data)
            tmjd = np.array(tim_['mjd']) + np.array(tim_['time']) / 24. / 3600 / 1000
            tim = Time(tmjd, format='mjd')
            self.dspec = {'dspec': dspec, 'time_axis': tim, 'freq_axis': fghz, 'observatory': observatory, 'pol': pol}
            self.has_dspec = True
            self.eodspec_fname = '<Select or enter a valid fits filename>'
            # self.eodspec_fitsentry.setText(self.eodspec_fname)
        except:
            self.statusBar.showMessage('{} is not a valid dynamic spectrum FITS file'.format(self.eodspec_fname))

        if not self.qlookdspecbutton.isChecked():
            self.qlookdspecbutton.setChecked(True)
        else:
            self.showqlookdspec()

    def is_calibrated_tp_state(self):
        if self.is_calibrated_tp_button.isChecked() == True:
            self.statusBar.showMessage('Loaded spectrogram is calibrated total power dynamic spectrum.')
            self.is_calibrated_tp = True
        else:
            self.statusBar.showMessage('Loaded spectrogram is *not* calibrated total power dynamic spectrum.')
            self.is_calibrated_tp = False

    ## Surajit
    def savedata_state(self):
        if self.savedata_button.isChecked() == True:
            self.statusBar.showMessage('Saving the fit data')
            self.savedata = True
        else:
            self.statusBar.showMessage('Not saving the fit data')
            self.savedata = False

    def aiafile_select(self):
        """ Handle Browse button for AIA FITS file """
        self.aiafname, _file_filter = QFileDialog.getOpenFileName(self, 'Select AIA FITS File to open', './',
                                                                  "FITS Images (*.fits *.fit *.ft)")
        # self.aiaimg_fitsentry.setText(self.aiafname)
        self.aiafile_select_return()

    def aiafile_select_return(self):
        ''' Called when the FITS filename LineEdit widget gets a carriage-return.
            Trys to read FITS header and return header info (no data read at this time)
        '''
        # self.aiafname = self.aiaimg_fitsentry.text()
        self.eoimg_fitsdata = None
        try:
            hdu = fits.open(self.aiafname)
            # self.infoEdit.setPlainText(repr(hdu[1].header))
        except:
            self.statusBar.showMessage('Filename is not a valid FITS file', 2000)
            self.aiafname = '<Select or enter a valid fits filename>'
            # self.aiafitsentry.setText(self.fname)
            # self.infoEdit.setPlainText('')
        if not self.qlookimgbutton.isChecked():
            self.qlookimgbutton.setChecked(True)
        else:
            self.showqlookimg()
        # self.plot_qlookmap()

    def init_pgspecplot(self):
        """ Initial Spectral Plot if no data has been loaded """
        xticksv = list(range(0, 4))
        self.xticks = []
        xticksv_minor = []
        for v in xticksv:
            vp = 10 ** v
            xticksv_minor += list(np.arange(2 * vp, 10 * vp, 1 * vp))
            self.xticks.append((v, '{:.0f}'.format(vp)))
        self.xticks_minor = []
        for v in xticksv_minor:
            self.xticks_minor.append((np.log10(v), ''))

        if self.spec_in_tb:
            yticksv = list(range(1, 15))
        else:
            yticksv = list(range(-3, 3))
        yticksv_minor = []
        for v in yticksv:
            vp = 10 ** v
            yticksv_minor += list(np.arange(2 * vp, 10 * vp, 1 * vp))
        self.yticks_minor = []
        for v in yticksv_minor:
            self.yticks_minor.append((np.log10(v), ''))
        self.yticks = []
        if self.spec_in_tb:
            for v in yticksv:
                if v >= 6:
                    self.yticks.append([v, r'{:.0f}'.format(10 ** v / 1e6)])
                if v == 1:
                    self.yticks.append([v, r'{:.5f}'.format(10 ** v / 1e6)])
                elif v == 2:
                    self.yticks.append([v, r'{:.4f}'.format(10 ** v / 1e6)])
                elif v == 3:
                    self.yticks.append([v, r'{:.3f}'.format(10 ** v / 1e6)])
                elif v == 4:
                    self.yticks.append([v, r'{:.2f}'.format(10 ** v / 1e6)])
                elif v == 5:
                    self.yticks.append([v, r'{:.1f}'.format(10 ** v / 1e6)])
                else:
                    self.yticks.append([v, r'{:.0f}'.format(10 ** v / 1e6)])
        else:
            for v in yticksv:
                if v >= 0:
                    self.yticks.append([v, r'{:.0f}'.format(10 ** v)])
                if v == -3:
                    self.yticks.append([v, r'{:.3f}'.format(10 ** v)])
                if v == -2:
                    self.yticks.append([v, r'{:.2f}'.format(10 ** v)])
                if v == -1:
                    self.yticks.append([v, r'{:.1f}'.format(10 ** v)])

        self.update_pgspec()

    def plot_qlookmap(self):
        from utils.img_utils import submap_of_file1, resize_array
        #todo qlook is totally messed up at this moment, will be fixed soon.
        """Quicklook plot in the upper box using matplotlib.pyplot and sunpy.map"""
        # Plot a quicklook map
        # self.qlook_ax.clear()
        ax0 = self.qlookimg_axs[0]

        if self.has_eovsamap:
            ax0 = self.update_axes_projection(ax0, projection=self.meta['refmap'])
            nspw = self.meta['nfreq']
            self.eoimg_date = eoimg_date = Time(self.meta['refmap'].date.mjd +
                                                self.meta['refmap'].exposure_time.value / 2. / 24 / 3600, format='mjd')
            eotimestr = eoimg_date.isot[:-4]
            rsun_obs = sunpy.coordinates.sun.angular_radius(eoimg_date).value
            solar_limb = patches.Circle((0, 0), radius=rsun_obs, fill=False, color='k', lw=1, linestyle=':')
            ax0.add_patch(solar_limb)
            rect = patches.Rectangle((self.x0, self.y0), self.x1 - self.x0, self.y1 - self.y0,
                                     color='k', alpha=0.7, lw=1, fill=False)
            ax0.add_patch(rect)
            icmap = plt.get_cmap('RdYlBu')

            self.qlookimg_canvas.figure.suptitle('EOVSA at {}'.format(eotimestr))
        else:
            self.statusBar.showMessage('EOVSA FITS file does not exist', 2000)
            self.eoimg_fname = '<Select or enter a valid fits filename>'
            # self.eoimg_fitsentry.setText(self.eoimg_fname)
            # self.infoEdit.setPlainText('')
            self.has_eovsamap = False

        if os.path.exists(self.aiafname):
            try:
                #aiamap = sunpy.map.Map(sunpy.data.sample.SWAP_LEVEL1_IMAGE)
                aiamap = smap.Map(self.aiafname)
                self.has_aiamap = True
            except:
                self.statusBar.showMessage('Something is wrong with the provided AIA FITS file', 2000)
        else:
            self.statusBar.showMessage('AIA FITS file does not exist', 2000)
            self.aiafname = '<Select or enter a valid fits filename>'
            # self.aiaimg_fitsentry.setText(self.aiafname)
            # self.infoEdit.setPlainText('')
            self.has_aiamap = False
        cts = []
        if self.has_aiamap:
            aiacmap = plt.get_cmap('gray_r')
            #ax0 = self.update_axes_projection(ax0, projection=aiamap)
            #blbl = self.meta['refmap'].bottom_left_coord
            #submap_aia = aiamap.submap(bottom_left=self.meta['refmap'].bottom_left_coord, top_right=self.meta['refmap'].top_right_coord)
            if self.has_eovsamap:
                submap_aia = submap_of_file1(self.aiafname, self.meta['refmap'])
            else:
                submap_aia = aiamap
            ax0 = self.update_axes_projection(ax0, projection=submap_aia)
            bounds = ax0.axis()
            submap_aia.plot(axes=ax0, cmap=aiacmap, clip_interval=(1, 99.99) * u.percent)
            ax0.set_title('')
            aia_tit_str = 'AIA {0:.0f} at {1:s}'.format(aiamap.wavelength.value, aiamap.date.isot[:19])
            ax0.text(0.02, 0.98, aia_tit_str, ha='left', va='top', transform=ax0.transAxes, fontsize=10)
        else:
            ax0 = self.update_axes_projection(ax0, projection=self.meta['refmap'])
            bounds = ax0.axis()
        if self.has_eovsamap:
            for s, sp in enumerate(self.cfreqs):
                #data = self.data[self.pol_select_idx, s, ...]
                data = self.data[self.pol_select_idx, self.cur_frame_idx, s, ...]
                cur_sunmap = smap.Map(data, self.meta['refmap'].meta)
                clvls = self.clevels * np.nanmax(data) * u.K
                rcmap = [icmap(self.freq_dist(self.cfreqs[s]))] * len(clvls)
                if not self.has_aiamap:
                    cur_sunmap.draw_contours(clvls, axes=ax0, colors=rcmap, alpha=self.calpha)
                    print(sp, clvls, np.max(cur_sunmap.data))
                else:
                    stdata1 = resize_array(data, submap_aia.data.shape)
                    ax0.contour(stdata1, [self.clevels*np.nanmax(stdata1)], colors=rcmap, alpha=self.calpha)
                #else:
                #    submap_eovsa = cur_sunmap.submap(bottom_left=aiamap.bottom_left_coord,
                #                           top_right=aiamap.top_right_coord)
                #    submap_eovsa.draw_contours(clvls, axes=ax0, colors=rcmap, alpha=self.calpha)
                #cts.append(cur_sunmap.draw_contours(clvls, axes=ax0, colors=rcmap, alpha=self.calpha))
                # if not self.opencontour:
                #     continue
                #     # todo
            #ax0.axis(bounds)

        # ax0.set_xlim([-1200, 1200])
        # ax0.set_ylim([-1200, 1200])
        ax0.set_xlabel('Solar X [arcsec]')
        ax0.set_ylabel('Solar Y [arcsec]')
        # ax0.set_title('')
        ax0.set_aspect('equal')
        # self.qlookimg_canvas.figure.subplots_adjust(left=0.10, right=0.95,
        #                                        bottom=0.10, top=0.95,
        #                                        hspace=0, wspace=0.35)
        self.qlookimg_canvas.draw()

    def plot_dspec(self, cmap='viridis', vmin=None, vmax=None):
        from matplotlib import dates as mdates
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if not vmin:
            vmin = np.nanmin(vmin)
        if not vmax:
            vmax = np.nanmax(vmax)
        ax = self.qlookdspec_ax
        tim = self.dspec['time_axis']
        tim_plt = self.dspec['time_axis'].plot_date
        fghz = self.dspec['freq_axis']
        spec = self.dspec['dspec']
        if not vmin:
            vmin = np.nanmin(spec)
        if not vmax:
            vmax = np.percentile(spec.flatten(), 90)
        observatory = self.dspec['observatory']
        pol = self.dspec['pol']
        ##todo: now I am only using the first polarization and first baseline
        if spec.ndim < 2:
            print('Dynamic spectrum needs at least 2 dimensions. We have {0:d} here.'.format(spec.ndim))
            return
        elif spec.ndim == 2:
            nfreq, ntim = len(fghz), len(tim_plt)
            npol = 1
            nbl = 1
            spec_plt = spec
        else:
            (npol, nbl, nfreq, ntim) = spec.shape
            print('Dynamic spectrum has more than 2 dimensions {0:d}. '
                  'I am only using the first polarization and first baseline'.format(spec.ndim))
            spec_plt = spec[0, 0]

        im_spec = ax.pcolormesh(tim_plt, fghz, spec_plt, cmap=cmap,
                                vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title("{0:s} Stokes {1:s} Spectrogram on {2:s}".format(observatory, pol, tim[0].iso[:10]), fontsize=10)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.set_xlim(tim_plt[0], tim_plt[-1])
        ax.set_ylim(fghz[0], fghz[-1])
        ax.set_xlabel('Time [UT]', fontsize=9)
        ax.set_ylabel('Frequency [GHz]')
        for xlabel in ax.get_xmajorticklabels():
            xlabel.set_rotation(30)
            xlabel.set_horizontalalignment("right")

        # add vertical bar to indicate the time of the EOVSA image
        if hasattr(self, 'eoimg_date'):
            ax.plot([self.eoimg_date.plot_date] * 2, [1, 20], color='w', lw=1)

        divider = make_axes_locatable(ax)
        cax_spec = divider.append_axes('right', size='3.0%', pad=0.05)
        cax_spec.tick_params(direction='out')
        clb_spec = plt.colorbar(im_spec, ax=ax, cax=cax_spec)
        clb_spec.set_label('Flux [sfu]')
        self.qlookdspec_canvas.figure.subplots_adjust(left=0.1, right=0.85,
                                                      bottom=0.20, top=0.92,
                                                      hspace=0, wspace=0)
        self.qlookdspec_canvas.draw()
        self.plot_eoimgs_trange_on_dspec()



    def plot_eoimgs_trange_on_dspec(self):
        from matplotlib import dates as mdates
        def onclick_eoimg_trange(event):
            for irect, rect in enumerate(rectangles):
                if rect.contains(event)[0]:  # Check if click is inside the rectangle
                    self.cur_frame_idx = irect
                    self.eoimg_seq_slider.setValue(self.cur_frame_idx)
                    self.init_pgspecplot()
                    self.plot_pg_eovsamap()
                    self.calc_roi_spec(None)
                    self.eoimg_fname = self.eoimg_time_seq[self.cur_frame_idx]
                    #plot_spans()  # Call to redraw the spans
                    break

        def get_or_create_twinx(ax):
            fig = ax.figure
            twin_ax = None
            for other_ax in fig.axes:
                if other_ax is not ax and other_ax.get_position() == ax.get_position():
                    if ax.get_shared_x_axes().joined(ax, other_ax) or ax.get_shared_y_axes().joined(ax, other_ax):
                        twin_ax = other_ax
                        break
            if twin_ax is None:
                twin_ax = ax.twinx()
            return twin_ax

        def plot_spans():
            ax.cla()  # Clear previous spans
            for cframei, (start, end) in enumerate(eoimg_tr_list):
                #c_color = 'crimson' if cframei == self.cur_frame_idx else 'navy'
                c_color = 'crimson'
                rect = ax.axvspan(start, end, color=c_color, alpha=0.5)
                rectangles.append(rect)

        if self.has_eovsamap and self.has_dspec:
            ax = get_or_create_twinx(self.qlookdspec_ax)
            #ax = self.qlookdspec_ax
            eoimg_tr_list = [[cdate.plot_date, (cdate + cdelta).plot_date] for cdate, cdelta in
                             zip(self.eoimg_date_seq, self.eoimg_exptime_seq)]
            rectangles = []
            plot_spans()  # Initial call to plot the spans
            ax.figure.canvas.mpl_connect('button_press_event', onclick_eoimg_trange)
            if len(rectangles)>1:
                cur_total_range = self.eoimg_date_seq[-1] - self.eoimg_date_seq[0]
                ax.set_xlim([max(ax.get_xlim()[0], (self.eoimg_date_seq[0] - 3*cur_total_range).plot_date),
                            min(ax.get_xlim()[1], (self.eoimg_date_seq[-1] + 3*cur_total_range).plot_date)])
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
        else:
            print('EOVSA image(s) or Dspec is(are) not loaded yet')


    def update_axes_projection(self, cax, projection):
        pos = cax.get_position()
        cfig = self.qlookimg_canvas.figure
        cfig.delaxes(cax)
        new_ax = cfig.add_axes(pos, projection=projection)
        return new_ax

    def plot_pg_eovsamap(self):
        """This is to plot the eovsamap with the pyqtgraph's ImageView Widget"""
        if self.has_eovsamap:
            # plot the images
            # need to invert the y axis to put the origin to the lower left (tried invertY but labels are screwed up)
            # self.pg_img_plot.setLimits(xMin=self.x0, xMax=self.x1, yMin=self.y0, yMax=self.y1)
            # self.pgdata = self.data[self.pol_select_idx, :, ::-1, :].reshape(
            #     (self.meta['nfreq'], self.meta['ny'], self.meta['nx']))
            # self.pgdata = self.data[self.pol_select_idx, :, :, :].reshape(
            #     (self.meta['nfreq'], self.meta['ny'], self.meta['nx']))
            self.pgdata = self.data[self.pol_select_idx,self.cur_frame_idx, :, :, :].reshape(
                (self.meta['nfreq'], self.meta['ny'], self.meta['nx']))
            #self.pgdata = self.data[self.pol_select_idx,self.cur_frame_idx, :, :, :]
            pos = np.where(self.pgdata > 1e9)
            self.pgdata[pos] = 1e9
            del pos
            pos = np.where(self.pgdata < -1000)
            self.pgdata[pos] = -1000
            del pos
            # self.pg_img_canvas.setImage(self.pgdata, xvals=self.cfreqs)
            # self.pg_img_canvas.setImage(self.pgdata, xvals=self.cfreqs, pos=[self.x0, self.y0],
            #                             scale=[self.meta['header']['CDELT1'], self.meta['header']['CDELT2']])
            self.pg_img_canvas.setImage(self.pgdata, xvals=self.cfreqs, pos=[self.x0, self.y0],
                                        scale=[self.meta['header']['CDELT1'], self.meta['header']['CDELT2']])
            # self.pg_img_canvas.setImage(self.data[self.pol_select_idx], xvals=self.cfreqs)
            self.pg_img_canvas.getView().invertY(False)

            self.pg_img_canvas.setColorMap(self.pgcmap)
            self.pg_freq_current = self.pg_img_canvas.timeLine.getXPos()
            self.pg_freq_idx = np.argmin(np.abs(self.cfreqs - self.pg_freq_current))
            self.pg_img_canvas.getImageItem().hoverEvent = self.pg_map_hover_event

            # define the initial background ROI region
            if not self.has_bkg:
                self.bkg_roi = pg.RectROI([self.x0 + self.dx / 2.0, self.y1 + self.dx / 2.0 - self.ysiz / 5],
                                          [self.xsiz / 5, self.ysiz / 5], pen='w')
                self.bkg_roi.addScaleHandle([1.0, 0.0], [0.0, 1.0])
                self.bkg_roi.addRotateHandle([1.0, 0.5], [0.5, 0.5])
                self.pg_img_canvas.addItem(self.bkg_roi)
                self.has_bkg = True
                self.bkg_roi_label = pg.TextItem("Background", anchor=(0, 0), color='w')
                self.bkg_roi_label.setParentItem(self.bkg_roi)
                self.bkg_rgn_update()
                self.bkg_roi.sigRegionChanged.connect(self.bkg_rgn_update)

    def pg_map_hover_event(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        vbox = self.pg_img_canvas.getView()
        if event.isExit():
            # self.pg_img_hover_label.setText("")
            self.pg_img_mouse_pos_widget.setText("")
            return

        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.meta['ny'] - 1))
        j = int(np.clip(j, 0, self.meta['nx'] - 1))
        self.pg_freq_current = self.pg_img_canvas.timeLine.getXPos()
        self.pg_freq_idx = np.argmin(np.abs(self.cfreqs - self.pg_freq_current))
        val = self.pgdata[self.pg_freq_idx, i, j]
        # ppos = self.pg_img_canvas.getImageItem().mapToParent(pos)
        solx, soly = self.mapx[j], self.mapy[::-1][i]
        text_to_display = '[Cursor] x: {0:6.1f}", y: {1:6.1f}", freq: {3:4.1f} GHz, ' \
                          'T<sub>B</sub>={4:6.2f} MK'.format(solx, soly, self.pg_freq_idx,
                                                             self.pg_freq_current, val / 1e6)
        self.pg_img_mouse_pos_widget.setText(text_to_display)
        # print(text_to_display)
        # self.pg_img_hover_label.setText(text_to_display, color='w')

    def update_fitmask(self):
        # update fit mask and the values for currently selected roi group
        if self.has_rois:
            for roi in self.rois[self.roi_group_idx]:
                # define error in spectrum
                if self.spec_in_tb:
                    spec = roi.tb_max
                    spec_bound = self.tb_spec_bound
                    spec_rms = self.bkg_roi.tb_rms
                else:
                    spec = roi.total_flux
                    spec_bound = self.flx_spec_bound
                    spec_rms = gstools.sfu2tb(roi.freqghz * 1e9 * u.Hz, self.bkg_roi.tb_rms * u.K,
                                              area=roi.total_area * u.arcsec ** 2, reverse=True).value
                # add fractional err in quadrature
                spec_err = np.sqrt(spec_rms ** 2. + (self.spec_frac_err * spec) ** 2.)

                spec_tofit0 = ma.masked_less_equal(spec, spec_bound[0])
                freqghz_tofit0 = ma.masked_outside(self.cfreqs, self.fit_freq_bound[0], self.fit_freq_bound[1])
                roi.mask_tofit = mask_tofit = np.logical_or(np.logical_or(freqghz_tofit0.mask, spec_tofit0.mask),
                                                            roi.freq_mask)
                roi.freqghz_tofit = ma.masked_array(self.cfreqs, mask_tofit)
                roi.spec_tofit = ma.masked_array(spec, mask_tofit)
                roi.tb_max_tofit = ma.masked_array(roi.tb_max, mask_tofit)
                roi.total_flux_tofit = ma.masked_array(roi.total_flux, mask_tofit)

                if self.has_bkg:
                    roi.spec_err_tofit = ma.masked_array(spec_err, roi.mask_tofit)
                    roi.tb_rms_tofit = ma.masked_array(self.bkg_roi.tb_rms, roi.mask_tofit)

    def init_pgdistspec_widget(self):
        """Use Pyqtgraph's PlotWidget for the distance-spectral plot"""
        plot = pg.PlotItem(labels={'bottom': ('Frequency [GHz]', ''), 'left': ('Distance [arcsec]', '')})
        self.distSpecCanvasSet[self.new_roi.roi_id] = pg.ImageView(view=plot)
        self.specplotarea.insertWidget(0, self.distSpecCanvasSet[self.new_roi.roi_id])
        for i in range(self.specplotarea.count()):
            self.specplotarea.setStretch(i, 1)

    def update_pgspec(self):
        """Use Pyqtgraph's PlotWidget for the spectral plot"""
        self.speccanvas.clear()
        if self.spec_in_tb:
            spec_bound = self.tb_spec_bound
        else:
            spec_bound = self.flx_spec_bound
        self.speccanvas.setLimits(yMin=np.log10(spec_bound[0]), yMax=np.log10(spec_bound[1]))
        self.plot_pgspec()
        self.pgspec_add_boundbox()

    def update_rois_on_canvas(self):
        plot_item = self.pg_img_canvas.getView()
        for item in plot_item.items:
            if item.__class__.__module__ == pg.graphicsItems.ROI.__name__:
                self.pg_img_canvas.getView().removeItem(item)
        for current_roi in self.rois[self.roi_group_idx]:
            self.pg_img_canvas.getView().addItem(current_roi)
        self.pg_img_canvas.update()
        ##todo: not all the rois on the canvas can not be removed, seems to be random

    def plot_pgspec(self):
        self.update_fitmask()
        if self.spec_in_tb:
            spec_bound = self.tb_spec_bound
        else:
            spec_bound = self.flx_spec_bound

        self.spec_dataplots = []
        self.spec_dataplots_tofit = []
        self.spec_rmsplots = []
        current_roi_group = self.rois[self.roi_group_idx]
        for n, roi in enumerate(current_roi_group):
            if roi.type == "sliceROI":
                if self.spec_in_tb:
                    spectrogram = roi.tb_im
                else:
                    spectrogram = roi.tb_im_flux
                im_pix_siz = np.nanmean([self.meta['header']['CDELT1'], self.meta['header']['CDELT2']])
                self.distSpecCanvasSet[roi.roi_id].clear()
                self.distSpecCanvasSet[roi.roi_id].setImage(spectrogram.T, pos=[self.cfreqs[0], 0],
                                                            scale=[np.nanmean(np.diff(self.cfreqs)), im_pix_siz])
                self.distSpecCanvasSet[roi.roi_id].setColorMap(self.pgcmap)
                view = self.distSpecCanvasSet[roi.roi_id].getView()
                view.invertY(False)
                view.setAspectLocked(False)
                nf_sub, nd_sub = spectrogram.shape
                view.setLimits(xMin=self.cfreqs[0], xMax=self.cfreqs[-1], yMin=0, yMax=nd_sub * im_pix_siz)
            else:
                if n == self.current_roi_idx or (n == len(current_roi_group) + self.current_roi_idx):
                    symbolfill = (n, 9)
                else:
                    symbolfill = None
                if self.spec_in_tb:
                    spec = roi.tb_max
                    spec_bound = self.tb_spec_bound
                else:
                    spec = roi.total_flux
                    spec_bound = self.flx_spec_bound
                if ma.is_masked(roi.freqghz):
                    log_freqghz = np.log10(roi.freqghz.compressed())
                    log_spec = np.log10(spec.compressed())
                else:
                    log_freqghz = np.log10(roi.freqghz)
                    log_spec = np.log10(spec)
                spec_dataplot = self.speccanvas.plot(x=log_freqghz, y=log_spec, pen=None,
                                                     symbol='o', symbolPen=(n, 9), symbolBrush=None)
                spec_dataplot_tofit = self.speccanvas.plot(x=np.log10(roi.freqghz_tofit.compressed()),
                                                           y=np.log10(roi.spec_tofit.compressed()),
                                                           pen=None,
                                                           symbol='o', symbolPen=(n, 9), symbolBrush=symbolfill)

                self.speccanvas.addItem(spec_dataplot)
                self.speccanvas.addItem(spec_dataplot_tofit)
                self.spec_dataplots.append(spec_dataplot)
                self.spec_dataplots_tofit.append(spec_dataplot_tofit)
                # print('ROI', n)
                # print(roi.freqghz_tofit.compressed())
                # print(roi.spec_tofit.compressed())

                # Add errorbar if rms is defined
                if self.has_bkg:
                    # define error in spectrum
                    if self.spec_in_tb:
                        spec_rms = self.bkg_roi.tb_rms
                    else:
                        spec_rms = gstools.sfu2tb(roi.freqghz * 1e9 * u.Hz, self.bkg_roi.tb_rms * u.K,
                                                  area=roi.total_area * u.arcsec ** 2, reverse=True).value
                    # add fractional err in quadrature
                    spec_err = np.sqrt(spec_rms ** 2. + (self.spec_frac_err * spec) ** 2.)
                    spec_rmsplot = self.speccanvas.plot(x=np.log10(self.cfreqs), y=np.log10(spec_rms), pen='k',
                                                        symbol='d', symbolPen='k', symbolBrush=None)
                    self.speccanvas.addItem(spec_rmsplot)
                    self.spec_rmsplots.append(spec_rmsplot)
                else:
                    spec_err = (self.spec_frac_err * spec) ** 2.
                err_bounds_min = np.maximum(spec - spec_err, np.ones_like(spec) * spec_bound[0])
                err_bounds_max = spec + spec_err
                errplot = pg.ErrorBarItem(x=np.log10(roi.freqghz), y=np.log10(spec),
                                          top=np.log10(err_bounds_max) - np.log10(spec),
                                          bottom=np.log10(spec) - np.log10(err_bounds_min), beam=0.025, pen=(n, 9))

                self.speccanvas.addItem(errplot)

            self.fbar = self.speccanvas.plot(x=np.log10([self.pg_img_canvas.timeLine.getXPos()] * 2),
                                             y=[np.log10(spec_bound[0]), np.log10(spec_bound[1])], pen='k')
            self.speccanvas.addItem(self.fbar)
            self.speccanvas.setLimits(yMin=np.log10(spec_bound[0]), yMax=np.log10(spec_bound[1]))
            xax = self.speccanvas.getAxis('bottom')
            yax = self.speccanvas.getAxis('left')
            xax.setLabel("Frequency [GHz]")
            if self.spec_in_tb:
                yax.setLabel("Brightness Temperature [MK]")
            else:
                yax.setLabel("Flux Density [sfu]")
            xax.setTicks([self.xticks, self.xticks_minor])
            yax.setTicks([self.yticks, self.yticks_minor])

    def pgspec_add_boundbox(self):
        # add frequency bound
        self.rgn_freq_bound = pg.LinearRegionItem(brush=(20, 50, 200, 20))
        self.rgn_freq_bound.setZValue(10)
        self.rgn_freq_bound.setRegion([np.log10(self.fit_freq_bound[0]), np.log10(self.fit_freq_bound[1])])
        self.speccanvas.addItem(self.rgn_freq_bound)
        self.rgn_freq_bound.sigRegionChangeFinished.connect(self.update_freq_bound_rgn)

    def update_fbar(self):
        if self.fbar is not None:
            try:
                self.speccanvas.removeItem(self.fbar)
                self.fbar = self.speccanvas.plot(x=np.log10([self.pg_img_canvas.timeLine.getXPos()] * 2), y=[1, 15],
                                                 pen='k')
                self.speccanvas.addItem(self.fbar)
            except:
                pass

    #    def bkg_rgn_select(self):
    #        """Select a region to calculate rms on spectra"""
    #        if self.bkg_selection_button.isChecked():
    #            self.bkg_selection_button.setStyleSheet("background-color : lightblue")
    #            #self.bkg_roi = pg.RectROI([0, 0], [40, 40], pen='w')
    #            #self.pg_img_canvas.addItem(self.bkg_roi)
    #            #self.bkg_rgn_update()
    #            self.bkg_roi.sigRegionChanged.connect(self.bkg_rgn_update)
    #        else:
    #            # if unChecked remove the ROI box from the plot
    #            self.bkg_selection_button.setStyleSheet("background-color : lightgrey")
    #            self.pg_img_canvas.removeItem(self.bkg_roi)

    def bkg_rgn_update(self):
        """Select a region to calculate rms on spectra"""
        bkg_subim = self.bkg_roi.getArrayRegion(self.pgdata, self.pg_img_canvas.getImageItem(), axes=(1, 2))
        nf_bkg, ny_bkg, nx_bkg = bkg_subim.shape
        if self.bkg_roi.pos()[1] < self.y0 + self.ysiz * 0.1:
            self.bkg_roi_label.setAnchor((0, 1))
        else:
            self.bkg_roi_label.setAnchor((0, 0))
        # if self.bkg_roi.pos()[1] < self.y0 + self.ysiz * 0.1:
        self.bkg_roi.freqghz = self.cfreqs
        self.bkg_roi.tb_mean = np.nanmean(bkg_subim, axis=(1, 2))
        self.bkg_roi.tb_rms = np.std(bkg_subim, axis=(1, 2))
        self.bkg_roi.tb_max = np.nanmax(bkg_subim, axis=(1, 2))
        self.bkg_roi.total_pix = ny_bkg * nx_bkg
        # Total area of the ROI in arcsec^2
        self.bkg_roi.total_area = self.bkg_roi.total_pix * self.meta['header']['CDELT1'] * self.meta['header']['CDELT2']
        # Total flux of the ROI in sfu
        self.bkg_roi.total_flux = gstools.sfu2tb(self.bkg_roi.freqghz * 1e9 * u.Hz, self.bkg_roi.tb_mean * u.K,
                                                 area=self.bkg_roi.total_area * u.arcsec ** 2, reverse=True).value
        self.pg_img_bkg_roi_info_widget.setText('[Background] freq: {0:.2f} GHz, '
                                                'T<sub>B</sub><sup>rms</sup>: {1:.2f} MK, '
                                                'T<sub>B</sub><sup>mean</sup>: {2:.2f} MK, Flux: {3:.2f} sfu'.
                                                format(self.pg_freq_current,
                                                       self.bkg_roi.tb_rms[self.pg_freq_idx] / 1e6,
                                                       self.bkg_roi.tb_mean[self.pg_freq_idx] / 1e6,
                                                       self.bkg_roi.total_flux[self.pg_freq_idx]))
        # self.bkg_roi_label = pg.TextItem("Background", anchor=(0, 0), color='w')
        # self.bkg_roi_label.setParentItem(self.bkg_roi)
        self.update_pgspec()

    def findlastsliceROI(self):
        roi_type_list = [roi.type for roi in self.rois[self.roi_group_idx]][::-1]
        if "sliceROI" in roi_type_list:
            return self.rois[self.roi_group_idx][::-1][roi_type_list.index("sliceROI")]
        else:
            return None

    def add_new_roi(self):

        """Add a ROI region to the selection"""
        self.roi_type = self.toolBarButtonGroup.checkedButton().text()
        ischildROI = False
        if self.add2slice.isChecked():
            if self.roi_type in ["LineSegmentROI", "PolyLineROI"]:
                roi_xcen = self.xcen
                roi_ycen = self.ycen
                roi_size = np.array([self.xsiz, self.ysiz]) / 20
            elif self.roi_type in ["PolygonROI"]:
                self.statusBar.showMessage(
                    '"add ROI to Slice" is not yet supported for PolygonROI.')
                return
            else:
                lastsliceROI = self.findlastsliceROI()
                if lastsliceROI is None:
                    self.statusBar.showMessage(
                        'There is no sliceROI existed in the current ROI group, use the sliceROI tools to add one first.')
                    return
                else:
                    ischildROI = True
                    # print(lastsliceROI)
                    childROI = lastsliceROI.childROI
                    nchildROI = len(childROI.keys())

                    pts = [h.pos() for h in lastsliceROI.endpoints]
                    # print(pts)
                    # print(childROI.keys())
                    dist = [0]
                    pts_orig = lastsliceROI.pos()
                    pts_x = [pts[0].x()]
                    pts_y = [pts[0].y()]
                    for idx, pt in enumerate(pts[:-1]):
                        d = pg.Point(pts[idx + 1] - pts[idx])
                        dist.append(d.length())
                        pts_x.append(pts[idx + 1].x())
                        pts_y.append(pts[idx + 1].y())
                    dist = np.cumsum(dist)
                    dnew = np.linspace(dist[0], dist[-1], nchildROI + 1)
                    pts_x = np.array(pts_x) + pts_orig.x()
                    pts_y = np.array(pts_y) + pts_orig.y()
                    pts_xnew = np.interp(dnew, dist, pts_x)
                    pts_ynew = np.interp(dnew, dist, pts_y)
                    roi_xcen = pts_xnew[-1]
                    roi_ycen = pts_ynew[-1]
                    if nchildROI > 0:
                        roi_size = childROI[0].size()
                    else:
                        roi_size = np.array([self.xsiz, self.ysiz]) / 20

                    for idx, (k, v) in enumerate(childROI.items()):
                        v.sigRegionChanged.disconnect(self.calc_roi_spec)
                        v.setPos((pts_xnew[idx] - roi_size[0] / 2.0, pts_ynew[idx] - roi_size[0] / 2.0))
                        v.setSize(roi_size)
                    for idx, (k, v) in enumerate(childROI.items()):
                        v.sigRegionChanged.connect(self.calc_roi_spec)
                    # if nchildROI == 0:
                    #     pass
                    # else:
                    #     pass

                    # if there is no slice roi. return. infobar output: add a slice first.
                    # else:
                    #   if there is childROI
                    #       read the last childROI's size and distribute this childROI
                    #   else
                    #       use default size and place this childROI in the center of the slice.
        else:
            roi_xcen = self.xcen
            roi_ycen = self.ycen
            roi_size = np.array([self.xsiz, self.ysiz]) / 20

        # if ischildROI:
        #     colorid = len(lastsliceROI.childROI.keys())
        # else:
        colorid = len(self.rois[self.roi_group_idx])
        if self.roi_type == "RectROI":
            self.new_roi = pg.RectROI([roi_xcen - roi_size[0] / 2.0,
                                       roi_ycen - roi_size[1] / 2.0], roi_size,
                                      pen=(colorid, 9),
                                      removable=True, centered=True)
            # self.new_roi.setSize(self.new_roi.size(), center=(0.5, 0.5))
            # self.new_roi.setAngle(self.new_roi.angle(), center=(0.5, 0.5))
            self.new_roi.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        elif self.roi_type == "EllipseROI":
            self.new_roi = pg.EllipseROI([roi_xcen - roi_size[0] / 2.0,
                                          roi_ycen - roi_size[1] / 2.0], roi_size,
                                         pen=(colorid, 9), removable=True)
            # self.new_roi.setSize(self.new_roi.size(), center=(0.5, 0.5))
            # self.new_roi.setAngle(self.new_roi.angle(), center=(0.5, 0.5))
        elif self.roi_type == "PolygonROI":
            self.new_roi = pg.PolyLineROI([[roi_xcen - roi_size[0] * 1,
                                            roi_ycen - roi_size[1] * 0.5],
                                           [roi_xcen + roi_size[0] * 0,
                                            roi_ycen - roi_size[1] * 0.5],
                                           [roi_xcen + roi_size[0] * 1,
                                            roi_ycen + roi_size[1] * 0.5],
                                           [roi_xcen - roi_size[0] * 0,
                                            roi_ycen + roi_size[1] * 0.5]],
                                          pen=(colorid, 9), removable=True, closed=True)
            self.new_roi.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        elif self.roi_type == "LineSegmentROI":
            self.new_roi = pg.LineSegmentROI([[roi_xcen - roi_size[0] / 2.0,
                                               roi_ycen - roi_size[1] / 2.0],
                                              [roi_xcen + roi_size[0] / 2.0,
                                               roi_ycen + roi_size[1] / 2.0]],
                                             pen=(colorid, 9),
                                             removable=True)
        elif self.roi_type == "PolyLineROI":
            self.new_roi = PolyLineROIX([[roi_xcen - roi_size[0] / 2.0,
                                          roi_ycen - roi_size[1] / 2.0],
                                         [roi_xcen + roi_size[0] / 2.0,
                                          roi_ycen + roi_size[1] / 2.0]],
                                        pen=(colorid, 9),
                                        removable=True, closed=False)
        else:
            self.new_roi = pg.RectROI([roi_xcen - roi_size[0] / 2.0,
                                       roi_ycen - roi_size[1] / 2.0], roi_size,
                                      pen=(colorid, 9),
                                      removable=True)
            self.new_roi.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        if hasattr(self, 'predefinedroi'):
            self.new_roi = self.predefinedroi

        self.new_roi.ischildROI = ischildROI
        self.pg_img_canvas.addItem(self.new_roi)
        if not self.vis_roi:
            self.new_roi.setVisible(False)
        self.new_roi.freq_mask = np.ones_like(self.cfreqs) * False
        self.new_roi.sigRegionChanged.connect(self.calc_roi_spec)
        self.new_roi.sigRemoveRequested.connect(self.remove_ROI)

        # choose which group to add
        #self.add_to_roigroup_selection()
        if self.roi_type in ["LineSegmentROI", "PolyLineROI"]:
            self.new_roi.type = "sliceROI"
            self.new_roi.childROI = {}
        else:
            self.new_roi.type = "ROI"
            if ischildROI:
                self.new_roi.colorid = colorid
                self.new_roi.parentROI = lastsliceROI
                self.new_roi.childROIkey = nchildROI
                self.new_roi.parentROI.childROI[self.new_roi.childROIkey] = self.new_roi
        print('New ROI has been added to group {}'.format(self.roi_group_idx))
        self.rois[self.roi_group_idx].append(self.new_roi)
        self.new_roi.roi_id = self.current_roi_idx
        self.nroi_current_group = len(self.rois[self.roi_group_idx])

        if self.vis_roi:
            self.roi_selection_widget.clear()
            self.roi_selection_widget.addItems([str(i) for i in range(self.nroi_current_group)])
            self.current_roi_idx = self.nroi_current_group - 1
            self.roi_selection_button.setText(str(self.current_roi_idx))
            self.has_rois = True
        # self.roi_freq_lowbound_selector.setValue(self.data_freq_bound[0])
        # self.roi_freq_hibound_selector.setValue(self.data_freq_bound[1])
            if self.new_roi.type == "sliceROI":
                self.init_pgdistspec_widget()
            self.calc_roi_spec(None)

        # self.roi_slider_rangechange()


    def add_pre_defined_roi(self, serialized_rois):
        self.add2slice.setChecked(False)
        self.vis_roi = False
        for roi_str in serialized_rois:
            cur_roi = self.deserialize_roi(roi_str)
            self.predefinedroi = cur_roi
            self.add_new_roi()
        delattr(self, 'predefinedroi')


    def remove_ROI(self, evt):
        if evt.ischildROI:
            evt.parentROI.childROI.pop(evt.childROIkey)
            chilROI = {}
            for idx, (k, v) in enumerate(evt.parentROI.childROI.items()):
                chilROI[idx] = v
                v.childROIkey = idx
            del evt.parentROI.childROI
            evt.parentROI.childROI = chilROI
        if evt.type in ["sliceROI"]:
            self.distSpecCanvasSet[evt.roi_id].deleteLater()
            for idx, (k, v) in enumerate(evt.childROI.items()):
                self.rois[self.roi_group_idx].remove(v)
                self.pg_img_canvas.removeItem(v)
        self.rois[self.roi_group_idx].remove(evt)

        self.nroi_current_group = len(self.rois[self.roi_group_idx])
        self.roi_selection_widget.clear()
        self.roi_selection_widget.addItems([str(i) for i in range(self.nroi_current_group)])
        self.current_roi_idx = self.nroi_current_group - 1
        for idx, roi in enumerate(self.rois[self.roi_group_idx]):
            if roi.type != "sliceROI":
                roi.setPen((idx, 9))
        self.pg_img_canvas.removeItem(evt)

        if np.count_nonzero(self.rois) < 1:
            self.has_rois = False
            self.speccanvas.clear()
        else:
            self.calc_roi_spec(None)
        del evt

    # self.view.scene().removeItem(evt)

    # ROI = None
    def add_to_roigroup_selection(self):
        items = self.add_to_roigroup_widget.selectedItems()
        if len(items) > 0:
            self.add_to_roigroup_button.setText(items[0].text())
            self.roi_group_idx = int(items[0].text())
            if self.roi_group_idx > len(self.rois)-1:
                self.rois.append([])
        else:
            self.roi_group_idx = 0
        self.roi_group_selection_update()
        self.update_rois_on_canvas()

    def roigroup_selection(self):
        items = self.roigroup_selection_widget.selectedItems()
        if len(items) > 0:
            self.roigroup_selection_button.setText(items[0].text())
            self.roi_group_idx = int(items[0].text())
        else:
            self.roi_group_idx = 0
        self.roi_group_selection_update()
        self.update_rois_on_canvas()

    def roi_selection_action(self):
        items = self.roi_selection_widget.selectedItems()
        if len(items) > 0:
            self.roi_selection_button.setText(items[0].text())
            self.current_roi_idx = int(items[0].text())
        else:
            self.current_roi_idx = len(self.rois[self.roi_group_idx]) - 1
        self.update_pgspec()
        # print(self.current_roi_idx)

    def roi_group_selection_update(self):
        self.nroi_group = len(self.rois)
        self.roigroup_selection_widget.clear()
        self.roigroup_selection_widget.addItems([str(i) for i in range(len(self.rois))])
        self.roigroup_selection_button.setText(str(self.roi_group_idx))
        self.add_to_roigroup_widget.clear()
        self.add_to_roigroup_widget.addItems([str(i) for i in range(len(self.rois) + 1)])
        self.add_to_roigroup_button.setText(str(self.roi_group_idx))
        self.nroi_current_group = len(self.rois[self.roi_group_idx])
        self.roi_selection_widget.clear()
        self.roi_selection_widget.addItems([str(i) for i in range(self.nroi_current_group)])
        self.current_roi_idx = max(self.nroi_current_group - 1,0)
        self.roi_selection_button.setText(str(self.current_roi_idx))
        self.update_pgspec()

    def group_roi_op_selector(self):
        cur_action = self.sender()
        cur_op = cur_action.text()
        if cur_op == 'Save Group':
            roiutils.save_roi_group(self)

    def open_grid_window(self):
        # self.grid_dialog = Grid_Dialog(pygsfit_object=self)
        # self.grid_dialog.accepted.connect(self.handle_grid_rois)
        # self.grid_dialog.exec_()
        #self.grid_dialog = QDialog()
        ui = Grid_Dialog(self,pygsfit_object=self)
        ui.setupUi(ui)
        ui.data_transmitted.connect(self.handle_grid_rois)
        ui.show()
        ui.exec_()
        # self.grid_dialog.on_confirm.connect(self.handle_grid_rois)
        # self.grid_dialog.show()
        # self.grid_dialog.exec_()


    def handle_grid_rois(self, recieved_rois):
        self.vis_roi = True
        if len(recieved_rois[0]) > 20:
            self.vis_roi = False
        self.grid_rois = []
        self.rois.append([])
        self.roi_group_idx += 1
        self.rectButton.setChecked(True)
        for cri, crect in enumerate(recieved_rois[0]):
            self.xcen = crect[0][0]
            self.ycen = crect[0][1]
            self.xsiz = crect[1][0]*20.0
            self.ysiz = crect[1][1]*20.0
            self.add_new_roi()
        self.xsiz, self.ysiz = [self.meta['nx'] * self.dx,
                                    self.meta['ny'] * self.dy]
        self.xcen, self.ycen = [(self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0]
        self.grid_rois = self.rois[-1]
        if len(recieved_rois[0]) >20:
            cmsgBox = QMessageBox()
            cmsgBox.setText("ROIs are save, but the number of ROIs > 20, ROIs will not be displayed")
            cmsgBox.addButton("No Problem", QMessageBox.AcceptRole)
            cmsgBox.exec_()
            self.rois.pop()
            self.roi_group_idx -= 1
            self.nroi_current_group = len(self.rois[self.roi_group_idx])
        self.number_grid_rois = len(self.grid_rois)
        self.grid_roi_number_lcd.display(self.number_grid_rois)
        self.has_grid_rois=True
        self.pixelized_grid_rois=recieved_rois[1]
        self.roi_group_selection_update()
        self.update_rois_on_canvas()

    def group_of_rois_to_binary_map(self):
        #convert a group of ROIs to a map filled with task index
        ##todo convert to binary map for batch mode
        pass

    def exec_customized_rois_window(self):
        try:
            self.customized_rois_Form = QDialog()
            ui = roiutils.roi_dialog(img_size=[self.meta['nx'], self.meta['ny']], cfreqs=self.cfreqs)
            ui.setupUi(self.customized_rois_Form)
            self.customized_rois_Form.show()
            cur_result = self.customized_rois_Form.exec()
            crtf_list = ui.getResult(self.customized_rois_Form)
            return (crtf_list, cur_result)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, 'No EOVSA Image Loaded', 'Load EOVSA Image first!')
            msg_box.exec_()

    def add_manually_defined_rois(self):
        dialog_output = self.exec_customized_rois_window()
        if not dialog_output[1] or len(dialog_output[0]) == 0:
            print('No ROI is added!')
        else:
            roiutils.add_md_rois(self, inp_str_list=dialog_output[0])

    def calc_roi_spec(self, evt):
        # print('=================Update ROI SPEC===============')
        # roi = self.rois[self.roi_group_idx][self.current_roi_idx]
        # roi = self.new_roi
        # print('calc_roi_spec', evt)

        ## if the signal emitter is a child ROI of a sliceROI
        if evt is not None:
            rois2update = [evt]
            if evt.ischildROI:
                ROIsize = evt.size()
                ROIangle = evt.angle()
                ROIpos = evt.pos()
                # print(evt.parentROI.childROI)
                for idx, (k, v) in enumerate(evt.parentROI.childROI.items()):
                    update = 0
                    v.sigRegionChanged.disconnect(self.calc_roi_spec)
                    if ROIsize != v.size():
                        update = 1
                        v.setSize(ROIsize, center=(0.5, 0.5))
                    if ROIangle != v.angle():
                        update = 1
                        v.setAngle(ROIangle, center=(0.5, 0.5))
                    if ROIpos != v.pos():
                        update = 0
                    # print(update,v)
                    if update:
                        rois2update.append(v)
                for idx, (k, v) in enumerate(evt.parentROI.childROI.items()):
                    v.sigRegionChanged.connect(self.calc_roi_spec)
        else:
            rois2update = self.rois[self.roi_group_idx]

        for roi in rois2update:
            ## todo: investigate why using axes = (2, 1) returns entirely different (wrong!) results

            subim = roi.getArrayRegion(self.pgdata,
                                       self.pg_img_canvas.getImageItem(), axes=(1, 2))

            roi.freqghz = self.cfreqs
            roi.freq_bound = copy.copy(self.roi_freq_bound)
            tb2flx_1d = gstools.sfu2tb(roi.freqghz * 1e9 * u.Hz, np.ones_like(roi.freqghz) * u.K,
                                       area=self.meta['header']['CDELT1'] * self.meta['header'][
                                           'CDELT2'] * u.arcsec ** 2, reverse=True).value

            if roi.type == "sliceROI":
                nf_sub, nd_sub = subim.shape
                roi.total_pix = nd_sub

                roi.tb_max = np.nanmax(subim, axis=1)
                roi.tb_mean = np.nanmean(subim, axis=1)

                roi.tb_im = subim

                tb2flx_2d = np.tile(tb2flx_1d, nd_sub).reshape(nd_sub, nf_sub).transpose()
                roi.tb_im_flux = roi.tb_im * tb2flx_2d
            else:
                nf_sub, ny_sub, nx_sub = subim.shape
                roi.total_pix = ny_sub * nx_sub

                roi.tb_max = np.nanmax(subim, axis=(1, 2))
                roi.tb_mean = np.nanmean(subim, axis=(1, 2))

            # Total area of the ROI in arcsec^2
            roi.total_area = roi.total_pix * self.meta['header']['CDELT1'] * self.meta['header']['CDELT2']
            # Total flux of the ROI in sfu
            roi.total_flux = gstools.sfu2tb(roi.freqghz * 1e9 * u.Hz, roi.tb_mean * u.K,
                                            area=roi.total_area * u.arcsec ** 2, reverse=True).value
            # print(tb2flx_1d * roi.tb_mean * roi.total_pix /roi.total_flux)
            if np.sum(roi.freq_mask) > 0:
                roi.freqghz = ma.masked_array(roi.freqghz, roi.freq_mask)
                roi.tb_max = ma.masked_array(roi.tb_max, roi.freq_mask)
                roi.tb_mean = ma.masked_array(roi.tb_mean, roi.freq_mask)
                roi.total_flux = ma.masked_array(roi.total_flux, roi.freq_mask)
        if self.update_gui:
            if 'roi' in vars():
                self.roi_info.setText('[Current ROI] xcen: {0:6.1f}", ycen: {1:6.1f}", '
                                  'xwid: {2:6.1f}", ywid: {3:6.1f}", '
                                  'freq: {4:4.1f} GHz, '
                                  'T<sub>B</sub><sup>max</sup>: {5:5.1f} MK, '
                                  'T<sub>B</sub><sup>mean</sup>: {6:5.1f} MK, Flux: {6:5.1f} sfu'.
                                  format(roi.pos()[0] + roi.size()[0] / 2, roi.pos()[1] + roi.size()[1] / 2,
                                         roi.size()[0], roi.size()[1],
                                         self.pg_freq_current,
                                         roi.tb_max[self.pg_freq_idx] / 1e6,
                                         roi.tb_mean[self.pg_freq_idx] / 1e6,
                                         roi.total_flux[self.pg_freq_idx]))
        # self.update_fitmask()
            self.update_pgspec()

    def combine_roi_group_flux(self):
        """
        Function to combine flux density of a selected ROI group
        @return:
            self.img_tot_fghz: a masked array of frequencies of the selected ROI group
            self.img_tot_flux: a masked array of total flux density of the selected ROI group
            self.img_tot_flux_plot: a handler for pyqtgraph plot of the total flux
        """
        self.spec_in_tb = False
        self.update_pgspec()
        rois_ = self.rois[self.roi_group_idx]
        nroi = len(rois_)
        if nroi > 1:
            for n in range(nroi):
                roi = rois_[n]
                if n == 0:
                    img_tot_flux_arr = rois_[n].total_flux
                    freq_masks = rois_[n].freq_mask
                else:
                    img_tot_flux_arr = ma.vstack((img_tot_flux_arr, rois_[n].total_flux))
                    freq_masks = ma.vstack((freq_masks, rois_[n].freq_mask))
            self.img_tot_flux = ma.max(img_tot_flux_arr, axis=0)
            img_tot_freq_mask = np.logical_and.reduce(freq_masks)
            self.img_tot_fghz = ma.masked_array(self.cfreqs, img_tot_freq_mask)
        else:
            self.img_tot_flux = rois_[0].total_flux
            self.img_tot_fghz = rois_[0].freqghz
        if self.spec_in_tb == False:
            if hasattr(self, 'img_tot_flux_plot'):
                self.speccanvas.removeItem(self.img_tot_flux_plot)
            else:
                self.img_tot_flux_plot = self.speccanvas.plot(x=np.log10(self.img_tot_fghz),
                                                              y=np.log10(self.img_tot_flux),
                                                              pen=dict(color='k', width=4),
                                                              symbol=None, symbolBrush=None)
                self.speccanvas.addItem(self.img_tot_flux_plot)

    def calc_tpcal_factor(self):
        # find out and plot total power spectrum at the given time
        self.spec_in_tb = False
        if hasattr(self, 'eoimg_date') and hasattr(self, 'dspec') and self.is_calibrated_tp:
            t_idx = np.argmin(np.abs(self.dspec['time_axis'] - self.eoimg_date))
            self.tp_spec = self.dspec['dspec'][:, t_idx]
            self.tp_fghz = self.dspec['freq_axis']
            if hasattr(self, 'tp_flux_plot'):
                self.speccanvas.removeItem(self.tp_flux_plot)
            else:
                self.tp_flux_plot = self.speccanvas.plot(x=np.log10(self.tp_fghz), y=np.log10(self.tp_spec),
                                                         pen=dict(color='k', width=8),
                                                         symbol=None, symbolBrush=None)
                self.speccanvas.addItem(self.tp_flux_plot)
            if hasattr(self, 'img_tot_flux') and hasattr(self, 'img_tot_fghz'):
                self.tp_cal_factor = np.ones_like(self.img_tot_flux)
                for n, fghz in enumerate(self.img_tot_fghz):
                    fidx_tp = np.argmin(np.abs(self.tp_fghz - self.img_tot_fghz[n]))
                    self.tp_cal_factor[n] = self.img_tot_flux[n] / self.tp_spec[fidx_tp]
                print('Total Power Calibration Factor updated')
                print(self.tp_cal_factor)
        else:
            print('Either image time or calibrated total power dynamic spectrum does not exist.')

    def apply_tpcal_factor(self):
        if self.apply_tpcal_factor_button.isChecked() == True:
            self.statusBar.showMessage('Apply total power correction factor to data.')
            #self.data[self.pol_select_idx] /= self.tp_cal_factor[:, None, None]
            self.data[self.pol_select_idx] /= self.tp_cal_factor[None, :, None, None]
            self.tpcal_factor_applied = True
            self.calc_roi_spec(None)
        else:
            self.statusBar.showMessage('Unapply total power correction factor to data.')
            #self.data[self.pol_select_idx] *= self.tp_cal_factor[:, None, None]
            self.data[self.pol_select_idx] *= self.tp_cal_factor[None, :, None, None]
            self.tpcal_factor_applied = False
            self.calc_roi_spec(None)

    # def roi_slider_rangechange(self):
    #    self.roi_select_slider.setMaximum(self.nroi - 1)

    # def roi_slider_valuechange(self):
    #    self.current_roi_idx = self.roi_select_slider.value()
    #    self.roi_info.setPlainText('Selected ROI {}'.format(self.current_roi_idx))
    #    self.update_pgspec()

    def roi_freq_lowbound_valuechange(self):
        self.roi_freq_bound[0] = self.roi_freq_lowbound_selector.value()
        self.statusBar.showMessage('Selected Lower Frequency Bound for ROI is {0:.1f} GHz'.
                                   format(self.roi_freq_bound[0]))
        self.update_freq_mask(self.new_roi)
        self.update_pgspec()

    def roi_freq_hibound_valuechange(self):
        self.roi_freq_bound[1] = self.roi_freq_hibound_selector.value()
        self.statusBar.showMessage('Selected Higher Frequency Bound for ROI is {0:.1f} GHz'.
                                   format(self.roi_freq_bound[1]))
        self.update_freq_mask(self.new_roi)
        self.update_pgspec()

    def update_freq_mask(self, roi):
        # unmask everything first
        if ma.is_masked(roi.freqghz):
            roi.freqghz.mask = ma.nomask
            roi.tb_max.mask = ma.nomask
            roi.tb_mean.mask = ma.nomask
            roi.total_flux.mask = ma.nomask

        # update the masks
        roi.freq_bound = self.roi_freq_bound
        roi.freqghz = ma.masked_outside(self.cfreqs, self.roi_freq_bound[0], self.roi_freq_bound[1])
        roi.freq_mask = roi.freqghz.mask
        roi.tb_max = ma.masked_array(roi.tb_max, roi.freq_mask)
        roi.tb_mean = ma.masked_array(roi.tb_mean, roi.freq_mask)
        roi.total_flux = ma.masked_array(roi.total_flux, roi.freq_mask)

    # def roi_grid_size_valuechange(self):
    #     self.roi_grid_size = self.roi_grid_size_selector.value()

    def freq_lowbound_valuechange(self):
        self.fit_freq_bound[0] = self.freq_lowbound_selector.value()
        self.update_pgspec()

    def freq_hibound_valuechange(self):
        self.fit_freq_bound[1] = self.freq_hibound_selector.value()
        self.update_pgspec()

    def spec_frac_err_valuechange(self):
        self.spec_frac_err = self.spec_frac_err_selector.value()
        self.update_pgspec()

    def update_freq_bound_rgn(self):
        self.rgn_freq_bound.setZValue(10)
        min_logfreq, max_logfreq = self.rgn_freq_bound.getRegion()
        self.fit_freq_bound = [10. ** min_logfreq, 10. ** max_logfreq]
        self.update_fitmask()
        self.freq_lowbound_selector.setValue(self.fit_freq_bound[0])
        self.freq_hibound_selector.setValue(self.fit_freq_bound[1])
        for spec_dataplot, spec_dataplot_tofit, spec_rmsplot in \
                zip(self.spec_dataplots, self.spec_dataplots_tofit, self.spec_rmsplots):
            self.speccanvas.removeItem(spec_dataplot)
            self.speccanvas.removeItem(spec_dataplot_tofit)
            self.speccanvas.removeItem(spec_rmsplot)
        self.plot_pgspec()

    def fit_method_selector(self):
        print("Selected Fit Method is: {}".format(self.fit_method_selector_widget.currentText()))
        self.fit_method = self.fit_method_selector_widget.currentText()
        self.init_fit_kws()
        self.update_fit_kws_widgets()

    def ele_function_selector(self):
        print("Selected Electron Distribution Function is: {}".format(self.ele_function_selector_widget.currentText()))
        self.ele_dist = self.ele_function_selector_widget.currentText()
        self.init_params()

    def init_params(self):
        if self.ele_dist == 'powerlaw':
            self.fit_params = lmfit.Parameters()
            self.fit_params.add_many(('Bx100G', 2., True, 0.1, 100., None, None),
                                     ('log_nnth', 5., True, 3., 11, None, None),
                                     ('delta', 4., True, 1., 30., None, None),
                                     ('Emin_keV', 10., False, 1., 100., None, None),
                                     ('Emax_MeV', 10., False, 0.05, 100., None, None),
                                     ('theta', 45., True, 0.01, 89.9, None, None),
                                     ('log_nth', 10, True, 4., 13., None, None),
                                     ('T_MK', 1., False, 0.1, 100, None, None),
                                     ('depth_asec', 5., False, 1., 100., None, None))
            self.fit_params_nvarys = 0
            for key, par in self.fit_params.items():
                if par.vary:
                    self.fit_params_nvarys += 1

            self.fit_function = gstools.GSCostFunctions.SinglePowerLawMinimizerOneSrc
            self.update_fit_param_widgets()

        if self.ele_dist == 'thermal f-f':
            ## todo: thermal free-free cost function to be added
            self.fit_params = lmfit.Parameters()
            self.fit_params.add_many(('theta', 45., True, 0.01, 89.9, None, None),
                                     ('log_nth', 10, True, 4., 13., None, None),
                                     ('T_MK', 1., False, 0.1, 100, None, None),
                                     ('depth_asec', 5., False, 1., 100., None, None),
                                     ('area_asec2', 25., False, 1., 10000., None, None))
            self.fit_params_nvarys = 0
            for key, par in self.fit_params.items():
                if par.vary:
                    self.fit_params_nvarys += 1

            self.update_fit_param_widgets()

        if self.ele_dist == 'thermal f-f + gyrores':
            self.fit_params = lmfit.Parameters()
            self.fit_params.add_many(('Bx100G', 2., True, 0.1, 100., None, None),
                                     ('log_nnth', 5., False, 3., 11, None, None),
                                     ('delta', 4., False, 1., 30., None, None),
                                     ('Emin_keV', 10., False, 1., 100., None, None),
                                     ('Emax_MeV', 10., False, 0.05, 100., None, None),
                                     ('theta', 45., True, 0.01, 89.9, None, None),
                                     ('log_nth', 10, True, 4., 13., None, None),
                                     ('T_MK', 1., True, 0.1, 100, None, None),
                                     ('depth_asec', 5., False, 1., 100., None, None))
            self.fit_params_nvarys = 0
            for key, par in self.fit_params.items():
                if par.vary:
                    self.fit_params_nvarys += 1

            self.fit_function = gstools.GSCostFunctions.Ff_Gyroresonance_MinimizerOneSrc
            self.update_fit_param_widgets()

    def init_fit_kws(self):
        # first refresh the widgets
        if self.fit_method == 'nelder':
            self.fit_kws = {'maxiter': 2000, 'xatol': 0.01, 'fatol': 0.01}
        if self.fit_method == 'basinhopping':
            self.fit_kws = {'niter': 50, 'T': 90., 'stepsize': 0.8,
                            'interval': 25}
        if self.fit_method == 'mcmc':
            self.fit_kws = {'steps': 1000, 'burn': 300, 'thin': 10, 'workers':8}

    def update_fit_kws_widgets(self):
        # first delete every widget for the fit keywords
        if self.fit_kws_box.count() > 0:
            for n in reversed(range(self.fit_kws_box.count())):
                self.fit_kws_box.itemAt(n).widget().deleteLater()

        self.fit_kws_key_widgets = []
        self.fit_kws_value_widgets = []
        for n, key in enumerate(self.fit_kws):
            fit_kws_key_widget = QLabel(key)
            self.fit_kws_box.addWidget(fit_kws_key_widget)
            if type(self.fit_kws[key]) == int:
                fit_kws_value_widget = QSpinBox()
                fit_kws_value_widget.setRange(0, 10000)
                fit_kws_value_widget.setValue(self.fit_kws[key])
                fit_kws_value_widget.valueChanged.connect(self.update_fit_kws)
            if type(self.fit_kws[key]) == float:
                fit_kws_value_widget = QDoubleSpinBox()
                fit_kws_value_widget.setRange(0, 10000)
                fit_kws_value_widget.setValue(self.fit_kws[key])
                fit_kws_value_widget.setDecimals(2)
                fit_kws_value_widget.valueChanged.connect(self.update_fit_kws)
            if type(self.fit_kws[key]) == str:
                fit_kws_value_widget = QTextEdit()
                fit_kws_value_widget.setText(self.fit_kws[key])
                fit_kws_value_widget.valueChanged.connect(self.update_fit_kws)
            self.fit_kws_key_widgets.append(fit_kws_key_widget)
            self.fit_kws_value_widgets.append(fit_kws_value_widget)
            self.fit_kws_box.addWidget(fit_kws_value_widget)

    def update_fit_param_widgets(self):
        # first delete every widget for the fit parameters
        if self.fit_param_box.count() > 0:
            for n in reversed(range(self.fit_param_box.count())):
                self.fit_param_box.itemAt(n).widget().deleteLater()

        self.param_init_value_widgets = []
        self.param_vary_widgets = []
        self.param_min_widgets = []
        self.param_max_widgets = []
        self.param_fit_value_widgets = []
        self.fit_param_box.addWidget(QLabel('Name'), 0, 0)
        self.fit_param_box.addWidget(QLabel('Initial Guess'), 0, 1)
        self.fit_param_box.addWidget(QLabel('Vary'), 0, 2)
        self.fit_param_box.addWidget(QLabel('Minimum'), 0, 3)
        self.fit_param_box.addWidget(QLabel('Maximum'), 0, 4)
        self.fit_param_box.addWidget(QLabel('Fit Results'), 0, 5)
        for n, key in enumerate(self.fit_params):
            # param_layout = QHBoxLayout()
            # param_layout.addWidget(QLabel(key))
            param_init_value_widget = QDoubleSpinBox()
            param_init_value_widget.setDecimals(1)
            param_init_value_widget.setValue(self.fit_params[key].init_value)
            param_init_value_widget.valueChanged.connect(self.update_params)

            param_vary_widget = QCheckBox()
            param_vary_widget.setChecked(self.fit_params[key].vary)
            param_vary_widget.toggled.connect(self.update_params)

            param_min_widget = QDoubleSpinBox()
            param_min_widget.setDecimals(1)
            param_min_widget.setValue(self.fit_params[key].min)
            param_min_widget.valueChanged.connect(self.update_params)

            param_max_widget = QDoubleSpinBox()
            param_max_widget.setDecimals(1)
            param_max_widget.setValue(self.fit_params[key].max)
            param_max_widget.valueChanged.connect(self.update_params)

            param_fit_value_widget = QDoubleSpinBox()
            param_fit_value_widget.setDecimals(1)
            param_fit_value_widget.setValue(self.fit_params[key].value)
            param_fit_value_widget.valueChanged.connect(self.update_params)

            self.fit_param_box.addWidget(QLabel(fit_param_text[key]), n + 1, 0)
            self.fit_param_box.addWidget(param_init_value_widget, n + 1, 1)
            self.fit_param_box.addWidget(param_vary_widget, n + 1, 2)
            self.fit_param_box.addWidget(param_min_widget, n + 1, 3)
            self.fit_param_box.addWidget(param_max_widget, n + 1, 4)
            self.fit_param_box.addWidget(param_fit_value_widget, n + 1, 5)

            self.param_init_value_widgets.append(param_init_value_widget)
            self.param_vary_widgets.append(param_vary_widget)
            self.param_min_widgets.append(param_min_widget)
            self.param_max_widgets.append(param_max_widget)
            self.param_fit_value_widgets.append(param_fit_value_widget)

    def update_fit_kws(self):
        # print('==========Parameters Updated To the Following=======')
        for n, key in enumerate(self.fit_kws):
            if (isinstance(self.fit_kws_value_widgets[n], QSpinBox)
                    or isinstance(self.fit_kws_value_widgets[n], QDoubleSpinBox)):
                self.fit_kws[key] = self.fit_kws_value_widgets[n].value()
            if isinstance(self.fit_kws_value_widgets[n], QLineEdit):
                self.fit_kws[key] = self.fit_kws_value_widgets[n].toPlainText()
        # print(self.fit_kws)

    def update_params(self):
        # print('==========Parameters Updated To the Following=======')
        self.fit_params_nvarys = 0
        for n, key in enumerate(self.fit_params):
            self.fit_params[key].init_value = self.param_init_value_widgets[n].value()
            self.fit_params[key].vary = self.param_vary_widgets[n].isChecked()
            self.fit_params[key].min = self.param_min_widgets[n].value()
            self.fit_params[key].max = self.param_max_widgets[n].value()
            self.fit_params[key].value = self.param_init_value_widgets[n].value()
            if self.fit_params[key].vary:
                self.fit_params_nvarys += 1

    def update_init_guess(self):
        # Paste the fitting result to the initial value
        for n, key in enumerate(self.fit_params):
            self.param_init_value_widgets[n].setValue(self.fit_params_res[key].value)
            self.fit_params[key].init_value = self.fit_params_res[key].value

    def tb_flx_btnstate(self):
        if self.plot_tb_button.isChecked() == True:
            self.statusBar.showMessage('Plot Brightness Temperature')
            self.spec_in_tb = True
            if 'area_asec2' in self.fit_params.keys():
                del self.fit_params['area_asec2']
                self.update_fit_param_widgets()
        else:
            self.statusBar.showMessage('Plot Flux Density')
            self.spec_in_tb = False
            if not 'area_asec2' in self.fit_params.keys():
                self.fit_params.add_many(('area_asec2', 25., False, 1., 10000., None, None))
                self.update_fit_param_widgets()
        self.init_pgspecplot()
        self.update_pgspec()

    def showqlookimg(self):
        for i in range(self.qlookimgbox.count()):
            self.qlookimgbox.itemAt(i).widget().deleteLater()

        if self.qlookimgbutton.isChecked():
            self.qlookimg_canvas = FigureCanvas(Figure(figsize=(6, 4)))
            self.qlookimg_toolbar = NavigationToolbar(self.qlookimg_canvas, self)
            self.qlookimg_axs = [self.qlookimg_canvas.figure.subplots(1, 1)]
            self.qlookimgbox.addWidget(self.qlookimg_canvas)
            self.qlookimgbox.addWidget(self.qlookimg_toolbar)
            self.qlookimgbutton.setArrowType(Qt.DownArrow)
            self.plot_qlookmap()
            if self.qlookdspecbutton.isChecked():
                self.qlookimgbox.parent().setStretch(0, 1)
                self.qlookimgbox.parent().setStretch(1, 2)
                self.qlookimgbox.parent().setStretch(2, 0)
            else:
                self.qlookimgbox.parent().setStretch(0, 0)
                self.qlookimgbox.parent().setStretch(0, 0)
                self.qlookimgbox.parent().setStretch(1, 0)
        else:
            self.qlookimgbutton.setArrowType(Qt.RightArrow)
            self.qlookimgbox.parent().setStretch(0, 0)
            self.qlookimgbox.parent().setStretch(0, 0)
            self.qlookimgbox.parent().setStretch(1, 0)

    def showqlookdspec(self):
        for i in range(self.qlookdspecbox.count()):
            self.qlookdspecbox.itemAt(i).widget().deleteLater()
        if self.qlookdspecbutton.isChecked():
            self.qlookdspec_canvas = FigureCanvas(Figure(figsize=(6, 4)))
            self.qlookdspec_toolbar = NavigationToolbar(self.qlookdspec_canvas, self)
            self.qlookdspec_ax = self.qlookdspec_canvas.figure.subplots(1, 1)
            self.qlookdspecbox.addWidget(self.qlookdspec_canvas)
            self.qlookdspecbox.addWidget(self.qlookdspec_toolbar)
            self.qlookdspecbutton.setArrowType(Qt.DownArrow)
            if self.qlookimgbutton.isChecked():
                self.qlookdspecbox.parent().setStretch(0, 1)
                self.qlookdspecbox.parent().setStretch(1, 2)
                self.qlookdspecbox.parent().setStretch(2, 0)
            else:
                self.qlookdspecbox.parent().setStretch(0, 0)
                self.qlookdspecbox.parent().setStretch(1, 0)
                self.qlookdspecbox.parent().setStretch(2, 0)
            if self.has_dspec:
                self.plot_dspec()
        else:
            self.qlookdspecbutton.setArrowType(Qt.RightArrow)
            self.qlookdspecbox.parent().setStretch(0, 0)
            self.qlookdspecbox.parent().setStretch(1, 0)
            self.qlookdspecbox.parent().setStretch(2, 0)


    def emcee_lnprob_werr(self,params,freqs,spec=None,spec_err=None,spec_in_tb=True):
        #### we are not suppling the spec_err here to ensure that
        #### we get the original unscaled residuals. We will treat
        #### the errors by hand later.
        residual_orig = self.fit_function(params, freqs, \
                            spec=spec, spec_in_tb=spec_in_tb)
        model = residual_orig + spec
        lnf = float(params['lnf'].value)
        inv_sigma2 = 1.0 / (spec_err ** 2. + model ** 2 * np.exp(2 * lnf))
        num_freqs=freqs.size
        val=((residual_orig) ** 2 * inv_sigma2 - np.log(inv_sigma2))
        #### lmfit expects an array. But these residuals will anyway be sqaured and summed up
        #### after multiplying with -0.5. So I am passing an array of same numbers
        return val
        
    def do_spec_fit(self, local_roi_idx=None):
        #update_gui = Flase for paralell fitting
        if self.update_gui:
            local_roi_idx = self.current_roi_idx
        roi = self.rois[self.roi_group_idx][local_roi_idx]
        freqghz_tofit = roi.freqghz_tofit.compressed()
        spec_tofit = roi.spec_tofit.compressed()
        spec_err_tofit = roi.spec_err_tofit.compressed()
        max_nfev = 1000
        ## Set up fit keywords
        fit_kws = self.fit_kws
        if self.fit_method == 'basinhopping':
            fit_kws['minimizer_kwargs'] = {'method': 'Nelder-Mead'}
            max_nfev *= self.fit_kws['niter'] * (self.fit_params_nvarys + 1)
        if self.fit_method == 'nelder' or self.fit_method == 'mcmc':
            fit_kws = {'options': self.fit_kws}

        if self.update_gui:
            if hasattr(self, 'spec_fitplot'):
                self.speccanvas.removeItem(self.spec_fitplot)
        if self.fit_function != gstools.GSCostFunctions.SinglePowerLawMinimizerOneSrc:
            print("Not yet implemented")
        else:
            exported_fittig_info = []
            if self.fit_method != 'mcmc':
                mini = lmfit.Minimizer(self.fit_function, self.fit_params,
                                       fcn_args=(freqghz_tofit,),
                                       fcn_kws={'spec': spec_tofit, 'spec_err': spec_err_tofit,
                                                'spec_in_tb': self.spec_in_tb},
                                       max_nfev=max_nfev, nan_policy='omit')
                method = self.fit_method
                mi = mini.minimize(method=method, **fit_kws)
                exported_fittig_info.append(mi)
                print(method + ' minimization results')
                print(lmfit.fit_report(mi, show_correl=True))

                print('==========Fit Parameters Updated=======')
                if self.update_gui:
                    self.fit_params_res = mi.params
                    for n, key in enumerate(self.fit_params_res):
                        self.param_fit_value_widgets[n].setValue(self.fit_params_res[key].value)

                freqghz_toplot = np.logspace(0, np.log10(20.), 100)
                spec_fit_res = self.fit_function(mi.params, freqghz_toplot, spec_in_tb=self.spec_in_tb)
                if self.update_gui:
                    self.spec_fitplot = self.speccanvas.plot(x=np.log10(freqghz_toplot), y=np.log10(spec_fit_res),
                                                         pen=dict(color=pg.mkColor(local_roi_idx), width=4),
                                                         symbol=None, symbolBrush=None)
                    self.speccanvas.addItem(self.spec_fitplot)
            else:
                fit_kws_nelder = {'maxiter': 2000, 'tol': 0.01}
                mini = lmfit.Minimizer(self.fit_function, self.fit_params,
                                       fcn_args=(freqghz_tofit,),
                                       fcn_kws={'spec': spec_tofit, 'spec_err': spec_err_tofit,
                                                'spec_in_tb': self.spec_in_tb},
                                       max_nfev=max_nfev, nan_policy='omit')
                method = 'Nelder'
                mi = mini.minimize(method=method, **fit_kws_nelder)
                fit_params_res = mi.params
                fit_params_res.add('lnf', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
                

                burn = self.fit_kws['burn']
                thin = self.fit_kws['thin']
                steps = self.fit_kws['steps']

                emcee_kws = dict(steps=steps, burn=burn, \
                                 thin=thin, is_weighted=True, \
                                 progress=True)
                mini = lmfit.Minimizer(self.emcee_lnprob_werr, fit_params_res,
                                       fcn_args=(freqghz_tofit,),
                                       fcn_kws={'spec': spec_tofit, 'spec_err': spec_err_tofit,
                                               'spec_in_tb': self.spec_in_tb},
                                       max_nfev=max_nfev, nan_policy='omit')
                emcee_params = mini.minimize(method='emcee', **emcee_kws)

                print(lmfit.report_fit(emcee_params.params))
                if self.update_gui:
                    chain = emcee_params.flatchain
                    shape = chain.shape[0]
                    for n, key in enumerate(self.fit_params_res):
                        if key=='lnf':
                            continue
                        try:
                            self.param_fit_value_widgets[n].setValue(np.median(chain[key][burn:]))
                        except KeyError:
                            pass
                    freqghz_toplot = np.logspace(0, np.log10(20.), 100)

                    for i in range(burn, shape, thin):
                        for n, key in enumerate(self.fit_params_res):
                            try:
                                mi.params[key].value = chain[key][i]
                            except KeyError:
                                pass
                        spec_fit_res = self.fit_function(mi.params, freqghz_toplot, spec_in_tb=self.spec_in_tb)
                        self.spec_fitplot = self.speccanvas.plot(x=np.log10(freqghz_toplot), y=np.log10(spec_fit_res),
                                                                 pen=dict(color=pg.mkColor(local_roi_idx), width=4),
                                                                 symbol=None, symbolBrush=None)
                        self.spec_fitplot.setAlpha(0.01, False)
                        self.speccanvas.addItem(self.spec_fitplot)
                else:
                    freqghz_toplot = np.logspace(0, np.log10(20.), 100)
                mi.params = emcee_params.params
                exported_fittig_info.append((emcee_params, mi))

            if self.savedata == True:
                file_write_lock = threading.Lock()
                with file_write_lock:
                    #self.save_res_to_hdf5_file(local_roi_idx, mi, freqghz_toplot, spec_fit_res)
                    self.save_res_to_fits_file(local_roi_idx, mi, freqghz_toplot, spec_fit_res)

            if not self.update_gui:
                exported_fittig_info.append(freqghz_toplot)
                exported_fittig_info.append(self.fit_method)

    def save_res_to_hdf5_file(self, local_roi_idx, minimiz_res, freqghz_toplot ,spec_fit_res):
        ## Surajit
        roi = self.rois[self.roi_group_idx][local_roi_idx]
        def create_attr_array(fit_params, mi_params, param_name):
            stderr = mi_params[param_name].stderr if mi_params[
                                                         param_name].stderr is not None else np.nan
            return np.array([
                fit_params[param_name].init_value,
                fit_params[param_name].min,
                fit_params[param_name].max,
                mi_params[param_name].value,
                stderr
            ])


        roi_props = (self.rois[self.roi_group_idx][local_roi_idx]).saveState()
        roi_pos = [i for i in roi_props['pos']]
        roi_size = [i for i in roi_props['size']]
        roi_angle = roi_props['angle']
        if self.update_gui:
            filename, ok2 = QFileDialog.getSaveFileName(None,
                                                        "Save the selected group ()",
                                                        os.getcwd(),
                                                        "All Files (*)")
        else:
            # filename = os.path.join(self.batch_fitting_dir, 'roi_{0:0=3d}.hdf5'.format(local_roi_idx))
            filename = os.path.join(self.tmp_save_folder, 'roi_{0:0=3d}.hdf5'.format(local_roi_idx))
            ok2 = True
        if ok2 or not self.update_gui:
            print('Fitting results will be saved to: ', filename)
            # hf = h5py.File(filename, 'w')
            with h5py.File(filename, 'w') as hf:
                hf.attrs['roi_pos'] = roi_pos
                hf.attrs['roi_size'] = roi_size
                hf.attrs['roi_angle'] = roi_angle
                hf.attrs['image_file'] = self.eoimg_fname
                hf.attrs['spec_file'] = self.eodspec_fname
                hf.attrs['lower_freq'] = self.fit_freq_bound[0]
                hf.attrs['higher_freq'] = self.fit_freq_bound[1]

                param_names = ['Bx100G', 'log_nnth', 'delta', 'Emin_keV', 'Emax_MeV', 'theta',
                               'log_nth', 'T_MK', 'depth_asec']
                for name in param_names:
                    hf.attrs[name] = create_attr_array(self.fit_params, minimiz_res.params, name)

                if self.spec_in_tb:
                    spec = roi.tb_max
                    spec_bound = self.tb_spec_bound
                    spec_rms = self.bkg_roi.tb_rms
                else:
                    spec = roi.total_flux
                    spec_bound = self.flx_spec_bound
                    spec_rms = gstools.sfu2tb(roi.freqghz * 1e9 * u.Hz, self.bkg_roi.tb_rms * u.K,
                                              area=roi.total_area * u.arcsec ** 2, reverse=True).value
                    hf.attrs['area_asec2'] = np.array([self.fit_params['area_asec2'].init_value, \
                                                       self.fit_params['area_asec2'].min,
                                                       self.fit_params['area_asec2'].max, \
                                                       minimiz_res.params['area_asec2'].value,
                                                       minimiz_res.params['area_asec2'].stderr])
                # add fractional err in quadrature
                spec_err = np.sqrt(spec_rms ** 2. + (self.spec_frac_err * spec) ** 2.)
                hf.create_dataset('observed_spectrum', data=spec)
                hf.create_dataset('error', data=spec_err)
                hf.create_dataset('obs_freq', data=self.cfreqs)
                hf.create_dataset('model_freq', data=freqghz_toplot)
                hf.create_dataset('model_spectrum', data=spec_fit_res)
                # hf.close()

    def save_res_to_fits_file(self, local_roi_idx, minimiz_res, freqghz_toplot, spec_fit_res):
        # Create a FITS header with metadata
        roi = self.rois[self.roi_group_idx][local_roi_idx]
        header = fits.Header()
        header['roi_str'] = str(self.serialize_roi(roi))
        header['roi_ang'] = (self.rois[self.roi_group_idx][local_roi_idx]).saveState()['angle']
        header['img_file'] = self.eoimg_fname
        header['spec_file'] = self.eodspec_fname
        header['low_freq'] = self.fit_freq_bound[0]
        header['high_freq'] = self.fit_freq_bound[1]
        header['high_freq'] = self.fit_freq_bound[1]
        header['fit_method'] = self.fit_method
        header['fit_function'] = self.ele_dist

        def create_parameters_table(fit_params, mi_params):
            param_data = []
            for name, param in fit_params.items():
                stderr = mi_params[name].stderr if mi_params[name].stderr is not None else np.nan
                param_data.append((name, param.value, param.min, param.max, param.vary, stderr))

            dtype = [('name', 'S10'), ('value', 'f8'), ('min', 'f8'), ('max', 'f8'), ('vary', 'bool'), ('stderr', 'f8')]
            structured_array = np.array(param_data, dtype=dtype)
            return fits.BinTableHDU.from_columns(structured_array)


        # def create_attr_array(fit_params, mi_params, param_name):
        #     if param_name not in mi_params:
        #         return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])  # Default values for missing parameter
        #     stderr = mi_params[param_name].stderr if mi_params[param_name].stderr is not None else np.nan
        #     return np.array([
        #         fit_params[param_name].init_value if param_name in fit_params else np.nan,
        #         fit_params[param_name].min if param_name in fit_params else np.nan,
        #         fit_params[param_name].max if param_name in fit_params else np.nan,
        #         mi_params[param_name].value,
        #         stderr
        #     ])
        #
        # param_names = ['Bx100G', 'log_nnth', 'delta', 'Emin_keV', 'Emax_MeV', 'theta', 'log_nth', 'T_MK', 'depth_asec', 'area_asec2']
        # for i, name in enumerate(param_names):
        #     header['PARAM{}'.format(i)] = str(create_attr_array(self.fit_params, minimiz_res.params, name))

        primary_hdu = fits.PrimaryHDU(header=header)

        if self.spec_in_tb:
            spec = roi.tb_max
            spec_err = np.sqrt(self.bkg_roi.tb_rms ** 2. + (self.spec_frac_err * spec) ** 2.)
        else:
            spec = roi.total_flux
            spec_rms = gstools.sfu2tb(roi.freqghz * 1e9 * u.Hz, self.bkg_roi.tb_rms * u.K,
                                      area=roi.total_area * u.arcsec ** 2, reverse=True).value
            spec_err = np.sqrt(spec_rms ** 2. + (self.spec_frac_err * spec) ** 2.)

        observed_spectrum_hdu = fits.ImageHDU(spec, name='OBSERVED_SPECTRUM')
        error_hdu = fits.ImageHDU(spec_err, name='ERROR')
        obs_freq_hdu = fits.ImageHDU(self.cfreqs, name='OBS_FREQ')
        model_freq_hdu = fits.ImageHDU(freqghz_toplot, name='MODEL_FREQ')
        model_spectrum_hdu = fits.ImageHDU(spec_fit_res, name='MODEL_SPECTRUM')
        params_table_hdu = create_parameters_table(self.fit_params, minimiz_res.params)

        # Create an HDUList and save to a FITS file
        hdul = fits.HDUList(
            [primary_hdu, observed_spectrum_hdu, error_hdu, obs_freq_hdu, model_freq_hdu, model_spectrum_hdu, params_table_hdu])

        if self.update_gui:
            filename, ok2 = QFileDialog.getSaveFileName(None, "Save the selected group ()", os.getcwd(),
                                                        "FITS Files (*.fits)")
        else:
            filename = os.path.join(self.tmp_save_folder, 'roi_{0:0=3d}.fits'.format(local_roi_idx))
            ok2 = True

        if ok2 or not self.update_gui:
            print('Fitting results will be saved to: ', filename)
            hdul.writeto(filename, overwrite=True)
        # #just for testing
        # def read_fits_file(filename):
        #     with fits.open(filename) as hdul:
        #         primary_hdu = hdul[0]
        #         header = primary_hdu.header
        #         print("Metadata from FITS file:")
        #         for key in header.keys():
        #             print(f"{key}: {header[key]}")
        #         for hdu in hdul[1:]:
        #             print(f"\nData from HDU '{hdu.name}':")
        #             data = hdu.data
        #             print(data)
        # read_fits_file(filename) # for testing

    def update_gui_after_fitting(self):
        ##todo: update GUI after spectral fitting to make paralell mode work in the regular GUI mode
        #pass
        fitting_info = self.exported_fittig_info
        if hasattr(self, 'spec_fitplot'):
            self.speccanvas.removeItem(self.spec_fitplot)
        #--------
        if fitting_info[2]!='mcmc':
            self.fit_params_res = fitting_info[0].params
            spec_fit_res = self.fit_function(fitting_info[0].params, fitting_info[1], spec_in_tb=self.spec_in_tb)
            self.spec_fitplot = self.speccanvas.plot(x=fitting_info[1], y=np.log10(spec_fit_res),
                                                 pen=dict(color=pg.mkColor(self.current_roi_idx), width=4),
                                                 symbol=None, symbolBrush=None)
            self.speccanvas.addItem(self.spec_fitplot)
        else:
            self.fit_params_res = fitting_info[0][0].params
            chain = fitting_info[0][0].flatchain
            shape = chain.shape[0]
            #----------
            burn = self.fit_kws['burn']
            thin = self.fit_kws['thin']
            steps = self.fit_kws['steps']
            for n, key in enumerate(self.fit_params_res):
                try:
                    self.param_fit_value_widgets[n].setValue(np.median(chain[key][burn:]))
                except KeyError:
                    pass
            #freqghz_toplot = np.logspace(0, np.log10(20.), 100)

            for i in range(burn, shape, thin):
                for n, key in enumerate(self.fit_params_res):
                    try:
                        fitting_info[0][1].params[key].value = chain[key][i]
                    except KeyError:
                        pass
                spec_fit_res = self.fit_function(fitting_info[0][1].params, fitting_info[1], spec_in_tb=self.spec_in_tb)
                self.spec_fitplot = self.speccanvas.plot(x=np.log10(fitting_info[1]), y=np.log10(spec_fit_res),
                                                         pen=dict(color=pg.mkColor(self.current_roi_idx), width=4),
                                                         symbol=None, symbolBrush=None)
                self.spec_fitplot.setAlpha(0.01, False)
                self.speccanvas.addItem(self.spec_fitplot)
        for n, key in enumerate(self.fit_params_res):
            self.param_fit_value_widgets[n].setValue(self.fit_params_res[key].value)

    def parallel_fitting(self):

        self.total_tasks = len(self.rois[0])

        self.threadpool.setMaxThreadCount(self.ncpu_spinbox.value())
        self.fit_threads.clear()
        self.calc_roi_spec(None)
        self.update_fitmask()
        self.tmp_folder = tempfile.mkdtemp(dir = self.batch_fitting_dir)
        self.tmp_save_folder = self.tmp_folder
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(self.total_tasks)
        self.timer.start()

        for croi_idx, croi in enumerate(self.rois[0]):
            fit_task = gstools.FitTask(self, croi_idx)
            #fit_task.signals.completed.connect(self.task_completed)
            #fit_task.completed.connect(self.task_completed)
            #fit_task.signals.connect(self.task_completed)
            self.threadpool.start(fit_task)
        self.threadpool.waitForDone()
        self.check_tasks_completion()

        #import shutil
        #shutil.rmtree(self.tmp_folder)



    def check_tasks_completion(self):
        search_pattern = os.path.join(self.tmp_folder, '*.fits')
        fit_res_files = glob.glob(search_pattern)
        #if self.completed_tasks == self.total_tasks:
        if len(fit_res_files) == self.total_tasks:
            self.timer.stop()
            print("{} tasks are finished. ".format(self.total_tasks))
            final_save = os.path.join(self.batch_fitting_dir,
                                      os.path.basename(self.eoimg_fname).replace('.fits', '_fit_res.fits'))
            #self.combine_hdf5_files(final_path=final_save, original_files=hdf5_files)
            self.combine_fits_files(final_path=final_save, original_files=fit_res_files)

    @pyqtSlot()
    def task_completed(self):
        print('The slot is working')
        self.completed_tasks += 1
        # progress = int((self.completed_tasks / self.total_tasks) * 100)
        # self.progress_bar.setValue(progress)
        # self.progress_bar.show()
    def pathBatchFitRes(self):
        # Open the directory selection dialog
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options)
        if directory:
            print(f"Selected directory: {directory}")
        else:
            print("No directory selected, using home instead.")
            directory = os.path.expanduser("~")
        self.batch_fitting_dir = directory

    def closeEvent(self, event):
        if any(thread.isRunning() for thread in self.fit_threads):
            print('Fittings are still going on!')
            pass
        super().closeEvent(event)
    def export_batch_script(self):
        """
        Parameters:
        filename (str):
        """
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Save Fit Parameters",
                                                  "spectral_fitting_script.py",
                                                  "Python Files (*.py)",
                                                  options=options)
        if filename:
            if not filename.endswith('.py'):
                filename += '.py'
        # roi_filename = os.path.join(os.path.dirname(filename), 'roi_save.p')
        # if os.path.exists(roi_filename):
        #     os.path.join(os.path.dirname(roi_filename),
        #                  'T' + datetime.datetime.now().strftime("%H%M%S") + os.path.basename(roi_filename))
        # with open(roi_filename, 'wb') as cfile:
        #     if self.has_grid_rois:
        #         dill.dump([self.grid_rois], cfile)
        #     else:
        #         dill.dump([self.rois[self.roi_group_idx]], cfile)
        lines = ["import lmfit\n", "from pygsfit.pygsfit import App\n", "import dill\n"
                 "from utils import gstools\n","fit_params = lmfit.Parameters()\n","from PyQt5.QtWidgets import QApplication\n",
                 "import sys\n", "app = QApplication(sys.argv)\n","from numpy import *\n",
                 "f_obj = App()\n", "f_obj.hide()\n\n"]
        app_attributes_tbw = ['eoimg_fname', 'eoimg_time_seq', 'cur_frame_idx']
        app_attributes = ['fit_params_nvarys', 'fit_kws','eoimg_fitsdata','tp_cal_factor',
            'has_eovsamap', 'has_dspec', 'has_stokes','fit_method',
            'has_bkg', 'has_grid_rois',
            'data_freq_bound', 'tb_spec_bound', 'flx_spec_bound',
            'fit_freq_bound', 'roi_freq_bound', 'spec_frac_err',
            'roi_grid_size',
             'nroi_current_group', 'current_roi_idx',
            'number_grid_rois', 'pol_select_idx',
            'spec_in_tb', 'is_calibrated_tp', 'pixelized_grid_rois'
        ]
        fit_function_dict = {'powerlaw':'gstools.GSCostFunctions.SinglePowerLawMinimizerOneSrc',
                             'thermal f-f + gyrores':'gstools.GSCostFunctions.Ff_Gyroresonance_MinimizerOneSrc'}
        for name, param in self.fit_params.items():
            line = f"fit_params.add('{name}', value={param.value}, vary={param.vary}, min={param.min}, max={param.max})"
            if param.expr is not None:
                line += f", expr='{param.expr}'"
            line += "\n\n"
            lines.append(line)
        # file names
        for attr in app_attributes_tbw:
            value = getattr(self, attr)
            lines.append(f'f_obj.{attr} = {value!r}\n')
        #file selection has to be done before setting the attrs
        lines.append("f_obj.eoimg_files_seq_select_return()\n")
        for attr in app_attributes:
            value = getattr(self, attr)
            lines.append(f'f_obj.{attr} = {value!r}\n')
        lines.append("f_obj.fit_params = fit_params\n")
        lines.append(f"f_obj.fit_function = {fit_function_dict[self.ele_dist]} \n")
        #--------add lines to read rois save file
        if self.has_grid_rois:
            cur_roi_list = self.grid_rois
        else:
            cur_roi_list = self.rois[self.roi_group_idx]
        serialized_rois = [self.serialize_roi(roi) for roi in cur_roi_list]
        lines.append("serialized_rois = " + str(serialized_rois) + "\n")
        lines.append("f_obj.roi_group_idx = 0\n")
        lines.append("f_obj.has_rois = True\n")
        lines.append("f_obj.savedata = True\n")
        lines.append("f_obj.update_gui = False\n")
        lines.append("f_obj.current_roi_idx = 0\n")
        lines.append("f_obj.rois = [[]]\n")
        lines.append("f_obj.add_pre_defined_roi(serialized_rois)\n")
        lines.append("f_obj.pathBatchFitRes()\n")
        lines.append("f_obj.rois_to_fits()\n")
        lines.append("f_obj.completed_tasks = 0\n")
        lines.append("f_obj.parallel_fitting()\n")
        with open(filename, 'w') as file:
            file.writelines(lines)

        print(f"Fit parameters exported to {filename}")

    def serialize_roi(self,roi):
        if isinstance(roi, pg.RectROI):
            position = roi.pos()  # This returns a Point-like object for position
            size = roi.size()  # This returns a Point-like object where x and y represent width and height
            return f"RectROI,{position.x()},{position.y()},{size.x()},{size.y()}"
        elif isinstance(roi, pg.EllipseROI):
            position = roi.pos()  # Same as above for position
            size = roi.size()  # Same as above for size
            return f"EllipseROI,{position.x()},{position.y()},{size.x()},{size.y()}"
        elif isinstance(roi, pg.PolyLineROI):
            points = ','.join(f"{handle['pos'].x()},{handle['pos'].y()}" for handle in roi.handles)
            return f"PolyLineROI,{points}"
        else:
            raise ValueError("Unsupported ROI type")

    def deserialize_roi(self,roi_string):
        parts = roi_string.split(',')
        roi_type = parts[0]

        if roi_type == "RectROI":
            x, y, width, height = map(float, parts[1:])
            return pg.RectROI([x, y], [width, height])  # Adjust to match the constructor of your RectROI class
        elif roi_type == "EllipseROI":
            x, y, width, height = map(float, parts[1:])
            return pg.EllipseROI([x, y], [width, height])  # Adjust to match the constructor of your EllipseROI class
        elif roi_type == "PolyLineROI":
            # Parse the points for PolyLineROI
            point_pairs = parts[1:]
            points = [QPointF(float(point_pairs[i]), float(point_pairs[i + 1])) for i in range(0, len(point_pairs), 2)]
            return pg.PolyLineROI(points)  # Adjust to match the constructor of your PolyLineROI class
        else:
            raise ValueError("Unsupported ROI type")
    # def set_roi_index(self, roi_idx):
    #     self.current_roi_idx = roi_idx
    def rois_to_fits(self):
        arcsec_per_pixel = (self.meta['refmap'].meta['CDELT1'], self.meta['refmap'].meta['CDELT2'])
        roi_map = np.zeros(self.pg_img_canvas.getImageItem().image.shape, dtype=np.int64)
        #for index, roi in tqdm(enumerate(self.rois[0])):
        if hasattr(self, 'pixelized_grid_rois') and self.pixelized_grid_rois is not None:
            if len(self.pixelized_grid_rois) != len(self.rois[0]):
                raise ValueError('pixelized_grid_rois and the rois[0] are not isogenous! Please contact Authors.')
            for index, roi in tqdm(enumerate(self.pixelized_grid_rois), total=len(self.pixelized_grid_rois), desc="Processing ROIs"):
                #print(roi)
                roi_map[roi[0][0]:roi[0][0] + roi[1], roi[0][1]:roi[0][1] + roi[1]] = int(index + 1)
        else:
            for index, roi in tqdm(enumerate(self.rois[0]), total=len(self.rois[0]), desc="Processing ROIs"):
                # Convert ROI position to pixels and calculate size in pixels
                roi_pos_pixels = roiutils.convert_roi_pos_to_pixels(roi, self.meta['refmap'])
                roi_size_pixels = roiutils.calculate_roi_size_in_pixels(roi, arcsec_per_pixel)



                if isinstance(roi, pg.RectROI):
                    roi_map[roi_pos_pixels[0]:roi_pos_pixels[0]+roi_size_pixels[0], roi_pos_pixels[1]:roi_pos_pixels[1]+roi_size_pixels[1]] = int(index+1)
                elif isinstance(roi, pg.EllipseROI):
                    for i in range(roi_pos_pixels[0],roi_pos_pixels[0]+roi_size_pixels[0]):
                        for j in range(roi_pos_pixels[1],roi_pos_pixels[1]+roi_size_pixels[1]):
                        # Ellipse equation for the mask
                            if ((j - roi_pos_pixels[0]) ** 2 / roi_size_pixels[0] ** 2 + (i - roi_pos_pixels[1]) ** 2 /
                            roi_size_pixels[1] ** 2) <= 1:
                                if roi_map[i, j] == 0:  # Check for no overlap
                                    roi_map[i, j] = index + 1
                else:
                    raise ValueError('Polygon not yet implemented')

        # Save the map as a FITS file
        filename  = os.path.join(self.batch_fitting_dir, 'roi_map0-{}.fits'.format(len(self.rois[0])-1))
        if os.path.exists(filename):
            raise ValueError('rois map exists!')
        with fits.open(self.eoimg_fname) as hdul:
            # Extract the header from the primary HDU
            header = hdul[0].header
        hdu = fits.PrimaryHDU(data=roi_map, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename, overwrite=True)
        #fits.writeto(filename, roi_map,header=self.meta, overwrite=True)



    def combine_hdf5_files(self, final_path, original_files):
        with h5py.File(final_path, 'w') as hf_combined:
            index_dataset = []

            for i, file_path in enumerate(original_files):
                with h5py.File(file_path, 'r') as hf:
                    # Create a unique group for this file
                    group_name = f"file_{i}"
                    group = hf_combined.create_group(group_name)

                    # Copy the entire content of the file into this new group
                    for key in hf.keys():
                        hf.copy(hf[key], group)

                    # Append file information to index_dataset
                    index_dataset.append((file_path, i))

            # Optionally, create a dataset for file indices
            hf_combined.create_dataset('file_indices', data=index_dataset)

    def combine_fits_files(self, final_path, original_files):
        with fits.HDUList() as hdul_combined:
            for file in original_files:
                with fits.open(file) as hdul:
                    for hdu in hdul:
                        # Make a copy of the HDU
                        hdu_copy = hdu.copy()

                        # Rename the HDU to avoid conflicts and make identification easier
                        hdu_copy.name = os.path.basename(file).upper().replace('.FITS', '') + "_" + hdu_copy.name

                        # Append the copied HDU to the combined HDU list
                        hdul_combined.append(hdu_copy)

            # Write the combined HDU list to a new file
            hdul_combined.writeto(final_path, overwrite=True)

class fileSeqDialog(QWidget):
    dialog_completed = pyqtSignal(bool, int, int)
    def __init__(self, n_files, parent=None):
        super().__init__(parent)

        # Layout
        layout = QVBoxLayout()
        # Add widgets
        self.label = QLabel("{} files are detected in this sequence, do you want to make a image seq?\nDefine the range here:".format(n_files))
        self.input1 = QLineEdit(self)
        self.input1.setText(str(0))
        self.input2 = QLineEdit(self)
        self.input2.setText(str(n_files-1))

        self.yesButton = QPushButton("Yes", self)
        self.noButton = QPushButton("No", self)

        layout.addWidget(self.label)
        layout.addWidget(self.input1)
        layout.addWidget(self.input2)
        layout.addWidget(self.yesButton)
        layout.addWidget(self.noButton)

        self.setLayout(layout)

        # Connect buttons to functions
        self.yesButton.clicked.connect(self.on_yes)
        self.noButton.clicked.connect(self.on_no)

    def on_yes(self):
        value1 = self.input1.text()
        value2 = self.input2.text()

        if not value1 or not value2:
            self.warningLabel.setText("Please fill in both fields.")
            return

        self.dialog_completed.emit(True, int(value1), int(value2))
        self.close()
    def on_no(self):
        self.dialog_completed.emit(False, 0, 0)
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
