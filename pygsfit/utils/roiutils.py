import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os
import pickle
from regions import Regions, CRTFRegionParserError
from PyQt5.QtGui import QDesktopServices
import matplotlib.pyplot as plt
from astropy.time import Time
import sunpy
from matplotlib import patches
from matplotlib.backends.backend_qt5agg import FigureCanvas
from astropy.coordinates import SkyCoord
from matplotlib.figure import Figure
from sunpy import map as smap
import astropy.units as u
from pyqtgraph import PolyLineROI
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import LineCollection, QuadMesh
from shapely.geometry import LineString, box, Point
from .gstools import sfu2tb

class LineSegmentROIX(pg.ROI):
    r"""
    ROI subclass with two freely-moving handles defining a line.

    ============== =============================================================
    **Arguments**
    positions      (list of two length-2 sequences) The endpoints of the line
                   segment. Note that, unlike the handle positions specified in
                   other ROIs, these positions must be expressed in the normal
                   coordinate system of the ROI, rather than (0 to 1) relative
                   to the size of the ROI.
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """

    def __init__(self, positions=(None, None), pos=None, handles=(None, None), **args):
        if pos is None:
            pos = [0, 0]

        pg.ROI.__init__(self, pos, [1, 1], **args)
        if len(positions) > 2:
            raise Exception("LineSegmentROI must be defined by exactly 2 positions. For more points, use PolyLineROI.")

        for i, p in enumerate(positions):
            self.addFreeHandle(p, item=handles[i])

    @property
    def endpoints(self):
        # must not be cached because self.handles may change.
        return [h['item'] for h in self.handles]

    def listPoints(self):
        return [p['item'].pos() for p in self.handles]

    def getState(self):
        state = pg.ROI.getState(self)
        state['points'] = [pg.Point(h.pos()) for h in self.getHandles()]
        return state

    def saveState(self):
        state = pg.ROI.saveState(self)
        state['points'] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self, state):
        pg.ROI.setState(self, state)
        p1 = [state['points'][0][0] + state['pos'][0], state['points'][0][1] + state['pos'][1]]
        p2 = [state['points'][1][0] + state['pos'][0], state['points'][1][1] + state['pos'][1]]
        self.movePoint(self.getHandles()[0], p1, finish=False)
        self.movePoint(self.getHandles()[1], p2)

    def paint(self, p, *args):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(self.currentPen)
        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()
        p.drawLine(h1, h2)

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        p = QPainterPath()

        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()
        dh = h2 - h1
        if dh.length() == 0:
            return p
        pxv = self.pixelVectors(dh)[1]
        if pxv is None:
            return p

        pxv *= 4

        p.moveTo(h1 + pxv)
        p.lineTo(h2 + pxv)
        p.lineTo(h2 - pxv)
        p.lineTo(h1 - pxv)
        p.lineTo(h1 + pxv)

        return p

    def getArrayRegion(self, data, img, axes=(0, 1), order=1, returnMappedCoords=False, **kwds):
        """
        Use the position of this ROI relative to an imageItem to pull a slice
        from an array.

        Since this pulls 1D data from a 2D coordinate system, the return value
        will have ndim = data.ndim-1

        See :meth:`~pytqgraph.ROI.getArrayRegion` for a description of the
        arguments.
        """
        print([h.pos() for h in self.endpoints])
        imgPts = [self.mapToItem(img, h.pos()) for h in self.endpoints]
        rgns = []
        coords = []

        d = pg.Point(imgPts[1] - imgPts[0])
        o = pg.Point(imgPts[0])
        rgn = pg.functions.affineSlice(data, shape=(int(d.length()),), vectors=[pg.Point(d.norm())], origin=o,
                                       axes=axes,
                                       order=order, returnCoords=returnMappedCoords, **kwds)
        print(rgn)

        return rgn


class PolyLineROIX(pg.PolyLineROI):
    # def __init__(self, positions, closed=False, pos=None, **args):
    #     if pos is None:
    #         pos = [0, 0]
    #
    #     self.closed = closed
    #     self.segments = []
    #     pg.ROI.__init__(self, pos, size=[1,1], **args)
    #
    #     self.setPoints(positions)

    @property
    def endpoints(self):
        # must not be cached because self.handles may change.
        return [h['item'] for h in self.handles]

    def listPoints(self):
        return [p['item'].pos() for p in self.handles]

    def getArrayRegion(self, data, img, axes=(0, 1), order=1, returnMappedCoords=False, **kwds):
        """
        Use the position of this ROI relative to an imageItem to pull a slice
        from an array.

        Since this pulls 1D data from a 2D coordinate system, the return value
        will have ndim = data.ndim-1

        See :meth:`~pytqgraph.ROI.getArrayRegion` for a description of the
        arguments.
        """
        # print([h.pos() for h in self.endpoints])
        imgPts = [self.mapToItem(img, h.pos()) for h in self.endpoints]
        rgns = []
        coords = []

        rgns = []
        for idx, imgPt in enumerate(imgPts[:-1]):
            d = pg.Point(imgPts[idx + 1] - imgPts[idx])
            o = pg.Point(imgPts[idx])
            rgn = pg.functions.affineSlice(data, shape=(int(d.length()),), vectors=[pg.Point(d.norm())], origin=o,
                                           axes=axes,
                                           order=order, returnCoords=returnMappedCoords, **kwds)
            rgns.append(rgn)

        return np.hstack(rgns)

class roi_dialog(object):
    def __init__(self, img_size, cfreqs):
        self.world = False
        self.img_size = img_size
        self.cfreqs = cfreqs
        self.display_list = []
        # self.input_example = '[[{0}, {1}],[{2}, {3}]],[1.0, 18.0]'.format(int(img_size[0] * 0.3),
        #                                                                  int(img_size[1] * 0.3),
        #                                                                  int(img_size[0] * 0.7),
        #                                                                  int(img_size[1] * 0.7))
        PixSzeLt = [int(self.img_size[0] * 0.1), int(self.img_size[1] * 0.1), int(self.img_size[0] * 0.3),
                    int(self.img_size[1] * 0.3), int(self.img_size[0] * 0.5), int(self.img_size[1] * 0.5)]
        self.input_examples = []
        self.input_examples.append(
            'box[[{0}pix, {1}pix], [{2}pix, {3}pix]], range=[1.0GHz, 18.0GHz]'.format(PixSzeLt[2], PixSzeLt[3],
                                                                                      PixSzeLt[0], PixSzeLt[1]))
        self.input_examples.append(
            'centerbox[[{0}pix, {1}pix], [{2}pix, {3}pix]], range=[1.0GHz, 18.0GHz]'.format(PixSzeLt[4], PixSzeLt[5],
                                                                                            PixSzeLt[0], PixSzeLt[1]))
        self.input_examples.append(
            'rotbox[[{0}pix, {1}pix], [{2}pix, {3}pix], 15deg], range=[1.0GHz, 18.0GHz]'.format(PixSzeLt[2],
                                                                                                PixSzeLt[3],
                                                                                                PixSzeLt[0],
                                                                                                PixSzeLt[1]))
        self.input_examples.append(
            'ellipse[[{0}pix, {1}pix], [{2}pix, {3}pix], 15deg], range=[1.0GHz, 18.0GHz]'.format(PixSzeLt[4],
                                                                                                 PixSzeLt[5],
                                                                                                 PixSzeLt[0],
                                                                                                 PixSzeLt[0] * 0.75))
        self.input_examples.append(
            'circle[[{0}pix, {1}pix], {2}pix], range=[1.0GHz, 18.0GHz]'.format(PixSzeLt[4], PixSzeLt[5], PixSzeLt[0]))
        self.input_example = self.input_examples[0]

    def setupUi(self, Dialog):
        Dialog.setObjectName("Manually Defined ROI(s)")
        Dialog.resize(600, 302)
        self.ok_cancel = QDialogButtonBox(Dialog)
        self.ok_cancel.setGeometry(QRect(500, 40, 81, 71))
        self.ok_cancel.setOrientation(Qt.Vertical)
        self.ok_cancel.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.ok_cancel.setObjectName("ok_cancel")
        self.rois_list_view = QListView(Dialog)
        self.rois_list_view.setGeometry(QRect(10, 10, 481, 200))
        self.rois_list_view.setObjectName("rois_list_view")
        self.slm = QStringListModel()
        self.slm.setStringList(self.display_list)
        self.selected_index = [cur_item.row() for cur_item in self.rois_list_view.selectedIndexes()]

        # listView.clicked.connect(self.checkItem)
        self.rois_list_view.setModel(self.slm)
        self.lineEdit = QLineEdit(Dialog)
        self.lineEdit.setGeometry(QRect(10, 220, 561, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText(self.input_example)
        # self.label = QLabel(Dialog)
        # self.label.setGeometry(QRect(10, 160, 381, 21))
        # self.label.setObjectName("label")
        self.pushButton_url = QPushButton(Dialog)
        self.pushButton_url.setGeometry(QRect(500, 180, 81, 32))
        self.pushButton_url.setObjectName("pushButton_url")
        self.pushButton_url.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://casaguides.nrao.edu/index.php/CASA_Region_Format#Global_definitions")))
        self.pushButton_delete = QPushButton(Dialog)
        self.pushButton_delete.setGeometry(QRect(500, 140, 81, 32))
        self.pushButton_delete.setObjectName("pushButton_delete")
        self.pushButton_delete.clicked.connect(self.delete_from_list)
        # self.checkBox = QCheckBox(Dialog)
        # self.checkBox.setGeometry(QRect(10, 250, 87, 20))
        # self.checkBox.setObjectName("checkBox")
        self.pushButton_add_to_list = QPushButton(Dialog)
        self.pushButton_add_to_list.setGeometry(QRect(470, 250, 113, 32))
        self.pushButton_add_to_list.setObjectName("pushButton")
        self.pushButton_add_to_list.clicked.connect(self.add_to_list)
        self.pushButton_img_flx_preset = QPushButton(Dialog)
        self.pushButton_img_flx_preset.setGeometry(QRect(10, 250, 120, 32))
        self.pushButton_img_flx_preset.setObjectName("img_flx_roi")
        self.pushButton_img_flx_preset.clicked.connect(self.rois_for_cal_flux)
        self.pushButton_load_file = QPushButton(Dialog)
        self.pushButton_load_file.setGeometry(QRect(200, 250, 140, 32))
        self.pushButton_load_file.setObjectName("load_roi_file")
        self.pushButton_load_file.clicked.connect(self.roi_file_select)
        self.shape_comboBox = QComboBox(Dialog)
        self.shape_comboBox.setGeometry(QRect(350, 250, 104, 26))
        self.shape_comboBox.setObjectName("comboBox")
        self.shape_comboBox.addItems([''] * 7)
        # self.label_2 = QLabel(Dialog)
        # self.label_2.setGeometry(QRect(10, 190, 221, 16))
        # self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        self.ok_cancel.accepted.connect(Dialog.accept)
        self.ok_cancel.rejected.connect(Dialog.reject)
        QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Manually Define ROI(s)"))
        # self.label.setText(
        #    _translate("Dialog", "FOV in pixel/arcsec, freq_boundry in GHz"))
        # self.checkBox.setText(_translate("Dialog", "World"))
        self.pushButton_url.setText(_translate("Dialog", "Doc"))
        self.pushButton_img_flx_preset.setText(_translate("Dialog", "EOVSA Presets"))
        self.pushButton_load_file.setText(_translate("Dialog", "Load ROI File"))
        self.pushButton_add_to_list.setText(_translate("Dialog", "Add to List"))
        self.pushButton_delete.setText(_translate("Dialog", "Delete"))
        self.shape_comboBox.setItemText(0, _translate("Dialog", "box"))
        self.shape_comboBox.setItemText(1, _translate("Dialog", "centerbox"))
        self.shape_comboBox.setItemText(2, _translate("Dialog", "rotbox"))
        self.shape_comboBox.setItemText(3, _translate("Dialog", "ellipse"))
        self.shape_comboBox.setItemText(4, _translate("Dialog", "circle"))
        self.shape_comboBox.setItemText(5, _translate("Dialog", "poly"))
        self.shape_comboBox.setItemText(6, _translate("Dialog", "annulus"))
        self.shape_comboBox.currentIndexChanged.connect(self.change_example)
        # self.ok_cancel.accepted.connect(Dialog.accept)
        # self.ok_cancel.rejected.connect(Dialog.reject)
        # self.label_2.setText(_translate("Dialog", "e.g. [[x1,y1],[x2,y2]],[freq_lo,freq_hi]"))

    def add_to_list(self):
        self.input_string = self.lineEdit.text()
        self.cur_info_string = self.input_string
        try:
            tmp_crtf_region = Regions.parse(self.cur_info_string, format='crtf')[0]
            self.display_list.append(self.cur_info_string)
            self.slm.setStringList(self.display_list)
            self.rois_list_view.setModel(self.slm)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, 'Invalid Input!', 'The input is not a vliad CRTF string!')
            msg_box.exec_()

    def delete_from_list(self):
        self.selected_sting = [cur_item.row() for cur_item in self.rois_list_view.selectedIndexes()]
        del self.display_list[self.selected_sting[0]]
        self.slm.setStringList(self.display_list)
        self.rois_list_view.setModel(self.slm)

    def change_example(self):
        self.input_example = self.input_examples[self.shape_comboBox.currentIndex()]
        self.lineEdit.setText(self.input_example)

    # @staticmethod
    def getResult(self, Dialog):
        # dialog = roi_dialog(parent)
        # cur_result = dialog.exec_()
        return self.display_list

    def rois_for_cal_flux(self):
        selections = [str(x) for x in np.arange(2, 6).tolist()]
        seleted_item, okPressed = QInputDialog.getItem(None, "Number of ROIs", "Number of ROIs", selections, 0, False)
        num_of_rois = int(seleted_item)
        print(self.cfreqs)
        if okPressed:
            print(num_of_rois, ' ROIs are created')
            freq_boundry = np.linspace(0, len(self.cfreqs) - 1, num_of_rois + 1, dtype=int)
            size_list = [int(max(100.0 * self.cfreqs[0] / self.cfreqs[freq_ind], 5.)) for freq_ind in
                         freq_boundry[:-1]]
            crtf_str_list = []
            for nri in range(num_of_rois):
                crtf_str_list.append(
                    'centerbox[[{}pix, {}pix], [{}pix, {}pix]], range=[{}GHz, {}GHz]'.format(int(self.img_size[0] / 2),
                                                                                             int(self.img_size[1] / 2),
                                                                                             size_list[nri],
                                                                                             size_list[nri],
                                                                                             self.cfreqs[freq_boundry[nri]],
                                                                                             self.cfreqs[freq_boundry[nri + 1]]))
            for cur_str in crtf_str_list:
                self.display_list.append(cur_str)
            self.slm.setStringList(self.display_list)
            self.rois_list_view.setModel(self.slm)
        else:
            return
    def roi_file_select(self, Dialog):
        cur_file_name, _file_filter = QFileDialog.getOpenFileName(None, 'Select Region save',
                                                                     './', 'Region save (*.crtf *.reg *.ds9 *.fits)')
        #self.fname = 'EOVSA_20210507T190205.000000.outim.image.allbd.fits'
        cur_format = cur_file_name.split('.')[-1]
        if cur_format == 'reg':
            cur_format = 'ds9'
        try:
            cur_region = Regions.read(cur_file_name, format=cur_file_name.split('.')[-1])
        except:
            msg_box = QMessageBox(QMessageBox.Warning, 'Invalid Input!', 'The input can not be converted!')
            msg_box.exec_()
        self.display_list.append(cur_region.serialize(format='crtf'))
        self.slm.setStringList(self.display_list)
        self.rois_list_view.setModel(self.slm)
        return



def crtf_to_pgroi(crtf_str, eo_wcs, pen_arg):
    try:
        crtf_region = Regions.parse(crtf_str, format='crtf')[0]
    except:
        print(
            'check your input string format: https://casaguides.nrao.edu/index.php/CASA_Region_Format#Global_definitions')
    if 'Sky' in str(type(crtf_region)):
        crtf_region = crtf_region.to_pixel(eo_wcs)

    def get_corner_rotate(center_xy, width, height, rot_angle):
        distance = np.sqrt(width ** 2 + height ** 2)
        cur_theta = np.pi / 2. - np.arctan(width / height) - rot_angle
        cor_x = center_xy[0] - distance * np.sin(cur_theta)
        cor_y = center_xy[1] - distance * np.cos(cur_theta)
        return [cor_x, cor_y]

    print(str(type(crtf_region)))
    if 'CirclePixel' in str(type(crtf_region)):
        pg_roi = pg.CircleROI(pos=np.asarray(crtf_region.center.xy) - np.ones(2) * crtf_region.radius,
                              radius=crtf_region.radius, pen=pen_arg)
    elif 'RectanglePixel' in str(type(crtf_region)):
        pg_roi = pg.RectROI(
            pos=get_corner_rotate(center_xy=crtf_region.center.xy, width=crtf_region.width, height=crtf_region.height,
                                  rot_angle=crtf_region.angle.to_value(unit='radian')),
            size=[crtf_region.width, crtf_region.height], angle=crtf_region.angle.to_value(unit='degree'), pen=pen_arg)
    elif 'EllipsePixel' in str(type(crtf_region)):
        pg_roi = pg.EllipseROI(pos=get_corner_rotate(center_xy=crtf_region.center.xy, width=crtf_region.width,
                                                     height=crtf_region.height,
                                                     rot_angle=crtf_region.angle.to_value(unit='radian')),
                               size=[crtf_region.width, crtf_region.height],
                               angle=crtf_region.angle.to_value(unit='degree'), pen=pen_arg)
    # todo: add line ROI
    else:
        raise NameError('Sorry, only rectangle, circle, and ellipse are supported for this moment.')
    freq_range = [1.0, 18.0]
    if 'range' in crtf_region.meta:
        if str(crtf_region.meta['range'][0].unit) == 'GHz':
            freq_range = [crtf_region.meta['range'][0].value, crtf_region.meta['range'][1].value]
            print('Set freq range to {}'.format(freq_range))
    return (pg_roi, freq_range)

def rois_for_cal_flux(self1):
    selections = [str(x) for x in np.arange(2, 6).tolist()]
    seleted_item, okPressed = QInputDialog.getItem(None, "Number of ROIs", "Number of ROIs", selections, 0, False)
    num_of_rois = int(seleted_item)
    if okPressed:
        print(num_of_rois, ' ROIs are created')
        add_new_group(self1)
        freq_boundry = np.linspace(0, len(self1.cfreqs) - 1, num_of_rois + 1, dtype=int)
        size_list = [int(max(100.0 * self1.cfreqs[0] / self1.cfreqs[freq_ind], 5.)) for freq_ind in freq_boundry[:-1]]
        crtf_str_list = []
        for nri in range(num_of_rois):
            crtf_str_list.append(
                'centerbox[[{}pix, {}pix], [{}pix, {}pix]], range=[{}GHz, {}GHz]'.format(int(self1.meta['nx'] / 2),
                                                                                         int(self1.meta['ny'] / 2),
                                                                                         size_list[nri], size_list[nri],
                                                                                         freq_boundry[nri],
                                                                                         freq_boundry[nri + 1]))
        add_md_rois(self1,crtf_str_list)
    else:
        print('No ROI is defined.')


def save_roi_group(self1):
    data_saved = False
    selections = []
    for group_index, cur_group in enumerate(self1.rois):
        selections.append('Group {}, {} ROI(s)'.format(group_index, len(cur_group)))
    seleted_item, okPressed = QInputDialog.getItem(None, "Which Group?", "List of Group", selections, 0, False)
    if okPressed:
        print(seleted_item, ' is selected')
        seleted_group_index = int(seleted_item.split(' ')[1][:-1])
        fileName, ok2 = QFileDialog.getSaveFileName(None,
                                                    "Save the selected group ()",
                                                    os.getcwd(),
                                                    "All Files (*);;pickle save (*.p)")
        if ok2:
            with open(fileName, 'wb') as handle:
                pickle.dump([cur_roi.saveState() for cur_roi in self1.rois[seleted_group_index]], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
                print('The selected group has been saved to ', fileName)
                data_saved = True
    if not data_saved:
        print('No ROI is saved.')


def add_md_rois(self1, inp_str_list):
    add_new_group(self1)
    for si, cur_str in enumerate(inp_str_list):
        add_customized_roi(self1=self1, crtf_str=cur_str)


def add_customized_roi(self1, crtf_str):
    """Add a ROI region to the selection"""
    pg_roi_obj, freq_range = crtf_to_pgroi(crtf_str=crtf_str, eo_wcs=self1.eo_wcs,
                                           pen_arg=(len(self1.rois[self1.roi_group_idx]), 9))
    self1.new_roi = pg_roi_obj
    self1.pg_img_canvas.addItem(self1.new_roi)
    self1.new_roi.freq_mask = np.ones_like(self1.cfreqs) * False
    self1.new_roi.sigRegionChanged.connect(self1.calc_roi_spec)
    # choose which group to add
    self1.add_to_roigroup_selection()
    self1.rois[self1.roi_group_idx].append(self1.new_roi)
    self1.nroi_current_group = len(self1.rois[self1.roi_group_idx])
    self1.roi_selection_widget.clear()
    self1.roi_selection_widget.addItems([str(i) for i in range(self1.nroi_current_group)])
    self1.current_roi_idx = self1.nroi_current_group - 1
    self1.has_rois = True
    self1.calc_roi_spec()
    self1.roi_freq_lowbound_selector.setValue(freq_range[0])
    self1.roi_freq_hibound_selector.setValue(freq_range[1])


def add_new_group(self):
    if self.has_rois:
        self.rois.append([])
        self.roi_group_idx += 1
        if len(self.rois) > self.add_to_roigroup_widget.count():
            self.add_to_roigroup_widget.addItem(str(self.roi_group_idx))
            self.roigroup_selection_widget.addItem(str(self.roi_group_idx))
    else:
        self.roi_group_idx = 0
    self.add_to_roigroup_widget.setCurrentRow(self.roi_group_idx)
    self.add_to_roigroup_button.setText(self.add_to_roigroup_widget.currentItem().text())
    self.roigroup_selection_widget.setCurrentRow(self.roi_group_idx)
    self.roigroup_selection_button.setText(self.roigroup_selection_widget.currentItem().text())


class Grid_Dialog(QDialog):
    data_transmitted = pyqtSignal(object)
    def __init__(self, parent=None, pygsfit_object=None):
        super(Grid_Dialog, self).__init__(parent)
        self.pygsfit_object = pygsfit_object
        self.value_transmitted = False
        self.contour_level = np.array([0.5])
        self.cfreqs = self.pygsfit_object.cfreqs
        self.calpha = 1.0
        self.selected_indices = [0, 2, 5, 10, 20]
        self.num_sides = 12
        self.roi_group_idx = 0
        self.groi_idx = 0
        self.start_freq_idx = 2
        self.end_freq_idx = 30
        self.custom_factor = 6.e6
        self.n_sources = 1
        self.grid_size = 1
        # Plot area
        self.plot_scene = QGraphicsScene()
        self.graphicsView = QGraphicsView()
        self.graphicsView.setScene(self.plot_scene)
        self.roi_canvas = FigureCanvas(Figure(figsize=(8, 7)))
        self.ax = self.roi_canvas.figure.subplots()
        self.plot_scene.addWidget(self.roi_canvas)
        #self.plot_layout.addWidget(self.roi_canvas)
        self.update_plot()
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(893, 680)
        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QRect(530, 590, 341, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.buttonBox.accepted.connect(self.on_confirm)
        self.buttonBox.rejected.connect(Dialog.close)

        self.graphicsView = QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QRect(20, 20, 631, 521))
        self.graphicsView.setObjectName("graphicsView")
        # Plot area
        self.plot_scene = QGraphicsScene()
        self.graphicsView.setScene(self.plot_scene)
        self.roi_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.ax = self.roi_canvas.figure.subplots()
        self.plot_scene.addWidget(self.roi_canvas)
        self.update_plot()

        self.pushButton = QPushButton(Dialog)
        self.pushButton.setGeometry(QRect(160, 590, 91, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.create_polygon_rois)


        self.horizontalSlider = QSlider(Dialog)
        self.horizontalSlider.setGeometry(QRect(730, 40, 111, 20))
        self.horizontalSlider.setOrientation(Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setMinimum(2)
        self.horizontalSlider.setMaximum(99)
        self.horizontalSlider.setValue(50)
        self.horizontalSlider.valueChanged.connect(self.update_plot)


        self.spinBox = QSpinBox(Dialog)
        self.spinBox.setGeometry(QRect(740, 470, 42, 22))
        self.spinBox.setObjectName("spinBox")
        self.spinBox.setValue(self.num_sides)

        self.spinBox_2 = QSpinBox(Dialog)
        self.spinBox_2.setGeometry(QRect(800, 470, 42, 22))
        self.spinBox_2.setObjectName("spinBox_2")
        self.spinBox_2.setValue(self.n_sources)
        self.label = QLabel(Dialog)
        self.label.setGeometry(QRect(730, 450, 60, 16))
        self.label.setObjectName("label")

        self.label_2 = QLabel(Dialog)
        self.label_2.setGeometry(QRect(790, 450, 60, 16))
        self.label_2.setObjectName("label_2")
        self.spinBox_3 = QSpinBox(Dialog)
        self.spinBox_3.setGeometry(QRect(740, 320, 42, 22))
        self.spinBox_3.setObjectName("spinBox_3")
        self.spinBox_3.setValue(self.roi_group_idx)
        self.spinBox_4 = QSpinBox(Dialog)
        self.spinBox_4.setGeometry(QRect(790, 320, 42, 22))
        self.spinBox_4.setObjectName("spinBox_4")
        self.spinBox_4.setValue(self.groi_idx)
        self.label_3 = QLabel(Dialog)
        self.label_3.setGeometry(QRect(730, 300, 60, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QLabel(Dialog)
        self.label_4.setGeometry(QRect(800, 300, 60, 16))
        self.label_4.setObjectName("label_4")
        self.spinBox_5 = QSpinBox(Dialog)
        self.spinBox_5.setGeometry(QRect(740, 420, 42, 22))
        self.spinBox_5.setObjectName("spinBox_5")
        self.spinBox_5.setValue(self.start_freq_idx)
        self.spinBox_5.valueChanged.connect(self.update_plot)

        self.checkbox_u = QCheckBox('CustThresh:', Dialog)
        self.checkbox_u.setGeometry(QRect(690, 100, 100, 16))
        self.checkbox_u.setObjectName("CustThresh")
        self.checkbox_u.setChecked(False)

        self.lineEdit_u = QLineEdit(Dialog)
        self.lineEdit_u.setGeometry(QRect(690, 130, 110, 21))
        self.lineEdit_u.setObjectName("lineEdit_u")
        self.lineEdit_u.setText(str(self.custom_factor))

        self.combo_box_u = QComboBox(Dialog)
        self.combo_box_u.setGeometry(QRect(810, 130, 62, 21))
        self.combo_box_u.setObjectName("combo_box_u")
        self.combo_box_u.addItem('K')
        self.combo_box_u.addItem('sfu')
        self.combo_box_u.addItem('RMS')
        self.combo_box_u.setCurrentIndex(0)

        self.spinBox_6 = QSpinBox(Dialog)
        self.spinBox_6.setGeometry(QRect(800, 420, 42, 22))
        self.spinBox_6.setObjectName("spinBox_6")
        self.spinBox_6.setValue(self.end_freq_idx)
        self.spinBox_6.valueChanged.connect(self.update_plot)

        self.label_5 = QLabel(Dialog)
        self.label_5.setGeometry(QRect(730, 400, 60, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QLabel(Dialog)
        self.label_6.setGeometry(QRect(800, 400, 60, 16))
        self.label_6.setObjectName("label_6")

        self.label_7 = QLabel(Dialog)
        self.label_7.setGeometry(QRect(700, 20, 150, 20))
        self.label_7.setObjectName("label_7")


        self.pushButton_2 = QPushButton(Dialog)
        self.pushButton_2.setGeometry(QRect(50, 590, 100, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.draw_existing_roi)


        self.label_grid_unit = QLabel(Dialog)
        self.label_grid_unit.setGeometry(QRect(810, 180, 61, 31))
        self.label_grid_unit.setObjectName("label_grid_unit")

        self.pushButton_3 = QPushButton(Dialog)
        self.pushButton_3.setGeometry(QRect(740, 245, 100, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.grid_plot)

        self.horizontalSlider_2 = QSlider(Dialog)
        self.horizontalSlider_2.setGeometry(QRect(740, 210, 111, 20))
        self.horizontalSlider_2.setOrientation(Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalSlider_2.setValue(self.grid_size)
        self.horizontalSlider_2.valueChanged.connect(self.update_grid_size)

        self.label_grid_size = QLabel(Dialog)
        self.label_grid_size.setGeometry(QRect(670, 180, 70, 20))
        self.label_grid_size.setObjectName("label_grid_size")
        self.lcdNumber = QLCDNumber(Dialog)
        self.lcdNumber.setGeometry(QRect(740, 180, 64, 23))
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber.setStyleSheet("QLCDNumber { background-color: grey; color: green; }")
        self.lcdNumber.display(self.grid_size)
        self.retranslateUi(Dialog)
        QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Auto Mask"))
        self.label.setText(_translate("Dialog", "nSides"))
        self.label_2.setText(_translate("Dialog", "nSource"))
        self.label_3.setText(_translate("Dialog", "roiGroup"))
        self.label_4.setText(_translate("Dialog", "roi"))
        self.label_5.setText(_translate("Dialog", "fStart"))
        self.label_6.setText(_translate("Dialog", "fEnd"))
        self.label_7.setText(_translate("Dialog", "contour level"))
        self.label_grid_size.setText(_translate("Dialog", "Grid Size:"))
        self.label_grid_unit.setText(_translate("Dialog", "Pix"))
        #self.label_8.setText(_translate("Dialog", "RMS factor"))
        #self.label_check.setText(_translate("Dialog", "use rms"))
        self.pushButton_2.setText(_translate("Dialog", "Existing Roi"))
        self.pushButton_3.setText(_translate("Dialog", "Grid"))


    def update_plot(self):
        try:
            self.contour_level = np.array([self.horizontalSlider.value() / 100.0])
            self.label_7.setText("contour level:{0}%".format(self.horizontalSlider.value()))
            self.start_freq_idx = self.spinBox_5.value()
            self.end_freq_idx = self.spinBox_6.value()
        except:
            pass
        for collection in self.ax.collections:
            #if isinstance(collection, LineCollection) or isinstance(collection, QuadMesh):
            if isinstance(collection, LineCollection):
                    collection.remove()
        # for artist in self.ax.lines + self.ax.collections + self.ax.images:
        #     print('artist', artist)
        #     artist.remove()
        self.selected_indices = np.linspace(self.start_freq_idx, self.end_freq_idx, num=5, endpoint=True).astype(int)
        if self.pygsfit_object.has_eovsamap:
            nspw = self.pygsfit_object.meta['nfreq']
            self.pygsfit_object.eoimg_date = eoimg_date = Time(self.pygsfit_object.meta['refmap'].date.mjd +
                                                self.pygsfit_object.meta['refmap'].exposure_time.value / 2. / 24 / 3600, format='mjd')
            eotimestr = eoimg_date.isot[:-4]
            icmap = plt.get_cmap('RdYlBu')

            self.roi_canvas.figure.suptitle('EOVSA at {}'.format(eotimestr))
        else:
            self.statusBar.showMessage('EOVSA FITS file does not exist', 2000)
            self.eoimg_fname = '<Select or enter a valid fits filename>'
            # self.eoimg_fitsentry.setText(self.eoimg_fname)
            # self.infoEdit.setPlainText('')
        cts=[]
        pos = self.ax.get_position()
        self.roi_canvas.figure.delaxes(self.ax)
        self.ax = self.roi_canvas.figure.add_axes(pos, projection=self.pygsfit_object.meta['refmap'])
        for s, sp in enumerate(self.selected_indices):
            cdata = self.pygsfit_object.data[self.pygsfit_object.pol_select_idx, self.pygsfit_object.cur_frame_idx, sp, ...]
            cur_sunmap = smap.Map(cdata, self.pygsfit_object.meta['refmap'].meta)
            clvls = self.contour_level * np.nanmax(cdata) * u.K
            rcmap = [icmap(self.pygsfit_object.freq_dist(self.pygsfit_object.cfreqs[sp]))] * len(clvls)
            cts.append(cur_sunmap.draw_contours(clvls, axes=self.ax, colors=rcmap, alpha=self.calpha))

        self.ax.set_xlabel('Solar X [arcsec]')
        self.ax.set_ylabel('Solar Y [arcsec]')
        self.ax.set_aspect('equal')
        self.roi_canvas.draw()

    def create_polygon_rois(self):
        for artist in self.ax.patches:
            artist.remove()
        cdata = self.pygsfit_object.data
        self.start_freq_idx = self.spinBox_5.value()
        self.end_freq_idx = self.spinBox_6.value()
        images = cdata[0,self.pygsfit_object.cur_frame_idx, self.start_freq_idx:self.end_freq_idx, ...]
        rms = self.pygsfit_object.bkg_roi.tb_rms[self.start_freq_idx:self.end_freq_idx]
        self.custom_factor = float(self.lineEdit_u.text())
        self.n_sources = self.spinBox_2.value()
        self.num_sides = self.spinBox.value()
        combined_thresholded = np.zeros_like(images[0], dtype=bool)

        for i_idx, image in enumerate(images):
            if self.checkbox_u.isChecked() == True:
                if self.combo_box_u.currentText() == 'K':
                    threshold = self.custom_factor
                elif self.combo_box_u.currentText() == 'sfu':
                    threshold = sfu2tb(self.pygsfit_object.cfreqs[i_idx+self.start_freq_idx] * 1e9 * u.Hz, self.custom_factor * u.jansky * 1e4,
                                              area=self.pygsfit_object.dx * self.pygsfit_object.dy * u.arcsec ** 2, reverse=False).value
                elif self.combo_box_u.currentText() == 'RMS':
                    threshold = rms[i_idx] * self.custom_factor
            else:
                threshold = np.max(image) * self.contour_level[0]
            thresholded_image = image > threshold
            combined_thresholded |= thresholded_image
        #plt.imshow(combined_thresholded, cmap='viridis', interpolation='nearest',origin='lower')
        #plt.show()
        self.labeled_image, self.num_features = ndimage.label(combined_thresholded)
        region_sizes = [(i, np.sum(self.labeled_image == i)) for i in range(1, self.num_features + 1)]
        region_sizes.sort(key=lambda x: x[1], reverse=True)
        self.auto_rois_points = []
        if not hasattr(self,'all_poly'):
            self.all_poly = []
        for ridx_, (region_index, _) in enumerate(region_sizes[:self.n_sources]):
            positions = np.argwhere(self.labeled_image == region_index)
            if positions.size == 0:
                continue
            self.create_polygon_from_points(positions)

            cur_poly = Polygon([[px, py] for px, py in self.poly_points], closed=True, color='blue', alpha=0.5)
            #cur_roi = PolyLineROI([self.pix_to_world(px, py) for px, py in self.poly_points], closed=True)
            self.auto_rois_points.append([self.pix_to_world(px, py) for px, py in self.poly_points])
            self.ax.add_patch(cur_poly)
            self.all_poly.append(cur_poly)

            if ridx_+1 == self.n_sources:
                break
        self.roi_canvas.draw()
    def draw_existing_roi(self):
        ##todo add existing ROI to the plot
        pg_roi = self.pygsfit_object.rois[self.spinBox_3.value()][self.spinBox_4.value()]
        if not hasattr(self, 'all_poly'):
            self.all_poly = []
        if isinstance(pg_roi, pg.RectROI):
            x,y = convert_roi_pos_to_pixels(pg_roi, self.pygsfit_object.meta['refmap'])
            width, height = calculate_roi_size_in_pixels(pg_roi, (self.pygsfit_object.meta['refmap'].meta['CDELT1'], self.pygsfit_object.meta['refmap'].meta['CDELT2']))
            self.poly_points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        elif isinstance(pg_roi, pg.PolyLineROI):
            self.poly_points = convert_roi_pos_to_pixels(pg_roi, self.pygsfit_object.meta['refmap'])
        else:
            raise ValueError("Unsupported ROI type")
        print(self.poly_points)
        cur_poly = Polygon([[px, py] for px, py in self.poly_points], closed=True, color='blue', alpha=0.5)
        #cur_roi = PolyLineROI([self.pix_to_world(px, py) for px, py in self.poly_points], closed=True)
        self.ax.add_patch(cur_poly)
        self.all_poly.append(cur_poly)
        self.roi_canvas.draw()
    def create_polygon_from_points(self, points):
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        if len(hull_points) > self.num_sides:
            indices = np.round(np.linspace(0, len(hull_points) - 1, num=self.num_sides)).astype(int)
            hull_points = hull_points[indices]
        self.poly_points = [sub[::-1] for sub in hull_points.tolist()]

    def grid_plot(self):
        #self.grid_size = self.horizontalSlider_2.value()
        min_x = min_y = np.inf
        max_x = max_y = -np.inf
        for polygon_points in self.all_poly:
            current_min_x, current_min_y = np.min(polygon_points.get_xy(), axis=0)
            current_max_x, current_max_y = np.max(polygon_points.get_xy(), axis=0)
            min_x = min(min_x, current_min_x)
            min_y = min(min_y, current_min_y)
            max_x = max(max_x, current_max_x)
            max_y = max(max_y, current_max_y)
        margin = 10
        self.update_plot()
        #self.create_polygon_rois()
        #self.
        self.ax.set_xlim(min_x - margin, max_x + margin)
        self.ax.set_ylim(min_y - margin, max_y + margin)

        def is_cube_vertex_in_polygon(cube_vertices, polygon_vertices):
            polygon = Polygon(polygon_vertices)
            for iv, vertex in enumerate(cube_vertices):
                point = Point(vertex)
                is_inside = polygon.contains(point)
                if is_inside[0]:
                    return True
            return False

        self.grid_rois_centers = []
        self.grid_pixels = []
        for polygon_points in self.all_poly:
            poly_vertices = polygon_points.get_xy()
            min_x = int(min(vertex[0] for vertex in poly_vertices))
            max_x = int(max(vertex[0] for vertex in poly_vertices))
            min_y = int(min(vertex[1] for vertex in poly_vertices))
            max_y = int(max(vertex[1] for vertex in poly_vertices))
            intersected_cubes = set()
            for x in range(min_x, max_x, self.grid_size):
                for y in range(min_y, max_y, self.grid_size):
                    cube_vertices = [(x, y), (x + self.grid_size, y), (x, y + self.grid_size), (x + self.grid_size, y + self.grid_size)]
                    if is_cube_vertex_in_polygon(cube_vertices, poly_vertices):
                        intersected_cubes.add((x, y))
            grid_size_world = (self.grid_size * self.pygsfit_object.meta['refmap'].meta['CDELT1'],self.grid_size * self.pygsfit_object.meta['refmap'].meta['CDELT2'])
            for (x, y) in sorted(intersected_cubes, key=lambda element: (element[0], element[1])):
                rect = patches.Rectangle((x, y), self.grid_size, self.grid_size, linewidth=0.5, edgecolor='k', facecolor='none')
                world_xy = self.pix_to_world(x,y)
                self.grid_rois_centers.append(([a+b/2.0 for a,b in zip(world_xy, grid_size_world)],grid_size_world))
                self.grid_pixels.append(((x,y),self.grid_size))
                self.ax.add_patch(rect)
        self.roi_canvas.draw()

    def pix_to_world(self,px, py):
        pixel_coord_u = u.Quantity([px, py], u.pix)
        world_coord = self.pygsfit_object.meta['refmap'].pixel_to_world(pixel_coord_u[0], pixel_coord_u[1])
        return [world_coord.Tx.value, world_coord.Ty.value]


    def patch_to_pyRoi(self):
        pass

    def update_slider_label(self, value):
        self.horizontalSlider_label.setText(f"{value}%")
    def update_grid_size(self):
        self.grid_size = self.horizontalSlider_2.value()
        self.lcdNumber.display(self.grid_size)
    def on_confirm(self):
        value = (self.grid_rois_centers,  self.grid_pixels)
        if hasattr(self, 'auto_rois_points'):
            value += (self.auto_rois_points,)
        self.data_transmitted.emit(value)
        self.value_transmitted = True
        self.close()

    def closeEvent(self, event):
        if not self.value_transmitted:
            QMessageBox.warning(self, "Warning", "No data transmitted!")
        event.accept()

def convert_roi_pos_to_pixels(roi, ref_map):
    if isinstance(roi, pg.RectROI):
        world = roi.pos()
        coor = SkyCoord(world[0] * u.arcsec, world[1] * u.arcsec, frame=ref_map.coordinate_frame)
        pix_coor = ref_map.world_to_pixel(coor)
        return int(pix_coor.x.value), int(pix_coor.y.value)
    elif isinstance(roi, pg.PolyLineROI):
        pix_res = []
        for world in roi.positions:
            coor = SkyCoord(world[0] * u.arcsec, world[1] * u.arcsec, frame=ref_map.coordinate_frame)
            pix_coor = ref_map.world_to_pixel(coor)
            pix_res.append((int(pix_coor.x.value), int(pix_coor.y.value)))
        return pix_res
    else:
        print('only rect and ploy supported!')

# Function to calculate ROI size in pixels
def calculate_roi_size_in_pixels(roi, arcsec_per_pixel):
    size = roi.size()
    return int(size[0] / arcsec_per_pixel[0]), int(size[1] / arcsec_per_pixel[1])