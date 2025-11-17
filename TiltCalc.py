# TiltCalc.py
# Final plugin renamed from TiltFootprint -> TiltCalc
# Features:
# - Column Mapping (always show) with saved previous selections
# - DEM selection dialog (QListWidget) if multiple rasters present
# - Model C (terrain-aware) with sampling 30 m
# - Selected-features mode: 1 feature -> simulation, >=2 -> background task (cancelable)
# - Fallback to flat model if no DEM or DEM invalid
import math
import os

from qgis.PyQt.QtWidgets import (
    QAction, QMessageBox, QProgressBar, QPushButton,
    QDialog, QFormLayout, QLineEdit, QHBoxLayout, QVBoxLayout, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QDialogButtonBox,
    QComboBox
)
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QVariant, QSettings

from qgis.core import (
    QgsProject, QgsFeature, QgsGeometry, QgsField, QgsFields,
    QgsVectorLayer, QgsPointXY, QgsApplication,
    QgsTask, QgsMessageLog, Qgis, QgsRasterLayer,
    QgsCoordinateTransform, QgsCoordinateReferenceSystem
)

# -----------------------------
# DEM Selection Dialog (QListWidget)
# -----------------------------
class DemSelectDialog(QDialog):
    def __init__(self, raster_layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pilih DEM (Raster) untuk perhitungan")
        self.resize(420, 320)
        self.raster_layers = raster_layers

        v = QVBoxLayout()
        lbl = QLabel("Pilih salah satu layer raster (DEM/DTM/DSM):")
        v.addWidget(lbl)

        self.listw = QListWidget()
        for lyr in raster_layers:
            name = f"{lyr.name()}  (id:{lyr.id()})"
            item = QListWidgetItem(name)
            item.setData(256, lyr.id())
            self.listw.addItem(item)
        v.addWidget(self.listw)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)
        self.setLayout(v)

    def get_selected_layer(self):
        sel = self.listw.currentItem()
        if not sel:
            return None
        layer_id = sel.data(256)
        return QgsProject.instance().mapLayer(layer_id)


# -----------------------------
# Column Mapping Dialog (Simple + Remember Last Choice)
# -----------------------------
class ColumnMappingDialog(QDialog):
    def __init__(self, field_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Column Mapping (pilih kolom yang sesuai)")
        self.resize(380, 340)

        self.fields = list(field_names)
        # Use plugin-specific QSettings group name for TiltCalc
        self.settings = QSettings("TiltCalcPlugin", "ColumnMapping")

        vbox = QVBoxLayout()
        form = QFormLayout()

        # Combo boxes
        self.cmb_lat = QComboBox(); self.cmb_lon = QComboBox()
        self.cmb_ant = QComboBox(); self.cmb_elec = QComboBox()
        self.cmb_mech = QComboBox(); self.cmb_vbw = QComboBox()
        self.cmb_hbw = QComboBox(); self.cmb_az = QComboBox()
        self.cmb_site = QComboBox()

        combos = [
            self.cmb_lat, self.cmb_lon, self.cmb_ant, self.cmb_elec,
            self.cmb_mech, self.cmb_vbw, self.cmb_hbw, self.cmb_az, self.cmb_site
        ]
        # fill
        for cb in combos:
            cb.addItems(self.fields)

        # attempt auto-detect reasonable defaults (simple heuristics) if no saved settings
        self._apply_auto_detect_defaults()

        # load saved selections (overrides auto-detect if exists)
        self._load_previous_selection()

        form.addRow("Latitude:", self.cmb_lat)
        form.addRow("Longitude:", self.cmb_lon)
        form.addRow("Antenna Height:", self.cmb_ant)
        form.addRow("Electrical Tilt:", self.cmb_elec)
        form.addRow("Mechanical Tilt:", self.cmb_mech)
        form.addRow("Vertical Beamwidth:", self.cmb_vbw)
        form.addRow("Horizontal Beamwidth:", self.cmb_hbw)
        form.addRow("Azimuth:", self.cmb_az)
        form.addRow("Site ID:", self.cmb_site)

        vbox.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

        self.setLayout(vbox)

    def _apply_auto_detect_defaults(self):
        """Try to set sensible defaults by looking for common substrings."""
        lower_fields = [f.lower() for f in self.fields]

        def pick(possible_tokens):
            for tok in possible_tokens:
                for i, name in enumerate(lower_fields):
                    if tok in name:
                        return self.fields[i]
            return None

        # heuristics
        lat_cand = pick(["lat", "latitude", "y_coord", "y"])
        lon_cand = pick(["lon", "long", "longitude", "x_coord", "x"])
        ant_cand = pick(["ant", "height", "antenna", "ant_h"])
        elec_cand = pick(["elec", "elect", "e_tilt", "electrical"])
        mech_cand = pick(["mech", "mechanical", "m_tilt"])
        vbw_cand = pick(["vbeam", "vertical", "vbw"])
        hbw_cand = pick(["hbeam", "horizontal", "hbw", "new_beam"])
        az_cand = pick(["azi", "azimuth", "dir", "direction"])
        site_cand = pick(["site", "cell", "gcell", "site_id", "cell_id"])

        mapping = {
            self.cmb_lat: lat_cand,
            self.cmb_lon: lon_cand,
            self.cmb_ant: ant_cand,
            self.cmb_elec: elec_cand,
            self.cmb_mech: mech_cand,
            self.cmb_vbw: vbw_cand,
            self.cmb_hbw: hbw_cand,
            self.cmb_az: az_cand,
            self.cmb_site: site_cand
        }
        for cb, val in mapping.items():
            if val and val in self.fields:
                cb.setCurrentText(val)

    def _load_previous_selection(self):
        keys = ["LAT","LON","ANT","ELEC","MECH","VBW","HBW","AZ","SITE"]
        combos = [self.cmb_lat,self.cmb_lon,self.cmb_ant,self.cmb_elec,
                  self.cmb_mech,self.cmb_vbw,self.cmb_hbw,self.cmb_az,self.cmb_site]
        for key, cb in zip(keys, combos):
            saved = self.settings.value(key, "")
            if saved and saved in self.fields:
                cb.setCurrentText(saved)

    def save_selection(self):
        mapping = self.get_mapping()
        for key, val in mapping.items():
            self.settings.setValue(key, val)

    def get_mapping(self):
        return {
            "LAT": self.cmb_lat.currentText(),
            "LON": self.cmb_lon.currentText(),
            "ANT": self.cmb_ant.currentText(),
            "ELEC": self.cmb_elec.currentText(),
            "MECH": self.cmb_mech.currentText(),
            "VBW": self.cmb_vbw.currentText(),
            "HBW": self.cmb_hbw.currentText(),
            "AZ": self.cmb_az.currentText(),
            "SITE": self.cmb_site.currentText()
        }

# -----------------------------
# Simulation dialog (single feature)
# -----------------------------
class SimDialog(QDialog):
    def __init__(self, ant_h, elec, mech, hbw, vbw, az, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulasi Tilt Footprint")

        self.e_ant = QLineEdit(str(ant_h))
        self.e_elec = QLineEdit(str(elec))
        self.e_mech = QLineEdit(str(mech))
        self.e_hbw = QLineEdit(str(hbw))
        self.e_vbw = QLineEdit(str(vbw))
        self.e_az = QLineEdit(str(az))

        form = QFormLayout()
        form.addRow("Antenna Height (m):", self.e_ant)
        form.addRow("Electrical Tilt (°):", self.e_elec)
        form.addRow("Mechanical Tilt (°):", self.e_mech)
        form.addRow("H Beamwidth (°):", self.e_hbw)
        form.addRow("V Beamwidth (°):", self.e_vbw)
        form.addRow("Azimuth (°):", self.e_az)

        info = QLabel("Edit nilai untuk simulasi; klik OK untuk membuat footprint.")
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        hl = QHBoxLayout()
        hl.addWidget(btn_ok)
        hl.addWidget(btn_cancel)

        v = QVBoxLayout()
        v.addWidget(info)
        v.addLayout(form)
        v.addLayout(hl)

        self.setLayout(v)

    def values(self):
        return (
            float(self.e_ant.text()),
            float(self.e_elec.text()),
            float(self.e_mech.text()),
            float(self.e_hbw.text()),
            float(self.e_vbw.text()),
            float(self.e_az.text())
        )

# -----------------------------
# Background task (DEM-aware Model C)
# -----------------------------
class TiltTask(QgsTask):
    def __init__(self, description, rows, dem_layer=None, step_m=30.0, max_dist_m=5000.0):
        super().__init__(description, QgsTask.CanCancel)
        self.rows = rows
        self.dem = dem_layer
        self.step_m = float(step_m)
        self.max_dist_m = float(max_dist_m)

        self.transform = None
        if self.dem is not None and self.dem.isValid():
            try:
                src_crs = QgsCoordinateReferenceSystem("EPSG:4326")
                dst_crs = self.dem.crs()
                self.transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
            except Exception as e:
                QgsMessageLog.logMessage(f"Transform setup failed: {e}", "TiltCalc", Qgis.Warning)
                self.transform = None

        self.errors = 0
        self.result_polygons = []

    def get(self, row, *names):
        for n in names:
            for k in row.keys():
                if k.strip().lower() == n.strip().lower():
                    return row[k]
        raise Exception(f"Kolom tidak ditemukan: {names}")

    def compute_radius_flat(self, h, tilt, vbw):
        h = float(h); tilt = float(tilt); vbw = float(vbw)
        def safe(t):
            if t <= 0: return 1.0
            if t >= 89.999: return 1.0
            try:
                r = h / math.tan(math.radians(t))
                if r < 1: r = 1.0
                if r > 5000: r = 5000.0
                return r
            except:
                return 1.0
        r_main = safe(tilt)
        r_min = safe(tilt + vbw / 2.0)
        r_max = safe(tilt - vbw / 2.0)
        return r_min, r_main, r_max

    def get_elevation(self, lon, lat):
        if self.dem is None or not self.dem.isValid():
            return None
        try:
            p = QgsPointXY(lon, lat)
            if self.transform is not None:
                p_dem = self.transform.transform(p)
            else:
                p_dem = p
            ident = self.dem.dataProvider().identify(p_dem, QgsRasterLayer.IdentifyFormatValue)
            if ident.isValid():
                res = ident.results()
                for v in res.values():
                    try:
                        if v is None: continue
                        return float(v)
                    except:
                        continue
            return None
        except Exception as e:
            QgsMessageLog.logMessage(f"DEM identify error: {e}", "TiltCalc", Qgis.Warning)
            return None

    def find_intercept_distance(self, lon, lat, az_deg, tilt_deg, ant_h_m):
        elev_site = self.get_elevation(lon, lat)
        if elev_site is None:
            elev_site = 0.0
        antenna_elev_msl = elev_site + float(ant_h_m)

        step = max(1.0, float(self.step_m))
        max_d = float(self.max_dist_m)
        tan_tilt = math.tan(math.radians(tilt_deg)) if abs(tilt_deg) > 1e-6 else 0.0

        d = step
        while d <= max_d:
            if self.isCanceled(): return d
            rad = math.radians(az_deg)
            dx = d * math.sin(rad) / 111000.0
            dy = d * math.cos(rad) / 111000.0
            samp_lon = lon + dx
            samp_lat = lat + dy

            terrain_elev = self.get_elevation(samp_lon, samp_lat)
            if terrain_elev is None:
                terrain_elev = 0.0

            beam_alt_msl = antenna_elev_msl - d * tan_tilt
            if beam_alt_msl <= terrain_elev:
                return d
            d += step
        return max_d

    def make_sector_dem(self, lat, lon, az, beam, tilt_deg, ant_h):
        coords = []
        start = az - beam / 2.0
        end = az + beam / 2.0
        for a in range(int(start), int(end) + 1):
            if self.isCanceled(): return []
            d = self.find_intercept_distance(lon, lat, a, tilt_deg, ant_h)
            rad = math.radians(a)
            dx = d * math.sin(rad) / 111000.0
            dy = d * math.cos(rad) / 111000.0
            coords.append((lon + dx, lat + dy))
        coords.append((lon, lat))
        return coords

    def run(self):
        total = len(self.rows)
        count = 0
        try:
            for r in self.rows:
                if self.isCanceled(): return False
                try:
                    site = self.get(r, "SITE_ID")
                    lat = float(self.get(r, "Latitude"))
                    lon = float(self.get(r, "Longitude"))
                    ant_h = float(self.get(r, "Ant_Height"))
                    elec = float(self.get(r, "Elec_Tilt"))
                    mech = float(self.get(r, "Mech_Tilt"))
                    tilt_total = elec + mech
                    vbw = float(self.get(r, "VERTICAL_BEAMWIDTH_DEG"))
                    hbw = float(self.get(r, "new_beam"))
                    az = float(self.get(r, "Dir_logical_new"))
                except Exception:
                    self.errors += 1
                    count += 1
                    if total > 0: self.setProgress(int(count / total * 100))
                    continue

                for tilt_offset, label in [ (vbw/2.0, "MIN"), (0.0, "MAIN"), (-vbw/2.0, "MAX") ]:
                    tilt_variant = tilt_total + tilt_offset
                    coords = self.make_sector_dem(lat, lon, az, hbw, tilt_variant, ant_h)
                    if not coords:
                        try:
                            r_min, r_main, r_max = self.compute_radius_flat(ant_h, tilt_total, vbw)
                            radius = {"MIN": r_min, "MAIN": r_main, "MAX": r_max}.get(label, r_main)
                            pts = []
                            start = az - hbw / 2.0
                            end = az + hbw / 2.0
                            for a in range(int(start), int(end) + 1):
                                rad = math.radians(a)
                                dx = radius * math.sin(rad) / 111000.0
                                dy = radius * math.cos(rad) / 111000.0
                                pts.append((lon + dx, lat + dy))
                            pts.append((lon, lat))
                            coords = pts
                        except:
                            coords = [(lon, lat)]

                    self.result_polygons.append({
                        "site": site,
                        "type": label,
                        "coords": coords
                    })

                count += 1
                if total > 0:
                    self.setProgress(int(count / total * 100))
            return True
        except Exception as e:
            QgsMessageLog.logMessage(f"TiltTask exception: {e}", "TiltCalc", Qgis.Critical)
            return False

    def finished(self, result):
        pass

# -----------------------------
# Main plugin class (TiltCalc)
# -----------------------------
class TiltCalc:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.progress = None
        self.btnCancel = None
        self.cur_task = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if not os.path.exists(icon_path):
            print("ICON NOT FOUND:", icon_path)
        # UI name changed to TiltCalc
        self.action = QAction(QIcon(icon_path), "TiltCalc (DEM-aware, Mapping)", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&TiltCalc", self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&TiltCalc", self.action)

    def status(self, msg, timeout=3000):
        self.iface.mainWindow().statusBar().showMessage(msg, timeout)

    def update_progress(self, pct):
        if self.progress:
            pct = max(0, min(100, pct))
            self.progress.setValue(int(pct))

    # helper to build rows using mapping => internal expected field names
    def build_rows_from_features(self, feats, colmap):
        rows = []
        for f in feats:
            try:
                row = {
                    "SITE_ID": str(f[colmap["SITE"]]),
                    "Latitude": f[colmap["LAT"]],
                    "Longitude": f[colmap["LON"]],
                    "Ant_Height": f[colmap["ANT"]],
                    "Elec_Tilt": f[colmap["ELEC"]],
                    "Mech_Tilt": f[colmap["MECH"]],
                    "VERTICAL_BEAMWIDTH_DEG": f[colmap["VBW"]],
                    "new_beam": f[colmap["HBW"]],
                    "Dir_logical_new": f[colmap["AZ"]]
                }
                rows.append(row)
            except Exception:
                # skip if any missing
                continue
        return rows

    def run(self):
        layer = self.iface.activeLayer()
        if layer is None:
            QMessageBox.warning(None, "No Layer", "Tidak ada layer aktif.")
            return

        sel_count = layer.selectedFeatureCount()
        if sel_count == 0:
            QMessageBox.information(None, "No Selected Feature", "Silakan pilih 1 atau lebih GCELL sebelum menjalankan plugin.")
            return

        # Always show Column Mapping dialog
        field_names = [f.name() for f in layer.fields()]
        if not field_names:
            QMessageBox.critical(None, "Layer Error", "Layer aktif tidak memiliki field.")
            return

        mapdlg = ColumnMappingDialog(field_names, parent=self.iface.mainWindow())
        if mapdlg.exec_() != QDialog.Accepted:
            return
        colmap = mapdlg.get_mapping()
        mapdlg.save_selection()

        # For single-feature simulation: use mapping to read values and show SimDialog
        if sel_count == 1:
            feat = layer.selectedFeatures()[0]
            try:
                lat = float(feat[colmap["LAT"]])
                lon = float(feat[colmap["LON"]])
                ant_h = float(feat[colmap["ANT"]])
                elec = float(feat[colmap["ELEC"]])
                mech = float(feat[colmap["MECH"]])
                vbw = float(feat[colmap["VBW"]])
                hbw = float(feat[colmap["HBW"]])
                az = float(feat[colmap["AZ"]])
            except Exception:
                QMessageBox.warning(None, "Kolom tidak lengkap", "Kolom yang diperlukan tidak ada atau bernilai tidak valid di fitur terpilih.")
                return

            dlg = SimDialog(ant_h, elec, mech, hbw, vbw, az, parent=self.iface.mainWindow())
            if dlg.exec_() == QDialog.Accepted:
                ant2, elec2, mech2, hbw2, vbw2, az2 = dlg.values()
                self.simulate_one(lat, lon, ant2, elec2, mech2, hbw2, vbw2, az2)
            return

        # Multi-feature: detect raster layers
        raster_layers = []
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer) and lyr.isValid():
                raster_layers.append(lyr)

        dem_layer = None
        if len(raster_layers) == 0:
            reply = QMessageBox.question(None, "No raster found", "Tidak ditemukan layer raster di QGIS. Lanjutkan tanpa DEM (fallback ke model flat)?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.No:
                return
            dem_layer = None
        elif len(raster_layers) == 1:
            dem_layer = raster_layers[0]
        else:
            dlg = DemSelectDialog(raster_layers, parent=self.iface.mainWindow())
            if dlg.exec_() == QDialog.Accepted:
                dem_layer = dlg.get_selected_layer()
                if dem_layer is None:
                    QMessageBox.information(None, "DEM tidak dipilih", "Tidak ada DEM terpilih. Proses akan dilanjutkan tanpa DEM.")
            else:
                dem_layer = None

        # collect selected features and build rows
        feats = layer.selectedFeatures()
        rows = self.build_rows_from_features(feats, colmap)
        if not rows:
            QMessageBox.warning(None, "No valid rows", "Tidak ada baris valid untuk diproses setelah mapping.")
            return

        # UI progress
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.iface.mainWindow().statusBar().addWidget(self.progress)

        self.btnCancel = QPushButton("Cancel")
        self.iface.mainWindow().statusBar().addWidget(self.btnCancel)

        # Create and start task (sampling 30m)
        task = TiltTask("Processing Selected Features (DEM-aware)", rows, dem_layer, step_m=30.0, max_dist_m=5000.0)
        self.cur_task = task
        task.progressChanged.connect(lambda pct: self.update_progress(pct))

        def _on_finished_wrapper(result, t=task):
            self.task_finished(result, t)
        task.finished = _on_finished_wrapper

        self.btnCancel.clicked.connect(task.cancel)
        QgsApplication.taskManager().addTask(task)
        self.status(f"Started processing {len(rows)} selected features", 3000)

    def task_finished(self, result, task_obj):
        if self.progress:
            self.iface.mainWindow().statusBar().removeWidget(self.progress)
            self.progress = None
        if self.btnCancel:
            self.iface.mainWindow().statusBar().removeWidget(self.btnCancel)
            self.btnCancel = None

        if not result:
            if task_obj.isCanceled():
                self.status("Processing canceled by user.", 5000)
            else:
                self.status("Processing failed.", 5000)
            self.cur_task = None
            return

        polygons = task_obj.result_polygons
        errors = task_obj.errors

        # Output layer name changed to TiltCalc_DEM
        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "TiltCalc_DEM", "memory")
        pr = layer.dataProvider()

        fields = QgsFields()
        fields.append(QgsField("SITE_ID", QVariant.String))
        fields.append(QgsField("TYPE", QVariant.String))
        pr.addAttributes(fields)
        layer.updateFields()

        feats = []
        for rec in polygons:
            try:
                coords = rec["coords"]
                qpts = [QgsPointXY(x, y) for (x, y) in coords]
                geom = QgsGeometry.fromPolygonXY([qpts])
                feat = QgsFeature()
                feat.setFields(layer.fields())
                feat["SITE_ID"] = rec.get("site", "")
                feat["TYPE"] = rec.get("type", "")
                feat.setGeometry(geom)
                feats.append(feat)
            except Exception as e:
                QgsMessageLog.logMessage(f"Error creating feature: {e}", "TiltCalc", Qgis.Warning)
                continue

        if feats:
            pr.addFeatures(feats)
            layer.updateExtents()
            QgsProject.instance().addMapLayer(layer)

        if errors > 0:
            QMessageBox.warning(None, "Warning", f"{errors} baris dilewati karena kolom tidak valid.")
        self.status("Processing completed!", 5000)
        self.cur_task = None

    def simulate_one(self, lat, lon, ant_h, elec, mech, hbw, vbw, az):
        tilt_total = elec + mech
        r_min, r_main, r_max = self.compute_radius(ant_h, tilt_total, vbw)

        # Output simulation layer name changed to TiltCalc_Sim
        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "TiltCalc_Sim", "memory")
        pr = layer.dataProvider()

        fields = QgsFields()
        fields.append(QgsField("TYPE", QVariant.String))
        pr.addAttributes(fields)
        layer.updateFields()

        feats = []
        for radius, label in [(r_min, "MIN"), (r_main, "MAIN"), (r_max, "MAX")]:
            coords = []
            start = az - hbw / 2.0
            end = az + hbw / 2.0
            for a in range(int(start), int(end) + 1):
                rad = math.radians(a)
                dx = radius * math.sin(rad) / 111000
                dy = radius * math.cos(rad) / 111000
                coords.append((lon + dx, lat + dy))
            coords.append((lon, lat))

            qpts = [QgsPointXY(x, y) for (x, y) in coords]
            geom = QgsGeometry.fromPolygonXY([qpts])
            feat = QgsFeature()
            feat.setFields(layer.fields())
            feat["TYPE"] = label
            feat.setGeometry(geom)
            feats.append(feat)

        if feats:
            pr.addFeatures(feats)
            layer.updateExtents()
            QgsProject.instance().addMapLayer(layer)
        self.status("Simulasi selesai!", 4000)

    def compute_radius(self, h, tilt, vbeam):
        h = float(h); tilt = float(tilt); vbeam = float(vbeam)
        def safe(t):
            if t <= 0: return 1.0
            if t >= 89.999: return 1.0
            try:
                r = h / math.tan(math.radians(t))
                if r < 1: r = 1.0
                if r > 5000: r = 5000.0
                return r
            except:
                return 1.0
        r_main = safe(tilt)
        r_min = safe(tilt + vbeam / 2.0)
        r_max = safe(tilt - vbeam / 2.0)
        return r_min, r_main, r_max

# If your plugin system expects a classFactory, you can provide it like:
def classFactory(iface):
    return TiltCalc(iface)
