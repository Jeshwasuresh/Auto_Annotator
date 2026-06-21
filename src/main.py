"""
AI Auto_Annotator — Windows Edition

Changes from v4:
  • Draw box button REMOVED — just click & drag on canvas anytime (always on)
  • Edit Class / Remove fixed — no CTkToplevel dependency, uses reliable dialog
  • Save Preview Images removed
  • Manual annotation section now only shows: Class selector + Add/Rename/Remove + Undo
"""

import os, sys, json, threading, time, shutil, tempfile, re, multiprocessing
from pathlib import Path
multiprocessing.freeze_support()
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np

_integrity_check = bytes.fromhex(
    "4a65736877612032303235"
).decode("utf-8")
# DO NOT REMOVE — runtime integrity check

# ─────────────────────────────── THEME ───────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
ACCENT      = "#00D4FF"
ACCENT2     = "#FF6B35"
ACCENT3     = "#A855F7"
TRAIN_COL   = "#F0A500"
BG_DARK     = "#0D1117"
BG_CARD     = "#161B22"
BG_PANEL    = "#1C2128"
TEXT_PRI    = "#E6EDF3"
TEXT_MUT    = "#8B949E"
SUCCESS     = "#3FB950"
WARNING     = "#D29922"
DANGER      = "#F85149"
MANUAL_COL  = "#FFD700"

PALETTE = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7","#DDA0DD","#98D8C8",
           "#F7DC6F","#BB8FCE","#85C1E9","#F8C471","#82E0AA","#F1948A","#85929E",
           "#AED6F1","#A9DFBF","#FAD7A0","#D2B4DE","#A3E4D7","#FDEBD0"]

def cls_col(cid):  return PALETTE[int(cid) % len(PALETTE)]
def hex_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))

# ──────────────────────────── DATA MODEL ─────────────────────────────────────
class Annotation:
    def __init__(self, cls_id, cls_name, bbox, confidence=1.0, polygon=None, manual=False):
        self.cls_id     = int(cls_id)
        self.cls_name   = str(cls_name)
        self.bbox       = [int(v) for v in bbox]
        self.confidence = float(confidence)
        self.polygon    = polygon
        self.manual     = manual
        self.selected   = False

    def to_dict(self):
        return {"class_id":self.cls_id,"class_name":self.cls_name,
                "bbox":self.bbox,"confidence":round(self.confidence,4),
                "polygon":self.polygon,"manual":self.manual}

# ──────────────────────────── YOLO WRAPPER ───────────────────────────────────
class YOLOEngine:
    def __init__(self):
        self._base     = None
        self._base_key = None
        self.custom    = None

    def load_base(self, key, cb=None):
        from ultralytics import YOLO
        if self._base_key == key: return
        if cb: cb(f"Loading {key} …")
        self._base = YOLO(key)
        self._base_key = key
        if cb: cb(f"{key} ready ✓")

    def predict(self, bgr, conf=0.25, iou=0.45, seg=False, use_custom=False):
        model = self.custom if (use_custom and self.custom) else self._base
        if model is None: return []
        res = model(bgr, conf=conf, iou=iou, verbose=False)[0]
        out, names = [], res.names
        if seg and res.masks is not None:
            for box, mask in zip(res.boxes, res.masks.xy):
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cid = int(box.cls[0]); c = float(box.conf[0])
                poly = [(int(p[0]),int(p[1])) for p in mask]
                out.append(Annotation(cid, names[cid], [x1,y1,x2,y2], c, poly))
        else:
            for box in res.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cid = int(box.cls[0]); c = float(box.conf[0])
                out.append(Annotation(cid, names[cid], [x1,y1,x2,y2], c))
        return out

    def train(self, yaml_path, base_key, epochs=30, cb=None):
        from ultralytics import YOLO
        if cb: cb("🏋️ Starting training…", 0.0)
        trainer = YOLO(base_key)
        
        import time
        start_time = time.time()
        
        def on_fit_epoch_end(t):
            try:
                current_epoch = t.epoch + 1
                total_epochs = t.args.epochs
                pct = current_epoch / total_epochs
                
                elapsed = time.time() - start_time
                avg_time = elapsed / current_epoch
                eta = (total_epochs - current_epoch) * avg_time
                
                if eta > 60:
                    eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
                else:
                    eta_str = f"{int(eta)}s"
                    
                msg = f"🏋️ Epoch {current_epoch}/{total_epochs} ({int(pct*100)}%) | Elapsed: {int(elapsed)}s | ETA: {eta_str}"
                if cb: cb(msg, pct)
            except Exception:
                pass
                
        trainer.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        res = trainer.train(
            data=yaml_path, epochs=epochs, imgsz=640, batch=4,
            patience=10, verbose=False,
            project=str(Path(yaml_path).parent / "runs"),
            name="run", exist_ok=True,
            workers=0)
        best = Path(res.save_dir) / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError("best.pt not found after training")
        self.custom = YOLO(str(best))
        if cb: cb(f"✅ Done → {best.name}", 1.0)
        return str(best)

YOLO = YOLOEngine()

# ──────────────────────────── DRAWING ────────────────────────────────────────
def draw_anns(frame, anns, scale=1.0, show_conf=True, hi_manual=True):
    out = frame.copy()
    for a in anns:
        col   = MANUAL_COL if (a.manual and hi_manual) else cls_col(a.cls_id)
        bgr   = hex_bgr(col)
        thick = 3 if a.selected else 2
        x1,y1,x2,y2 = [int(v*scale) for v in a.bbox]
        if a.polygon:
            pts = np.array([(int(p[0]*scale),int(p[1]*scale)) for p in a.polygon],np.int32)
            ov = out.copy(); cv2.fillPoly(ov,[pts],bgr)
            cv2.addWeighted(ov,0.25,out,0.75,0,out)
            cv2.polylines(out,[pts],True,bgr,thick)
        else:
            cv2.rectangle(out,(x1,y1),(x2,y2),bgr,thick)
            if a.selected:
                cv2.rectangle(out,(x1-2,y1-2),(x2+2,y2+2),(255,255,255),1)
                # Draw 4 corner handles
                hs = 5
                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                for cx, cy in corners:
                    cv2.rectangle(out, (cx - hs, cy - hs), (cx + hs, cy + hs), (255, 255, 255), -1)
                    cv2.rectangle(out, (cx - hs, cy - hs), (cx + hs, cy + hs), bgr, 1)
        lbl = f"[M]{a.cls_name}" if a.manual else a.cls_name
        if show_conf and not a.manual: lbl += f" {a.confidence:.2f}"
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ly = max(y1-2, th+4)
        cv2.rectangle(out,(x1,ly-th-4),(x1+tw+4,ly),bgr,-1)
        cv2.putText(out,lbl,(x1+2,ly-2),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),1,cv2.LINE_AA)
    return out

# ──────────────────────────── ICON LIST ───────────────────────────────────────
class IconList(ctk.CTkScrollableFrame):
    def __init__(self, parent, on_select_cb, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_select_cb = on_select_cb
        self.items = []       # list of Path
        self.cards = []       # list of dict: {"frame", "img_lbl", "bar", "photo", "done"}
        self.selected_indices = set()
        self._last_selected_indices = set()
        self.load_id = 0      # increment on each clear to cancel previous load
        
        # Create a solid placeholder
        img = Image.new("RGB", (68, 48), "#1F2937")
        self.default_img = ImageTk.PhotoImage(img)
        self.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")

    def clear(self):
        self.load_id += 1 # cancel previous loading thread
        for c in self.cards:
            c["frame"].destroy()
        self.cards.clear()
        self.items.clear()
        self.selected_indices.clear()
        self._last_selected_indices.clear()

    def add_item(self, path):
        idx = len(self.items)
        self.items.append(path)

        # 3 columns layout
        card = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=6, border_width=1, border_color="#21262D", cursor="hand2")
        row = idx // 3
        col = idx % 3
        card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")

        # Thumbnail label
        img_lbl = ctk.CTkLabel(card, image=self.default_img, text="")
        img_lbl.pack(padx=2, pady=(2, 0))

        # Text filename label (smart truncation)
        name_text = path.name
        if len(name_text) > 10:
            name_text = name_text[:6] + "..." + name_text[-3:]
        name_lbl = ctk.CTkLabel(card, text=name_text, font=ctk.CTkFont(size=9), text_color=TEXT_MUT)
        name_lbl.pack(padx=2, pady=(0, 2))

        # Bottom status bar (color-coded)
        bar = ctk.CTkFrame(card, height=4, fg_color=TEXT_MUT, corner_radius=2)
        bar.pack(fill="x", padx=4, pady=2)

        # Click bind
        for widget in (card, img_lbl, name_lbl, bar):
            widget.bind("<Button-1>", lambda e, i=idx: self._on_card_click(i, e))

        self.cards.append({
            "frame": card,
            "img_lbl": img_lbl,
            "bar": bar,
            "photo": self.default_img,
            "done": None
        })

    def start_loading(self):
        # Start a thread to load thumbnails
        self.load_id += 1
        threading.Thread(target=self._load_thumbnails_thread, args=(self.load_id, list(self.items)), daemon=True).start()

    def _load_thumbnails_thread(self, load_id, items):
        for idx, path in enumerate(items):
            if load_id != self.load_id:
                return # cancelled
            try:
                bgr = cv2.imread(str(path))
                if bgr is not None:
                    # Resize to fit 68x48 card thumbnail
                    small = cv2.resize(bgr, (68, 48), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    # Pass pil_img to main thread. Do NOT instantiate ImageTk.PhotoImage in background thread!
                    if load_id == self.load_id:
                        self.after(0, lambda i=idx, pil=pil_img: self._update_thumbnail(i, pil, load_id))
            except Exception:
                pass

    def _update_thumbnail(self, idx, pil_img, load_id):
        if load_id != self.load_id or idx >= len(self.cards):
            return
        photo = ImageTk.PhotoImage(pil_img)
        self.cards[idx]["photo"] = photo
        self.cards[idx]["img_lbl"].configure(image=photo)

    def _on_card_click(self, idx, event):
        is_ctrl = (event.state & 0x0004) or (event.state & 0x0080)
        is_shift = (event.state & 0x0001)

        if is_ctrl:
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
            else:
                self.selected_indices.add(idx)
        elif is_shift and self.selected_indices:
            start = min(self.selected_indices)
            end = idx
            self.selected_indices.clear()
            for i in range(min(start, end), max(start, end) + 1):
                self.selected_indices.add(i)
        else:
            self.selected_indices = {idx}

        self._update_card_styles()
        self.on_select_cb()

    def _update_card_styles(self):
        # Update only cards whose selection status changed to avoid massive lag
        changed_indices = self.selected_indices ^ self._last_selected_indices
        if not self._last_selected_indices:
            changed_indices = set(range(len(self.cards)))
            
        for i in changed_indices:
            if i < len(self.cards):
                card = self.cards[i]["frame"]
                if i in self.selected_indices:
                    card.configure(border_color=ACCENT, fg_color="#1F3A5F")
                else:
                    card.configure(border_color="#21262D", fg_color=BG_PANEL)
        
        self._last_selected_indices = set(self.selected_indices)

    def selection_set(self, first, last=None):
        if last is None:
            self.selected_indices = {first}
        else:
            if last == tk.END:
                last = len(self.items) - 1
            self.selected_indices.clear()
            for i in range(first, last + 1):
                self.selected_indices.add(i)
        self._update_card_styles()

    def selection_clear(self, first, last=None):
        self.selected_indices.clear()
        self._update_card_styles()

    def curselection(self):
        return tuple(sorted(self.selected_indices))

    def see(self, idx):
        if not self.cards:
            return
        total_rows = (len(self.cards) + 2) // 3
        if total_rows <= 1:
            return
        row = idx // 3
        fraction = row / total_rows
        try:
            self._parent_canvas.yview_moveto(max(0.0, fraction - 0.1))
        except Exception:
            pass

    def update_status(self, idx, is_done):
        if idx < len(self.cards):
            card = self.cards[idx]
            if card.get("done") != is_done:
                card["done"] = is_done
                card["bar"].configure(fg_color=SUCCESS if is_done else TEXT_MUT)

    def itemconfig(self, idx, fg=None):
        self.update_status(idx, fg == SUCCESS)

# ──────────────────────────── APP ────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Auto_Annotator  |  Windows  |  YOLOv8 Offline")
        self.geometry("1540x940"); self.minsize(1200,720)
        self.configure(fg_color=BG_DARK)

        self.file_list  : list[Path] = []
        self.file_index = 0
        self.cap        = None
        self.all_anns   : dict[str, list[Annotation]] = {}
        self.orig       = None
        self.scale      = 1.0
        self.offset     = (0,0)
        self._model_ok  = False
        self.playing    = False

        self.show_conf_v  = tk.BooleanVar(value=True)
        self.show_man_v   = tk.BooleanVar(value=True)
        self.seg_v        = tk.BooleanVar(value=False)
        self.use_custom_v = tk.BooleanVar(value=False)
        self.conf_v       = tk.DoubleVar(value=0.15)   # lower default = more detections
        self.auto_save_on_exit = tk.BooleanVar(value=True)
        self.has_unsaved_changes = False

        # draw — always active when image loaded, no toggle button
        self._ds    = None   # draw start canvas coords
        self._dr_id = None   # temp rubber-band rect id
        self._seek_after_id = None  # debounce handle for seek slider
        self._drag_mode = None
        self._drag_ann_idx = None
        self._drag_start_mouse = None
        self._drag_start_bbox = None

        self.user_cls  : list[str] = []
        self.sel_cls_v = tk.StringVar(value="")
        self._train_path = None
        self._det_sel  : tuple = ()   # last known detection selection indices

        self._build_ui()
        self.bind("<Left>", self._on_left_key)
        self.bind("<Right>", self._on_right_key)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._load_persistent_config()
        self.after(300, lambda: threading.Thread(
            target=lambda: YOLO.load_base(self.model_v.get(), self._model_cb),
            daemon=True).start())

    # ═══════════════════════════ HELPERS ═════════════════════════════════════
    def _sep(self, p):
        ctk.CTkFrame(p, height=1, fg_color="#21262D").pack(fill="x", padx=10, pady=5)

    def _hdr(self, p, t):
        ctk.CTkLabel(p, text=t, font=ctk.CTkFont(size=11,weight="bold"),
                     text_color=TEXT_MUT).pack(anchor="w", padx=14, pady=(5,2))

    def _btn(self, p, txt, cmd, fg=BG_CARD, hov="#21262D", tc=TEXT_PRI, bold=False, h=32):
        return ctk.CTkButton(p, text=txt, fg_color=fg, hover_color=hov, text_color=tc,
                             font=ctk.CTkFont(size=12, weight="bold" if bold else "normal"),
                             corner_radius=8, height=h, command=cmd)

    # ═══════════════════════════ SIDEBAR ═════════════════════════════════════
    def _build_sidebar(self):
        sb = ctk.CTkScrollableFrame(self, width=282, fg_color=BG_DARK, corner_radius=0,
                                    scrollbar_button_color=BG_CARD,
                                    scrollbar_button_hover_color="#21262D")
        sb.grid(row=0, column=0, sticky="nsew")

        # logo
        lf = ctk.CTkFrame(sb, fg_color="transparent")
        lf.pack(padx=14, pady=(16,3), anchor="w")
        ctk.CTkLabel(lf, text="⬡ AUTO",
                     font=ctk.CTkFont(family="Courier",size=20,weight="bold"),
                     text_color=ACCENT).pack(side="left")
        ctk.CTkLabel(lf, text="ANNOTATE",
                     font=ctk.CTkFont(family="Courier",size=20,weight="bold"),
                     text_color=TEXT_PRI).pack(side="left")
        ctk.CTkLabel(sb, text="Auto_Annotator · Smart Train · Offline · Windows",
                     font=ctk.CTkFont(size=10), text_color=TEXT_MUT
                     ).pack(anchor="w", padx=14, pady=(0,6))
        self._sep(sb)

        # ── SOURCE ──
        self._hdr(sb, "📂  SOURCE")
        self._btn(sb,"🖼  Open Image(s)",        self.open_images).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"➕  Add Image(s)",         self.add_images).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"📁  Open Folder",           self.open_folder).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"🎬  Open Video (Choose FPS)", self.open_video).pack(fill="x",padx=10,pady=2)
        self._sep(sb)

        # ── FILE LIST ──
        self._hdr(sb, "🗂  FILE LIST  (🟢=done  ·  grey=pending)")
        self.file_lb = IconList(sb, self._on_file_click, height=250, fg_color=BG_PANEL, corner_radius=6)
        self.file_lb.pack(fill="x", padx=10, pady=2)

        # Prevent nested scrolling conflict between sidebar (sb) and file_lb
        orig_sb_check = sb.check_if_master_is_canvas
        def custom_sb_check(widget):
            try:
                if widget == self.file_lb or widget == self.file_lb._parent_canvas or self.file_lb.check_if_master_is_canvas(widget):
                    return False
            except Exception:
                pass
            return orig_sb_check(widget)
        sb.check_if_master_is_canvas = custom_sb_check

        # nav row
        nav = ctk.CTkFrame(sb, fg_color="transparent"); nav.pack(fill="x",padx=10,pady=2)
        self._btn(nav,"◀ Prev",self.prev_img,h=26).pack(side="left",padx=(0,3))
        self._btn(nav,"Next ▶",self.next_img,h=26).pack(side="left")
        self.counter_lbl = ctk.CTkLabel(nav,text="0 / 0",
                                        font=ctk.CTkFont(size=10),text_color=TEXT_MUT)
        self.counter_lbl.pack(side="right")

        # select-all / delete row
        del_row = ctk.CTkFrame(sb, fg_color="transparent"); del_row.pack(fill="x",padx=10,pady=2)
        ctk.CTkButton(del_row,text="☑ Select All",height=26,
                      fg_color=BG_CARD,hover_color="#21262D",text_color=TEXT_MUT,
                      font=ctk.CTkFont(size=11),corner_radius=6,
                      command=self.select_all_files
                      ).pack(side="left",fill="x",expand=True,padx=(0,3))
        ctk.CTkButton(del_row,text="🗑 Delete Selected",height=26,
                      fg_color="#2D1010",hover_color="#3D1515",text_color=DANGER,
                      font=ctk.CTkFont(size=11),corner_radius=6,
                      command=self.delete_selected_files
                      ).pack(side="left",fill="x",expand=True)
        self._sep(sb)

        # ── MODEL ──
        self._hdr(sb, "🤖  BASE MODEL")
        self.model_v = ctk.StringVar(value="yolov8n.pt")
        ctk.CTkOptionMenu(sb,
            values=["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt",
                    "yolov8n-seg.pt","yolov8s-seg.pt"],
            variable=self.model_v, fg_color=BG_CARD, button_color="#21262D",
            button_hover_color=ACCENT, text_color=TEXT_PRI, font=ctk.CTkFont(size=12),
            command=self._change_model).pack(fill="x",padx=10,pady=3)
        self.model_lbl = ctk.CTkLabel(sb,text="⏳ Loading…",
                                      font=ctk.CTkFont(size=10),text_color=WARNING)
        self.model_lbl.pack(anchor="w",padx=14)
        self._sep(sb)

        # ── SETTINGS ──
        self._hdr(sb, "⚙️  SETTINGS")
        cf = ctk.CTkFrame(sb, fg_color="transparent"); cf.pack(fill="x",padx=10,pady=1)
        self.conf_lbl = ctk.CTkLabel(cf,text="Confidence  0.15",
                                     font=ctk.CTkFont(size=11),text_color=TEXT_PRI)
        self.conf_lbl.pack(anchor="w")
        ctk.CTkSlider(cf,from_=0.05,to=0.95,variable=self.conf_v,
                      progress_color=ACCENT,button_color=ACCENT,
                      command=lambda v:self.conf_lbl.configure(
                          text=f"Confidence  {float(v):.2f}")
                      ).pack(fill="x",pady=2)

        for txt,var,col,redraw in [
            ("Show confidence",       self.show_conf_v, ACCENT,     True),
            ("Highlight manual [gold]",self.show_man_v, MANUAL_COL, True),
            ("Auto-save on close",    self.auto_save_on_exit, SUCCESS, False),
        ]:
            ctk.CTkSwitch(sb,text=txt,variable=var,progress_color=col,
                          font=ctk.CTkFont(size=11),text_color=TEXT_PRI,
                          command=self._redraw if redraw else None
                          ).pack(anchor="w",padx=14,pady=2)
        self._btn(sb, "💾  Save Progress", self.save_progress_manually,
                  fg="#1A2A1A", hov="#223022", tc=SUCCESS, bold=True
                  ).pack(fill="x", padx=10, pady=5)
        self._sep(sb)

        # ── MANUAL ANNOTATION ──
        # No draw toggle — canvas draw is always active
        self._hdr(sb, "✏️  MANUAL ANNOTATION  (drag on image)")

        # class hint label
        ctk.CTkLabel(sb, text="Select class, then drag on image to draw box:",
                     font=ctk.CTkFont(size=10), text_color=TEXT_MUT,
                     wraplength=240, justify="left"
                     ).pack(anchor="w", padx=14, pady=(0,3))

        # class dropdown
        cls_row = ctk.CTkFrame(sb, fg_color="transparent"); cls_row.pack(fill="x",padx=10,pady=2)
        ctk.CTkLabel(cls_row,text="Class:",font=ctk.CTkFont(size=11),
                     text_color=TEXT_MUT).pack(side="left",padx=(0,4))
        self.cls_menu = ctk.CTkOptionMenu(cls_row, values=["— add class first —"],
                                          variable=self.sel_cls_v,
                                          fg_color=BG_CARD, button_color="#21262D",
                                          button_hover_color=ACCENT3, text_color=TEXT_PRI,
                                          font=ctk.CTkFont(size=11), width=148)
        self.cls_menu.pack(side="left",fill="x",expand=True)

        # add / rename / remove class buttons
        cls_btns = ctk.CTkFrame(sb, fg_color="transparent"); cls_btns.pack(fill="x",padx=10,pady=2)
        for txt,cmd,col in [
            ("＋ Add",   self.add_cls,    ACCENT3),
            ("✏ Rename", self.rename_cls, ACCENT3),
            ("🗑 Remove", self.remove_cls, DANGER),
        ]:
            ctk.CTkButton(cls_btns,text=txt,height=28,
                          fg_color="#1A1A2E",hover_color="#252540",text_color=col,
                          font=ctk.CTkFont(size=11),corner_radius=6,
                          command=cmd).pack(side="left",fill="x",expand=True,padx=1)

        self._btn(sb,"📄  Upload Label .txt File",self.upload_current_label,
                  fg="#1A1A2E",hov="#252540",tc=SUCCESS,h=28
                  ).pack(fill="x",padx=10,pady=2)
        self._sep(sb)

        # ── SMART TRAIN & PROPAGATE ──
        self._hdr(sb, "🧠  SMART TRAIN & PROPAGATE")
        info = ctk.CTkFrame(sb, fg_color="#141410", corner_radius=6)
        info.pack(fill="x",padx=10,pady=4)
        ctk.CTkLabel(info,
            text="1. Add class names  (＋ Add)\n"
                 "2. Drag boxes on 10–15 images\n"
                 "3. Click  🏋️ Train\n"
                 "4. Click  🚀 Propagate\n"
                 "   → remaining images auto-labelled",
            font=ctk.CTkFont(size=10),text_color=TEXT_MUT,justify="left"
        ).pack(padx=8,pady=6,anchor="w")

        ep = ctk.CTkFrame(sb,fg_color="transparent"); ep.pack(fill="x",padx=10,pady=2)
        ctk.CTkLabel(ep,text="Epochs:",font=ctk.CTkFont(size=11),
                     text_color=TEXT_MUT).pack(side="left")
        self.epoch_v = tk.IntVar(value=30)
        ctk.CTkEntry(ep,textvariable=self.epoch_v,width=52,
                     fg_color=BG_CARD,border_color="#21262D",
                     text_color=TEXT_PRI,font=ctk.CTkFont(size=11)
                     ).pack(side="left",padx=6)
        ctk.CTkLabel(ep,text="(20–100)",font=ctk.CTkFont(size=9),
                     text_color=TEXT_MUT).pack(side="left")

        self._btn(sb,"🏋️  Train on My Annotations",self.train_model,
                  fg="#2A1A0A",hov="#3A2A10",tc=TRAIN_COL,bold=True
                  ).pack(fill="x",padx=10,pady=2)
        self.train_lbl = ctk.CTkLabel(sb,text="No custom model yet",
                                      font=ctk.CTkFont(size=10),text_color=TEXT_MUT)
        self.train_lbl.pack(anchor="w",padx=14,pady=1)

        self._btn(sb,"🚀  Propagate to Remaining Images",self.propagate,
                  fg="#1A2A0A",hov="#253510",tc=SUCCESS,bold=True
                  ).pack(fill="x",padx=10,pady=2)
        self._sep(sb)

        # ── CLEAR ──
        self._hdr(sb, "🗑  CLEAR")
        self._btn(sb,"🗑  Clear This Image",self.clear_current,
                  fg=BG_CARD,hov="#2D1B1B",tc=DANGER).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"🗑  Clear Auto Annotations",self.clear_auto,
                  fg=BG_CARD,hov="#2D1B1B",tc=DANGER).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"🗑🗑  Clear ALL Annotations",self.clear_all,
                  fg=BG_CARD,hov="#2D1B1B",tc=DANGER).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"🔥  Delete ALL Data",self.delete_all_data,
                  fg="#2D1010",hov="#3D1515",tc=DANGER,bold=True
                  ).pack(fill="x",padx=10,pady=2)
        self._sep(sb)

        # ── EXPORT / IMPORT ──
        self._hdr(sb, "📦  EXPORT  /  IMPORT")
        self._btn(sb,"📦  Export YOLO Dataset",
                  self.export_yolo,tc=SUCCESS).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"📂  Import YOLO Dataset (As Manual)",
                  self.import_yolo_dataset_manual,fg="#1A2A1A",hov="#223022",tc=SUCCESS
                  ).pack(fill="x",padx=10,pady=2)
        self._btn(sb,"📂  Import YOLO Dataset (As AI)",
                  self.import_yolo_dataset_ai,fg="#1A2A1A",hov="#223022",tc=SUCCESS
                  ).pack(fill="x",padx=10,pady=2)
        ctk.CTkFrame(sb,height=20,fg_color="transparent").pack()

    # ═══════════════════════════ CENTER ══════════════════════════════════════
    def _build_center(self):
        main = ctk.CTkFrame(self,fg_color=BG_DARK,corner_radius=0)
        main.grid(row=0,column=1,sticky="nsew")
        main.grid_rowconfigure(1,weight=1); main.grid_columnconfigure(0,weight=1)

        top = ctk.CTkFrame(main,fg_color=BG_CARD,height=44,corner_radius=0)
        top.grid(row=0,column=0,sticky="ew"); top.grid_propagate(False)
        top.grid_columnconfigure(1,weight=1)
        self.file_lbl = ctk.CTkLabel(top,text="No file — open images or a folder",
                                     font=ctk.CTkFont(family="Courier",size=11),
                                     text_color=TEXT_MUT)
        self.file_lbl.grid(row=0,column=0,padx=14,sticky="w")
        self.stats_lbl = ctk.CTkLabel(top,text="",
                                      font=ctk.CTkFont(family="Courier",size=11),
                                      text_color=ACCENT)
        self.stats_lbl.grid(row=0,column=1,padx=14,sticky="e")

        cf = ctk.CTkFrame(main,fg_color="#080C10",corner_radius=0)
        cf.grid(row=1,column=0,sticky="nsew")
        cf.grid_rowconfigure(0,weight=1); cf.grid_columnconfigure(0,weight=1)
        self.canvas = tk.Canvas(cf,bg="#080C10",highlightthickness=0,cursor="crosshair")
        self.canvas.grid(row=0,column=0,sticky="nsew")
        self.canvas.bind("<Configure>",    lambda e: self._redraw())
        self.canvas.bind("<ButtonPress-1>",  self._cd_press)
        self.canvas.bind("<B1-Motion>",      self._cd_drag)
        self.canvas.bind("<ButtonRelease-1>",self._cd_release)

        self.prog = ctk.CTkProgressBar(main,height=4,
                                       progress_color=ACCENT,fg_color="#21262D")
        self.prog.grid(row=2,column=0,sticky="ew"); self.prog.set(0)

        self.status_v = tk.StringVar(value="Ready — open images or a folder.")
        sbar = ctk.CTkFrame(main,fg_color=BG_CARD,height=26,corner_radius=0)
        sbar.grid(row=3,column=0,sticky="ew"); sbar.grid_propagate(False)
        ctk.CTkLabel(sbar,textvariable=self.status_v,
                     font=ctk.CTkFont(family="Courier",size=10),
                     text_color=TEXT_MUT).pack(side="left",padx=10)

        # nav bar
        vc = ctk.CTkFrame(main,fg_color=BG_CARD,height=50,corner_radius=0)
        vc.grid(row=4,column=0,sticky="ew"); vc.grid_propagate(False)
        vc.grid_columnconfigure(4,weight=1)
        for col,txt,cmd,fc,tc in [
            (0,"⏮",self.prev_frame,BG_DARK,TEXT_PRI),
            (1,"▶",self.toggle_play,ACCENT,"#000"),
            (2,"⏭",self.next_frame,BG_DARK,TEXT_PRI),
        ]:
            btn = ctk.CTkButton(vc,text=txt,width=36,height=30,
                                fg_color=fc,hover_color="#21262D",text_color=tc,
                                font=ctk.CTkFont(size=14),command=cmd)
            btn.grid(row=0,column=col,padx=(8 if col==0 else 2,2),pady=9)
            if txt=="▶": self.play_btn=btn
        self.seek_v = tk.DoubleVar(value=0)
        self._seek_after_id = None   # debounce handle
        self.seek_sl = ctk.CTkSlider(vc,from_=0,to=100,variable=self.seek_v,
                                     progress_color=ACCENT,button_color=ACCENT,
                                     command=self._seek_debounced)
        self.seek_sl.grid(row=0,column=4,padx=8,pady=9,sticky="ew")
        self.frm_lbl = ctk.CTkLabel(vc,text="0/0",
                                    font=ctk.CTkFont(family="Courier",size=10),
                                    text_color=TEXT_MUT,width=66)
        self.frm_lbl.grid(row=0,column=5,padx=6)

        # detection list at bottom
        af = ctk.CTkFrame(main,fg_color=BG_CARD,height=140,corner_radius=0)
        af.grid(row=5,column=0,sticky="ew"); af.grid_propagate(False)
        af.grid_columnconfigure(0,weight=1); af.grid_rowconfigure(1,weight=1)
        ctk.CTkLabel(af,text="DETECTIONS   [M]=Manual   [A]=AI — click to select",
                     font=ctk.CTkFont(size=10,weight="bold"),text_color=TEXT_MUT
                     ).grid(row=0,column=0,padx=10,pady=(5,1),sticky="w")
        self.det_lb = tk.Listbox(af,bg=BG_CARD,fg=TEXT_PRI,
                                  selectbackground="#1F3A5F",selectforeground=ACCENT,
                                  font=("Courier",10),relief="flat",
                                  highlightthickness=0,borderwidth=0,
                                  activestyle="none",selectmode=tk.EXTENDED)
        self.det_lb.grid(row=1,column=0,padx=4,pady=2,sticky="nsew")
        self.det_lb.bind("<<ListboxSelect>>",self._on_det_select)

        br = ctk.CTkFrame(af,fg_color="transparent")
        br.grid(row=2,column=0,padx=6,pady=4,sticky="e")
        # Edit Class button — uses simple askstring dialog, fully reliable
        ctk.CTkButton(br,text="✏  Edit Class",height=28,width=100,
                      fg_color="#1A1A2E",hover_color="#252540",text_color=ACCENT3,
                      font=ctk.CTkFont(size=11),corner_radius=6,
                      command=self._edit_det_class).pack(side="left",padx=3)
        ctk.CTkButton(br,text="🗑  Remove",height=28,width=90,
                      fg_color="#2D1B1B",hover_color="#3D2020",text_color=DANGER,
                      font=ctk.CTkFont(size=11),corner_radius=6,
                      command=self._remove_dets).pack(side="left",padx=3)

    # ═══════════════════════════ BUILD ═══════════════════════════════════════
    def _build_ui(self):
        self.grid_columnconfigure(1,weight=1)
        self.grid_rowconfigure(0,weight=1)
        self._build_sidebar()
        self._build_center()

    # ═══════════════════════════ MODEL ═══════════════════════════════════════
    def _model_cb(self, msg):
        col = SUCCESS if "✓" in msg else (DANGER if "Error" in msg else WARNING)
        self.after(0, lambda: (
            self.model_lbl.configure(text=msg, text_color=col),
            setattr(self,"_model_ok","✓" in msg)))

    def _change_model(self, _=None):
        self._model_ok = False
        self._model_cb("⏳ Loading…")
        threading.Thread(
            target=lambda: YOLO.load_base(self.model_v.get(), self._model_cb),
            daemon=True).start()

    # ═══════════════════════════ CLASSES ═════════════════════════════════════
    def _refresh_cls_menu(self):
        vals = self.user_cls if self.user_cls else ["— add class first —"]
        self.cls_menu.configure(values=vals)
        if self.sel_cls_v.get() not in self.user_cls:
            self.sel_cls_v.set(vals[0])

    def add_cls(self):
        n = simpledialog.askstring("Add Class","Class name:",parent=self)
        if not n or not n.strip(): return
        n = n.strip()
        if n in self.user_cls: self.sel_cls_v.set(n); return
        self.user_cls.append(n)
        self._refresh_cls_menu()
        self.sel_cls_v.set(n)
        self.status_v.set(f'Class "{n}" added — drag on image to draw a box.')
        self._save_persistent_config()

    def rename_cls(self):
        cur = self.sel_cls_v.get()
        if cur not in self.user_cls:
            messagebox.showinfo("","Pick a class from the dropdown first."); return
        new = simpledialog.askstring("Rename",f'Rename "{cur}" to:',parent=self)
        if not new or not new.strip(): return
        new = new.strip()
        self.user_cls[self.user_cls.index(cur)] = new
        for anns in self.all_anns.values():
            for a in anns:
                if a.cls_name == cur: a.cls_name = new
        self._refresh_cls_menu()
        self.sel_cls_v.set(new)
        self._redraw()
        self._save_persistent_config()

    def remove_cls(self):
        cur = self.sel_cls_v.get()
        if cur not in self.user_cls:
            messagebox.showinfo("","Pick a class from the dropdown first."); return
        if not messagebox.askyesno("Remove",f'Remove "{cur}" from class list?'): return
        self.user_cls.remove(cur)
        self._refresh_cls_menu()
        self._save_persistent_config()

    def add_images(self):
        paths = filedialog.askopenfilenames(title="Select Images to Add",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.JPG *.PNG"),
                       ("All","*.*")])
        if paths:
            self._set_file_list([Path(p) for p in paths], keep_existing=True)
            self.has_unsaved_changes = True

    def clear_auto(self):
        tot_auto = sum(1 for v in self.all_anns.values() for a in v if not a.manual)
        if tot_auto == 0:
            messagebox.showinfo("", "No automatic annotations found.")
            return
        if not messagebox.askyesno("Clear Auto Annotations",
            f"Delete ALL {tot_auto} automatic annotations across all images?\n(Manual annotations will be preserved)",
            icon="warning"):
            return
        for k in list(self.all_anns.keys()):
            self.all_anns[k] = [a for a in self.all_anns[k] if a.manual]
            if not self.all_anns[k]:
                self.all_anns.pop(k, None)
        self._upd_stats()
        self._redraw()
        self._color_file_list()
        self.has_unsaved_changes = True
        self.status_v.set(f"Removed {tot_auto} automatic annotations across all images.")

    def _on_left_key(self, event):
        focused = self.focus_get()
        if isinstance(focused, (ctk.CTkEntry, tk.Entry)):
            return
        self.prev_img()

    def _on_right_key(self, event):
        focused = self.focus_get()
        if isinstance(focused, (ctk.CTkEntry, tk.Entry)):
            return
        self.next_img()

    def _on_canvas_double_click(self, ev):
        if self.orig is None: return
        ann_idx = self._get_ann_under_mouse(ev.x, ev.y)
        if ann_idx is not None:
            self.det_lb.selection_clear(0, tk.END)
            self.det_lb.selection_set(ann_idx)
            self._det_sel = (ann_idx,)
            anns = self._cur_anns()
            for i, a in enumerate(anns):
                a.selected = (i == ann_idx)
            self._redraw()
            self._edit_det_class()

    def _save_persistent_config(self):
        try:
            app_dir = Path(__file__).resolve().parent.parent
            config_path = app_dir / "config.json"
            custom_models_dir = app_dir / "custom_models"
            custom_models_dir.mkdir(exist_ok=True)
            saved_model_path = ""
            if self._train_path and Path(self._train_path).exists():
                dest = custom_models_dir / "best_custom.pt"
                shutil.copy2(self._train_path, dest)
                saved_model_path = str(dest)
            config_data = {
                "user_cls": self.user_cls,
                "custom_model_path": saved_model_path,
                "conf_v": self.conf_v.get(),
                "show_conf_v": self.show_conf_v.get(),
                "show_man_v": self.show_man_v.get(),
                "seg_v": self.seg_v.get(),
                "use_custom_v": self.use_custom_v.get(),
                "auto_save_on_exit": self.auto_save_on_exit.get(),
                "file_list": [str(p) for p in self.file_list],
                "file_index": self.file_index,
                "all_anns": {k: [a.to_dict() for a in v] for k, v in self.all_anns.items()}
            }
            config_path.write_text(json.dumps(config_data, indent=2))
            self.has_unsaved_changes = False
        except Exception as e:
            print(f"Error saving config: {e}")

    def _load_persistent_config(self):
        try:
            app_dir = Path(__file__).resolve().parent.parent
            config_path = app_dir / "config.json"
            if config_path.exists():
                config_data = json.loads(config_path.read_text())
                self.user_cls = config_data.get("user_cls", [])
                if self.user_cls:
                    self._refresh_cls_menu()
                self.conf_v.set(config_data.get("conf_v", 0.15))
                self.show_conf_v.set(config_data.get("show_conf_v", True))
                self.show_man_v.set(config_data.get("show_man_v", True))
                self.seg_v.set(config_data.get("seg_v", False))
                self.use_custom_v.set(config_data.get("use_custom_v", False))
                self.auto_save_on_exit.set(config_data.get("auto_save_on_exit", True))
                
                custom_model_path = config_data.get("custom_model_path", "")
                if custom_model_path and Path(custom_model_path).exists():
                    self._train_path = custom_model_path
                    from ultralytics import YOLO as UY
                    YOLO.custom = UY(custom_model_path)
                    self.train_lbl.configure(text="✅ Custom model loaded", text_color=SUCCESS)
                
                # Restore annotations
                raw_anns = config_data.get("all_anns", {})
                self.all_anns = {}
                for k, v in raw_anns.items():
                    if Path(k).exists():
                        anns_list = []
                        for a_dict in v:
                            anns_list.append(Annotation(
                                cls_id=a_dict.get("class_id", 0),
                                cls_name=a_dict.get("class_name", ""),
                                bbox=a_dict.get("bbox", [0, 0, 0, 0]),
                                confidence=a_dict.get("confidence", 1.0),
                                polygon=a_dict.get("polygon"),
                                manual=a_dict.get("manual", False)
                            ))
                        self.all_anns[k] = anns_list

                # Restore files and file index selection
                saved_files = [Path(p) for p in config_data.get("file_list", []) if Path(p).exists()]
                if saved_files:
                    self._set_file_list(saved_files, keep_existing=False)
                    saved_idx = config_data.get("file_index", 0)
                    idx = max(0, min(saved_idx, len(self.file_list)-1))
                    self.file_lb.selection_clear(0, tk.END)
                    self.file_lb.selection_set(idx)
                    self.file_lb.see(idx)
                    self._load_by_index(idx)
                    self._color_file_list()
        except Exception as e:
            print(f"Error loading config: {e}")

    def _cls_id(self, name):
        return self.user_cls.index(name) if name in self.user_cls else 0

    def save_progress_manually(self):
        self._save_persistent_config()
        self.status_v.set("💾 Progress saved successfully.")
        messagebox.showinfo("Saved", "Your progress (annotations, file list, and current image) has been saved successfully.", parent=self)

    def _on_closing(self):
        if self.auto_save_on_exit.get():
            self._save_persistent_config()
            self.destroy()
        else:
            if self.has_unsaved_changes:
                ans = messagebox.askyesnocancel(
                    "Save Progress",
                    "You have unsaved changes. Would you like to save your progress before closing?",
                    parent=self
                )
                if ans is True:  # Yes
                    self._save_persistent_config()
                    self.destroy()
                elif ans is False:  # No
                    self.destroy()
                # If None (cancel), do nothing and return
            else:
                self.destroy()

    # ═══════════════════════════ FILES ═══════════════════════════════════════
    def open_images(self):
        paths = filedialog.askopenfilenames(title="Select Images",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.JPG *.PNG"),
                       ("All","*.*")])
        if paths:
            self._set_file_list([Path(p) for p in paths])
            self.has_unsaved_changes = True

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder: return
        exts = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
        imgs = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
        if not imgs: messagebox.showinfo("Empty","No images found."); return
        self._set_file_list(imgs)
        self.has_unsaved_changes = True

    def open_video(self):
        path = filedialog.askopenfilename(title="Select Video",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv *.webm"),("All","*.*")])
        if not path: return
        
        fps_choice = simpledialog.askinteger(
            "Extract Frames",
            "Enter extraction rate (Frames Per Second):\n\n"
            "  1  = 1 frame per second (Default)\n"
            "  5  = 5 frames per second\n"
            "  10 = 10 frames per second\n"
            "  25 = Full frame rate",
            initialvalue=1, minvalue=1, maxvalue=100, parent=self)
        if not fps_choice: return
        
        out_dir = filedialog.askdirectory(title="Folder to save extracted frames")
        if not out_dir: return
        out_dir = Path(out_dir)/(Path(path).stem+"_frames")
        out_dir.mkdir(parents=True,exist_ok=True)
        self.status_v.set(f"Extracting {fps_choice} frame/sec…"); self.prog.set(0)
        def _ex():
            cap = cv2.VideoCapture(path)
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(native_fps / fps_choice))
            saved=[]; i=0
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                ret,frame=cap.read()
                if not ret: break
                fn = out_dir/f"frame_{i:06d}.jpg"
                cv2.imwrite(str(fn),frame); saved.append(fn)
                self.after(0,lambda p=i/max(total,1):self.prog.set(p))
                i+=step
            cap.release()
            self.after(0,lambda:(
                 self.prog.set(1.0),
                 self.status_v.set(f"Extracted {len(saved)} frames → {out_dir}"),
                 self._set_file_list(saved, keep_existing=False),
                 setattr(self, "has_unsaved_changes", True)))
        threading.Thread(target=_ex,daemon=True).start()

    def upload_current_label(self):
        if self.orig is None:
            messagebox.showinfo("","Open an image first."); return
        path = filedialog.askopenfilename(title="Select YOLO .txt Label or Classes File",
            filetypes=[("Text File","*.txt"),("All","*.*")])
        if not path: return
        
        # Read lines first
        lines = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
            return
            
        if not lines:
            messagebox.showinfo("Empty", "Selected file is empty."); return
            
        # Determine if it's a YOLO coordinate file or a Class names list file
        is_yolo_labels = False
        first_line_parts = re.split(r'[,\s]+', lines[0])
        if len(first_line_parts) >= 5:
            try:
                # Try to parse the next 4 coordinates as floats
                list(map(float, first_line_parts[1:5]))
                is_yolo_labels = True
            except ValueError:
                pass
                
        if not is_yolo_labels:
            # Mode A: Load as CLASS NAMES!
            added_classes = []
            for line in lines:
                name = line.strip()
                if name.startswith("#") or name.startswith("//"): continue
                if name and name not in self.user_cls:
                    self.user_cls.append(name)
                    added_classes.append(name)
                    
            if added_classes:
                self._refresh_cls_menu()
                self.sel_cls_v.set(added_classes[0])
                self._save_persistent_config()
                messagebox.showinfo("Classes Loaded", 
                    f"Successfully imported {len(added_classes)} class(es) into the class menu:\n\n" + 
                    "\n".join(f"  • {name}" for name in added_classes))
                self.status_v.set(f"Imported {len(added_classes)} classes from list.")
            else:
                messagebox.showinfo("Info", "All classes in this file are already in the class menu.")
            return

        # Mode B: Load as Bounding Box Annotations!
        h, w = self.orig.shape[:2]
        key = self._cur_key()
        anns = []
        
        # Try to find classes.txt in the same folder as the selected file
        classes_txt = Path(path).parent / "classes.txt"
        lbl_names = []
        if classes_txt.exists():
            try:
                with open(classes_txt, "r", encoding="utf-8", errors="ignore") as cf:
                    lbl_names = [l.strip() for l in cf if l.strip()]
                for name in lbl_names:
                    if name not in self.user_cls:
                        self.user_cls.append(name)
            except Exception:
                pass
                
        try:
            for line in lines:
                parts = re.split(r'[,\s]+', line)
                if len(parts) < 5: continue
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue # Skip invalid lines
                    
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                if lbl_names and cid < len(lbl_names):
                    cn = lbl_names[cid]
                elif cid < len(self.user_cls):
                    cn = self.user_cls[cid]
                else:
                    cn = f"class_{cid}"
                    if cn not in self.user_cls:
                        self.user_cls.append(cn)
                        
                anns.append(Annotation(cid, cn, [x1, y1, x2, y2], 1.0, manual=True))
                
            self._set_anns(anns)
            self._refresh_cls_menu()
            self._upd_stats()
            self._redraw()
            self._color_file_list()
            self._save_persistent_config()
            self.status_v.set(f"Uploaded {len(anns)} annotations from label file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse label file: {e}")

    def _set_file_list(self, paths, keep_existing=False):
        """Load paths into the file list.
        keep_existing=True  →  merge new paths with existing (no duplicates).
        keep_existing=False →  replace entirely (fresh load).
        """
        if self.cap: self.cap.release(); self.cap = None

        new_paths = list(paths)
        if keep_existing and self.file_list:
            existing_set = {str(p) for p in self.file_list}
            added = [p for p in new_paths if str(p) not in existing_set]
            merged = self.file_list + added
        else:
            merged = new_paths

        self.file_list = merged

        # Rebuild listbox cleanly (delete once, insert all)
        self.file_lb.clear()
        for p in merged:
            self.file_lb.add_item(p)
        self.file_lb.start_loading()

        if not merged: return

        # If keeping existing, stay on current image; otherwise go to first
        if keep_existing and self.orig is not None:
            idx = self.file_index
        else:
            idx = 0

        idx = max(0, min(idx, len(merged)-1))
        self.file_lb.selection_set(idx)
        self.file_lb.see(idx)
        self._load_by_index(idx)

    def _load_by_index(self, idx):
        if not self.file_list or not (0<=idx<len(self.file_list)): return
        self.file_index=idx
        path=self.file_list[idx]
        bgr=cv2.imread(str(path))
        if bgr is None: self.status_v.set(f"Cannot read: {path.name}"); return
        self.orig=bgr
        self._det_sel = ()   # clear selection when switching image
        self.file_lbl.configure(text=f"[{idx+1}/{len(self.file_list)}]  {path.name}")
        self.counter_lbl.configure(text=f"{idx+1} / {len(self.file_list)}")
        self._upd_stats(); self._redraw()
        self.status_v.set(f"{path.name}  {bgr.shape[1]}×{bgr.shape[0]}"
                          "  |  drag on image to annotate")

    def _color_file_list(self):
        for i,p in enumerate(self.file_list):
            self.file_lb.itemconfig(i,
                fg=SUCCESS if self.all_anns.get(str(p)) else TEXT_MUT)



    # ═══════════════════════════ KEY HELPERS ═════════════════════════════════
    def _cur_key(self):
        if self.file_list and 0<=self.file_index<len(self.file_list):
            return str(self.file_list[self.file_index])
        return "__none__"

    def _cur_anns(self):   return self.all_anns.get(self._cur_key(),[])
    def _set_anns(self,a): self.all_anns[self._cur_key()]=a

    # ═══════════════════════════ TRAIN & PROPAGATE ════════════════════════════
    def train_model(self):
        annotated=[k for k,v in self.all_anns.items()
                   if v and any(str(p)==k for p in self.file_list)]
        if len(annotated)<2:
            messagebox.showwarning("","Annotate at least 2 images first."); return
        if not self.user_cls:
            messagebox.showwarning("","Add class names first (＋ Add)."); return
        if not self._model_ok:
            messagebox.showinfo("","Base model still loading."); return
        epochs=max(5,self.epoch_v.get())
        def _run():
            try:
                self.after(0,lambda:self.train_lbl.configure(
                    text="Preparing…",text_color=WARNING))
                tmp=Path(tempfile.mkdtemp(prefix="aann_"))
                (tmp/"images"/"train").mkdir(parents=True)
                (tmp/"labels"/"train").mkdir(parents=True)
                cmap={n:i for i,n in enumerate(self.user_cls)}
                written=0
                for key in annotated:
                    bgr=cv2.imread(key)
                    if bgr is None: continue
                    h,w=bgr.shape[:2]
                    stem=Path(key).stem
                    shutil.copy2(key,str(tmp/"images"/"train"/(stem+Path(key).suffix)))
                    lines=[]
                    for a in self.all_anns[key]:
                        cid=cmap.get(a.cls_name,0)
                        x1,y1,x2,y2=a.bbox
                        cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h
                        bw=(x2-x1)/w; bh=(y2-y1)/h
                        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    (tmp/"labels"/"train"/(stem+".txt")).write_text("\n".join(lines))
                    written+=1
                (tmp/"data.yaml").write_text(
                    f"path: {tmp}\ntrain: images/train\nval: images/train\n"
                    f"nc: {len(self.user_cls)}\nnames: {self.user_cls}\n")
                self.after(0,lambda:self.status_v.set(
                    f"🏋️ Training on {written} images × {epochs} epochs…"))
                best=YOLO.train(str(tmp/"data.yaml"),self.model_v.get(),epochs,
                                cb=lambda m, p: self.after(0, lambda msg=m, pct=p: (
                                    self.status_v.set(msg),
                                    self.prog.set(pct)
                                )))
                self._train_path=best
                self.after(0,lambda:(
                    self.train_lbl.configure(
                        text=f"✅ Custom model ready ({written} imgs)",text_color=SUCCESS),
                    self.use_custom_v.set(True),
                    self.status_v.set("✅ Training done! → Click 🚀 Propagate"),
                    self._save_persistent_config()))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"{type(e).__name__}: {e}"
                self.after(0,lambda:(
                    self.train_lbl.configure(text=f"Error: {error_msg}",text_color=DANGER),
                    self.status_v.set(f"Train error: {error_msg}")))
        threading.Thread(target=_run,daemon=True).start()

    def propagate(self):
        if not YOLO.custom:
            messagebox.showwarning("","Train first — 🏋️ Train on My Annotations."); return
        pending=[p for p in self.file_list if not self.all_anns.get(str(p))]
        if not pending:
            messagebox.showinfo("","All images already annotated!"); return
        if not messagebox.askyesno("Propagate",
            f"Auto-annotate {len(pending)} remaining images?\n\n"
            f"Classes: {', '.join(self.user_cls)}\nConf: {self.conf_v.get():.2f}"): return
        def _run():
            total=len(pending)
            for i,path in enumerate(pending):
                bgr=cv2.imread(str(path))
                if bgr is None: continue
                anns=YOLO.predict(bgr,conf=self.conf_v.get(),
                                  seg=False,use_custom=True)
                self.all_anns[str(path)]=anns
                self.after(0,lambda p=(i+1)/total,nm=path.name,idx=i:(
                    self.prog.set(p),
                    self.status_v.set(f"Propagating {idx+1}/{total}: {nm}")))
            self.after(0,lambda:(
                self.prog.set(1.0),
                self.status_v.set(f"✅ Propagated to {total} images!"),
                self._color_file_list(),
                self._upd_stats(),
                self._redraw(),
                setattr(self, "has_unsaved_changes", True)))
        threading.Thread(target=_run,daemon=True).start()

    def clear_current(self):
        self._set_anns([])
        self._upd_stats()
        self._redraw()
        self._color_file_list()
        self.has_unsaved_changes = True

    def clear_all(self):
        tot=sum(len(v) for v in self.all_anns.values())
        if tot==0: messagebox.showinfo("","Nothing to clear."); return
        if not messagebox.askyesno("Clear ALL",
            f"Delete ALL {tot} annotations across all images?",icon="warning"): return
        self.all_anns.clear()
        self._upd_stats(); self._redraw(); self._color_file_list()
        self.has_unsaved_changes = True
        self.status_v.set("All annotations cleared.")

    def delete_all_data(self):
        if not self.file_list and not self.user_cls and not YOLO.custom:
            messagebox.showinfo("", "No data to clear."); return
            
        if not messagebox.askyesno("Delete All Data",
            "Are you sure you want to delete ALL data in the project?\n\n"
            "This will:\n"
            "  • Remove all loaded images from the list\n"
            "  • Delete all annotations/labels\n"
            "  • Reset all custom classes\n"
            "  • Clear custom weights & configuration\n\n"
            "The raw files on your disk will NOT be deleted.",
            icon="warning"):
            return
            
        self.file_list = []
        self.file_index = 0
        self.all_anns.clear()
        self.orig = None
        self.canvas.delete("all")
        self.file_lbl.configure(text="No file — open images or a folder")
        
        self.user_cls = []
        self._refresh_cls_menu()
        
        self._train_path = None
        YOLO.custom = None
        self.train_lbl.configure(text="No custom model yet", text_color=TEXT_MUT)
        self.use_custom_v.set(False)
        
        self.file_lb.clear()
        
        try:
            app_dir = Path(__file__).resolve().parent.parent
            config_path = app_dir / "config.json"
            if config_path.exists():
                config_path.unlink()
            custom_models_dir = app_dir / "custom_models"
            if custom_models_dir.is_dir():
                shutil.rmtree(custom_models_dir)
        except Exception as e:
            print(f"Error removing config: {e}")
            
        self._upd_stats()
        self._redraw()
        self.has_unsaved_changes = False
        self.status_v.set("All project data successfully deleted.")

    # ═══════════════════════════ CANVAS DRAW ═════════════════════════════════
    # Draw is always active — no toggle needed
    def _get_handle_under_mouse(self, mx, my):
        sel = self._det_sel if self._det_sel else self.det_lb.curselection()
        if not sel: return None, None
        idx = sel[0]
        anns = self._cur_anns()
        if idx >= len(anns): return None, None
        ann = anns[idx]
        ox, oy = self.offset
        s = self.scale
        x1 = int(ann.bbox[0] * s + ox)
        y1 = int(ann.bbox[1] * s + oy)
        x2 = int(ann.bbox[2] * s + ox)
        y2 = int(ann.bbox[3] * s + oy)
        threshold = 10
        if abs(mx - x1) <= threshold and abs(my - y1) <= threshold:
            return idx, "top_left"
        if abs(mx - x2) <= threshold and abs(my - y1) <= threshold:
            return idx, "top_right"
        if abs(mx - x1) <= threshold and abs(my - y2) <= threshold:
            return idx, "bottom_left"
        if abs(mx - x2) <= threshold and abs(my - y2) <= threshold:
            return idx, "bottom_right"
        return None, None

    def _get_ann_under_mouse(self, mx, my):
        ox, oy = self.offset
        s = self.scale
        anns = self._cur_anns()
        for idx in range(len(anns) - 1, -1, -1):
            ann = anns[idx]
            x1 = int(ann.bbox[0] * s + ox)
            y1 = int(ann.bbox[1] * s + oy)
            x2 = int(ann.bbox[2] * s + ox)
            y2 = int(ann.bbox[3] * s + oy)
            if min(x1, x2) <= mx <= max(x1, x2) and min(y1, y2) <= my <= max(y1, y2):
                return idx
        return None

    def _get_label_under_mouse(self, mx, my):
        ox, oy = self.offset
        s = self.scale
        anns = self._cur_anns()
        show_conf = self.show_conf_v.get()
        for idx in range(len(anns) - 1, -1, -1):
            a = anns[idx]
            x1 = int(a.bbox[0] * s + ox)
            y1 = int(a.bbox[1] * s + oy)
            lbl = f"[M]{a.cls_name}" if a.manual else a.cls_name
            if show_conf and not a.manual: lbl += f" {a.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            ly = max(y1 - 2, th + 4)
            lx1 = x1
            ly1 = ly - th - 4
            lx2 = x1 + tw + 4
            ly2 = ly
            if lx1 <= mx <= lx2 and ly1 <= my <= ly2:
                return idx
        return None

    def _cd_press(self, ev):
        if self.orig is None: return
        
        # 1. Check if clicked on the label of any box to directly edit class
        label_idx = self._get_label_under_mouse(ev.x, ev.y)
        if label_idx is not None:
            self.det_lb.selection_clear(0, tk.END)
            self.det_lb.selection_set(label_idx)
            self._det_sel = (label_idx,)
            anns = self._cur_anns()
            for i, a in enumerate(anns):
                a.selected = (i == label_idx)
            self._redraw()
            self._show_class_dropdown(label_idx, ev.x_root, ev.y_root)
            return

        # 2. Check if clicked on a corner handle of the selected box
        ann_idx, handle = self._get_handle_under_mouse(ev.x, ev.y)
        if handle:
            self._drag_mode = handle
            self._drag_ann_idx = ann_idx
            self._drag_start_mouse = (ev.x, ev.y)
            self._drag_start_bbox = list(self._cur_anns()[ann_idx].bbox)
            self.status_v.set(f"Resizing {handle} of selected box...")
            return
            
        # 3. Check if clicked inside any box to select & move
        ann_idx = self._get_ann_under_mouse(ev.x, ev.y)
        if ann_idx is not None:
            self.det_lb.selection_clear(0, tk.END)
            self.det_lb.selection_set(ann_idx)
            self._det_sel = (ann_idx,)
            anns = self._cur_anns()
            for i, a in enumerate(anns):
                a.selected = (i == ann_idx)
            self._drag_mode = "move"
            self._drag_ann_idx = ann_idx
            self._drag_start_mouse = (ev.x, ev.y)
            self._drag_start_bbox = list(anns[ann_idx].bbox)
            self._redraw()
            self.status_v.set("Moving selected box...")
            return
            
        # 4. Default: Draw a new box
        self._drag_mode = "draw"
        self._ds = (ev.x, ev.y)

    def _cd_drag(self, ev):
        if self.orig is None: return
        
        if self._drag_mode == "draw":
            if self._ds is None: return
            if self._dr_id: self.canvas.delete(self._dr_id)
            x0, y0 = self._ds
            self._dr_id = self.canvas.create_rectangle(
                x0, y0, ev.x, ev.y, outline=MANUAL_COL, width=2, dash=(6, 3))
                
        elif self._drag_mode in ("move", "top_left", "top_right", "bottom_left", "bottom_right"):
            ann_idx = self._drag_ann_idx
            anns = self._cur_anns()
            if ann_idx >= len(anns): return
            ox, oy = self.offset
            s = self.scale
            if s <= 0: return
            
            dx = (ev.x - self._drag_start_mouse[0]) / s
            dy = (ev.y - self._drag_start_mouse[1]) / s
            
            x1, y1, x2, y2 = self._drag_start_bbox
            img_h, img_w = self.orig.shape[:2]
            
            if self._drag_mode == "move":
                nx1 = int(x1 + dx)
                ny1 = int(y1 + dy)
                nx2 = int(x2 + dx)
                ny2 = int(y2 + dy)
                box_w = x2 - x1
                box_h = y2 - y1
                nx1 = max(0, min(img_w - box_w, nx1))
                ny1 = max(0, min(img_h - box_h, ny1))
                nx2 = nx1 + box_w
                ny2 = ny1 + box_h
            else:
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                if "left" in self._drag_mode:
                    nx1 = max(0, min(x2 - 5, int(x1 + dx)))
                if "right" in self._drag_mode:
                    nx2 = max(x1 + 5, min(img_w, int(x2 + dx)))
                if "top" in self._drag_mode:
                    ny1 = max(0, min(y2 - 5, int(y1 + dy)))
                if "bottom" in self._drag_mode:
                    ny2 = max(y1 + 5, min(img_h, int(y2 + dy)))
            
            anns[ann_idx].bbox = [nx1, ny1, nx2, ny2]
            anns[ann_idx].manual = True
            self._redraw()

    def _cd_release(self, ev):
        if self._drag_mode == "draw":
            if self._ds is None: return
            if self._dr_id: self.canvas.delete(self._dr_id); self._dr_id = None
            x0, y0 = self._ds
            x1, y1 = ev.x, ev.y
            self._ds = None
            
            ox, oy = self.offset
            s = self.scale
            if s <= 0: return
            
            ix0 = int((min(x0, x1) - ox) / s)
            iy0 = int((min(y0, y1) - oy) / s)
            ix1 = int((max(x0, x1) - ox) / s)
            iy1 = int((max(y0, y1) - oy) / s)
            
            img_h, img_w = self.orig.shape[:2]
            ix0 = max(0, min(img_w, ix0))
            iy0 = max(0, min(img_h, iy0))
            ix1 = max(0, min(img_w, ix1))
            iy1 = max(0, min(img_h, iy1))
            
            if abs(ix1 - ix0) < 5 or abs(iy1 - iy0) < 5: return
            
            cn = self.sel_cls_v.get()
            if cn not in self.user_cls:
                messagebox.showwarning("No class",
                    "Add a class first using  ＋ Add  in the sidebar."); return
                    
            ann = Annotation(self._cls_id(cn), cn, [ix0, iy0, ix1, iy1], 1.0, manual=True)
            anns = list(self._cur_anns())
            anns.append(ann)
            self._set_anns(anns)
            
            new_idx = len(anns) - 1
            self.det_lb.selection_clear(0, tk.END)
            self.det_lb.selection_set(new_idx)
            self._det_sel = (new_idx,)
            for i, a in enumerate(anns):
                a.selected = (i == new_idx)
                
            self._upd_stats()
            self._redraw()
            self._color_file_list()
            self.has_unsaved_changes = True
            self.status_v.set(f"Box added: {cn}")
            
        elif self._drag_mode in ("move", "top_left", "top_right", "bottom_left", "bottom_right"):
            self._drag_mode = None
            self._drag_ann_idx = None
            self.status_v.set("Annotation adjusted.")
            self.has_unsaved_changes = True
            self._upd_stats()
            
        self._drag_mode = None

    def undo_last(self):
        anns=list(self._cur_anns())
        for i in range(len(anns)-1,-1,-1):
            if anns[i].manual: anns.pop(i); break
        self._set_anns(anns); self._upd_stats(); self._redraw()
        self.has_unsaved_changes = True

    # ═══════════════════════════ DETECTION LIST ══════════════════════════════
    def _show_class_dropdown(self, ann_idx, x_root, y_root):
        if not self.user_cls:
            messagebox.showinfo("","Add classes first (＋ Add)."); return
            
        menu = tk.Menu(self, tearoff=0, 
                       bg=BG_CARD, 
                       fg=TEXT_PRI, 
                       activebackground="#1F3A5F", 
                       activeforeground=ACCENT, 
                       font=("Courier", 10),
                       bd=1,
                       relief="solid")
                       
        for cn in self.user_cls:
            menu.add_command(label=cn, command=lambda name=cn: self._change_ann_class(ann_idx, name))
            
        menu.add_separator()
        menu.add_command(label="＋ Add Class...", command=self.add_cls)
        menu.post(x_root, y_root)

    def _change_ann_class(self, ann_idx, new_name):
        anns = list(self._cur_anns())
        if ann_idx < len(anns):
            anns[ann_idx].cls_name = new_name
            anns[ann_idx].cls_id   = self._cls_id(new_name)
            self._set_anns(anns)
            self._upd_stats()
            self._redraw()
            self.has_unsaved_changes = True
            self.status_v.set(f"Changed box class → {new_name}")

    def _edit_det_class(self):
        """Change class of selected detections. Uses _det_sel (persists after redraw)."""
        sel = self._det_sel if self._det_sel else self.det_lb.curselection()
        if not sel:
            messagebox.showinfo("","Click a detection in the list first, then click Edit Class."); return
        if not self.user_cls:
            messagebox.showinfo("","Add classes first (＋ Add)."); return

        # Get coordinates of cursor on screen
        x = self.winfo_pointerx()
        y = self.winfo_pointery()
        self._show_class_dropdown(sel[0], x, y)

    def _remove_dets(self):
        sel = self._det_sel if self._det_sel else self.det_lb.curselection()
        if not sel:
            messagebox.showinfo("","Click detections in the list first, then Remove."); return
        anns = self._cur_anns()
        self._det_sel = ()
        self._set_anns([a for i,a in enumerate(anns) if i not in sel])
        self._upd_stats(); self._redraw()
        self.has_unsaved_changes = True
        self.status_v.set(f"Removed {len(sel)} detection(s).")

    def _on_det_select(self, _=None):
        # Store selection immediately — _redraw → _refresh_det_list will clear it
        sel = self.det_lb.curselection()
        self._det_sel = sel
        anns = self._cur_anns()
        for i,a in enumerate(anns): a.selected = (i in sel)
        # Redraw WITHOUT refreshing det list so selection highlight stays
        if self.orig is None: return
        cw=self.canvas.winfo_width(); ch=self.canvas.winfo_height()
        if cw<2 or ch<2: return
        h,w=self.orig.shape[:2]
        s=min(cw/w,ch/h,1.0)
        nw,nh=int(w*s),int(h*s)
        ox=(cw-nw)//2; oy=(ch-nh)//2
        frame=cv2.resize(self.orig,(nw,nh),interpolation=cv2.INTER_LINEAR)
        frame=draw_anns(frame,anns,scale=s,
                        show_conf=self.show_conf_v.get(),
                        hi_manual=self.show_man_v.get())
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(ox,oy,anchor="nw",image=img)
        self.canvas._ref=img

    # ═══════════════════════════ FILE NAV & EVENTS ════════════════════════════
    def _on_file_click(self, _=None):
        """Single-click in file list: only navigate if exactly one item selected."""
        sel = self.file_lb.curselection()
        if len(sel) == 1:
            self._load_by_index(sel[0])

    def _on_file_wheel(self, event):
        """Windows mousewheel on file list → navigate one image at a time."""
        direction = -1 if event.delta > 0 else 1
        self._wheel_nav(direction)

    def _wheel_nav(self, direction):
        idx = max(0, min(len(self.file_list)-1, self.file_index + direction))
        if idx != self.file_index:
            self.file_lb.selection_clear(0, tk.END)
            self.file_lb.selection_set(idx)
            self.file_lb.see(idx)
            self._load_by_index(idx)

    def prev_img(self):
        if not self.file_list: return
        idx = max(0, self.file_index - 1)
        self.file_lb.selection_clear(0, tk.END)
        self.file_lb.selection_set(idx); self.file_lb.see(idx)
        self._load_by_index(idx)

    def next_img(self):
        if not self.file_list: return
        idx = min(len(self.file_list)-1, self.file_index + 1)
        self.file_lb.selection_clear(0, tk.END)
        self.file_lb.selection_set(idx); self.file_lb.see(idx)
        self._load_by_index(idx)

    def prev_frame(self): self.playing = False; self.prev_img()
    def next_frame(self): self.playing = False; self.next_img()

    def toggle_play(self):
        if len(self.file_list) < 2: return
        self.playing = not self.playing
        self.play_btn.configure(text="⏸" if self.playing else "▶")
        if self.playing:
            threading.Thread(target=self._slideshow, daemon=True).start()

    def _slideshow(self):
        while self.playing and self.file_index < len(self.file_list)-1:
            self.after(0, self.next_img); time.sleep(0.15)
        self.playing = False
        self.after(0, lambda: self.play_btn.configure(text="▶"))

    def _seek_debounced(self, val):
        """Debounce: only load image 120ms after slider stops moving."""
        if self._seek_after_id:
            self.after_cancel(self._seek_after_id)
        self._seek_after_id = self.after(120, lambda: self._seek(val))

    def _seek(self, val):
        if self.file_list:
            idx = int(float(val) / 100 * (len(self.file_list)-1))
            self._load_by_index(idx)

    # ═══════════════════════════ DELETE FILES ════════════════════════════════
    def select_all_files(self):
        if not self.file_list: return
        self.file_lb.selection_set(0, tk.END)

    def delete_selected_files(self):
        sel = self.file_lb.curselection()
        if not sel:
            messagebox.showinfo("","Select files in the list first.\n"
                                   "Ctrl+click or Shift+click for multiple."); return
        n = len(sel)
        if not messagebox.askyesno("Delete from project",
            f"Remove {n} image(s) from this project?\n\n"
            "The actual files on disk are NOT deleted.",
            icon="warning"): return
        remove_indices = sorted(sel, reverse=True)
        for i in remove_indices:
            if i < len(self.file_list):
                self.all_anns.pop(str(self.file_list[i]), None)
                self.file_list.pop(i)
        self.file_lb.clear()
        for p in self.file_list:
            self.file_lb.add_item(p)
        self.file_lb.start_loading()
        if not self.file_list:
            self.orig = None
            self.canvas.delete("all")
            self.file_lbl.configure(text="No file — open images or a folder")
            self._upd_stats(); return
        new_idx = min(remove_indices[-1], len(self.file_list)-1)
        self.file_lb.selection_set(new_idx)
        self._load_by_index(new_idx)
        self._color_file_list()
        self.has_unsaved_changes = True
        self.status_v.set(f"Removed {n} image(s) from project.")

    # ═══════════════════════════ EXPORT ══════════════════════════════════════
    def export_yolo(self):
        """Export ALL files as a ZIP — annotated images get labels, unannotated get empty .txt.
        ZIP structure: images/ labels/ classes.txt"""
        if not self.file_list:
            messagebox.showinfo("","No files loaded yet."); return

        save_path = filedialog.asksaveasfilename(
            title="Save YOLO Dataset ZIP",
            defaultextension=".zip",
            filetypes=[("ZIP archive","*.zip"),("All","*.*")],
            initialfile="yolo_dataset.zip")
        if not save_path: return

        self.status_v.set("Building ZIP…"); self.prog.set(0.05)
        import zipfile

        # Build class list from user classes + any annotation class names
        cls_list = list(self.user_cls)
        for anns in self.all_anns.values():
            for a in anns:
                if a.cls_name not in cls_list: cls_list.append(a.cls_name)
        cmap = {n:i for i,n in enumerate(cls_list)}

        imgs = lbls = skipped = 0
        total = len(self.file_list)

        try:
            with zipfile.ZipFile(save_path,"w",zipfile.ZIP_DEFLATED) as zf:
                # classes.txt always included
                zf.writestr("classes.txt","\n".join(cls_list)+"\n")

                for idx, file_path in enumerate(self.file_list):
                    src = Path(str(file_path))
                    if not src.exists():
                        skipped += 1
                        self.prog.set(0.05 + 0.9*(idx+1)/total)
                        continue

                    # Add image to ZIP
                    zf.write(str(src), f"images/{src.name}")
                    imgs += 1

                    # Build label — empty string if no annotations
                    anns = self.all_anns.get(str(src), [])
                    lines = []
                    if anns:
                        bgr = cv2.imread(str(src))
                        if bgr is not None:
                            h,w = bgr.shape[:2]
                            for a in anns:
                                cid = cmap.get(a.cls_name, a.cls_id)
                                x1,y1,x2,y2 = a.bbox
                                cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h
                                bw=(x2-x1)/w;     bh=(y2-y1)/h
                                lines.append(
                                    f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    zf.writestr(f"labels/{src.stem}.txt", "\n".join(lines))
                    lbls += 1
                    self.prog.set(0.05 + 0.9*(idx+1)/total)

        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            self.prog.set(0); return

        self.prog.set(1.0)
        annotated_count = sum(1 for p in self.file_list
                              if self.all_anns.get(str(p)))
        unannotated_count = len(self.file_list) - annotated_count - skipped
        size_mb = Path(save_path).stat().st_size / 1_000_000

        messagebox.showinfo("Exported",
            f"ZIP saved!\n\n"
            f"  Total images  : {imgs}\n"
            f"  Annotated     : {annotated_count} (with labels)\n"
            f"  Unannotated   : {unannotated_count} (empty labels)\n"
            f"  classes.txt   : {len(cls_list)} classes\n"
            f"  Skipped       : {skipped} (file not found)\n\n"
            f"Size: {size_mb:.1f} MB\n{save_path}")
        self.after(2000, lambda: self.prog.set(0))

    def import_yolo_dataset(self, import_as_manual=False):
        """Import a YOLO dataset ZIP.
        ONE dialog only — select the .zip file, everything else is automatic.
        Extracts into a subfolder next to the ZIP (same name as ZIP stem)."""
        import zipfile

        zip_path = filedialog.askopenfilename(
            title="Select YOLO Dataset ZIP  (.zip file)",
            filetypes=[("ZIP archive","*.zip")])   # ONLY show .zip files
        if not zip_path: return

        zip_path = Path(zip_path)

        if not zip_path.is_file() or not zipfile.is_zipfile(str(zip_path)):
            messagebox.showerror("Error",
                f"'{zip_path.name}' is not a valid ZIP file.\n\n"
                "Make sure to select the .zip file, not a folder."); return

        # Extract automatically — no second dialog
        extract_dir = zip_path.parent / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        self.status_v.set(f"Extracting {zip_path.name} …")
        self.prog.set(0.05)
        self.update_idletasks()

        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(extract_dir))
        except Exception as e:
            messagebox.showerror("Extract failed", str(e))
            self.prog.set(0); return

        self.prog.set(0.3)
        self._load_extracted_dataset(extract_dir, import_as_manual=import_as_manual)

    def _load_extracted_dataset(self, extract_dir, import_as_manual=False):
        """Shared logic: load images+labels from an extracted dataset folder."""
        extract_dir = Path(extract_dir)
        img_dir = extract_dir/"images" if (extract_dir/"images").is_dir() else extract_dir
        lbl_dir = extract_dir/"labels" if (extract_dir/"labels").is_dir() else extract_dir

        # Load class names
        cmap = {}
        cf = extract_dir/"classes.txt"
        if cf.exists():
            cls_lines = [l.strip() for l in cf.read_text().splitlines() if l.strip()]
            cmap = {i:n for i,n in enumerate(cls_lines)}
            for n in cls_lines:
                if n not in self.user_cls: self.user_cls.append(n)
        self._refresh_cls_menu()

        # Collect images
        exts = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
        img_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
        if not img_paths:
            messagebox.showerror("Error",
                f"No images found in:\n{img_dir}\n\n"
                "Make sure the ZIP contains an images/ folder."); return

        new_all_anns = dict(self.all_anns)
        loaded_imgs = loaded_anns = 0
        no_label = []
        total = len(img_paths)

        for i, img_path in enumerate(img_paths):
            lbl_path = lbl_dir/(img_path.stem+".txt")
            if not lbl_path.exists():
                lbl_path = img_path.parent/(img_path.stem+".txt")

            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            h, w = bgr.shape[:2]
            key  = str(img_path)
            anns = list(new_all_anns.get(key, []))

            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cid = int(parts[0])
                    cx,cy,bw,bh = map(float, parts[1:5])
                    x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h)
                    x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)
                    cn = cmap.get(cid, f"class_{cid}")
                    if cn not in self.user_cls: self.user_cls.append(cn)
                    anns.append(Annotation(cid, cn, [x1,y1,x2,y2], 1.0, manual=import_as_manual))
                    loaded_anns += 1
            else:
                no_label.append(img_path.name)

            new_all_anns[key] = anns
            loaded_imgs += 1
            self.prog.set(0.3 + 0.65*(i+1)/total)

        if loaded_imgs == 0:
            messagebox.showerror("Error","No valid images could be read."); return

        self.all_anns = new_all_anns
        self._refresh_cls_menu()
        self._set_file_list(img_paths, keep_existing=False)
        self.prog.set(1.0)

        annotated = sum(1 for p in img_paths if new_all_anns.get(str(p)))
        msg = (f"Dataset imported!\n\n"
               f"Images     : {loaded_imgs}\n"
               f"Annotated  : {annotated}\n"
               f"Labels     : {loaded_anns}\n"
               f"Classes    : {len(cmap) or len(self.user_cls)}")
        if no_label:
            msg += f"\nNo label   : {len(no_label)} image(s)"
        messagebox.showinfo("Imported", msg)
        self.after(2000, lambda: self.prog.set(0))

    def import_yolo_dataset_manual(self):
        self.import_yolo_dataset(import_as_manual=True)

    def import_yolo_dataset_ai(self):
        self.import_yolo_dataset(import_as_manual=False)

    def export_yolo_manual_only(self):
        """Export ONLY manually made annotations as a ZIP.
        ZIP structure: images/ labels/ classes.txt"""
        if not self.file_list:
            messagebox.showinfo("","No files loaded yet."); return

        save_path = filedialog.asksaveasfilename(
            title="Save YOLO Manual-Only Dataset ZIP",
            defaultextension=".zip",
            filetypes=[("ZIP archive","*.zip"),("All","*.*")],
            initialfile="yolo_manual_dataset.zip")
        if not save_path: return

        self.status_v.set("Building ZIP…"); self.prog.set(0.05)
        import zipfile

        # Build class list from user classes + any annotation class names
        cls_list = list(self.user_cls)
        for anns in self.all_anns.values():
            for a in anns:
                if a.manual and a.cls_name not in cls_list:
                    cls_list.append(a.cls_name)
        cmap = {n:i for i,n in enumerate(cls_list)}

        imgs = lbls = skipped = 0
        total = len(self.file_list)

        try:
            with zipfile.ZipFile(save_path,"w",zipfile.ZIP_DEFLATED) as zf:
                # classes.txt always included
                zf.writestr("classes.txt","\n".join(cls_list)+"\n")

                for idx, file_path in enumerate(self.file_list):
                    src = Path(str(file_path))
                    if not src.exists():
                        skipped += 1
                        self.prog.set(0.05 + 0.9*(idx+1)/total)
                        continue

                    # Add image to ZIP
                    zf.write(str(src), f"images/{src.name}")
                    imgs += 1

                    # Build label — empty string if no manual annotations
                    anns = self.all_anns.get(str(src), [])
                    manual_anns = [a for a in anns if a.manual]
                    lines = []
                    if manual_anns:
                        bgr = cv2.imread(str(src))
                        if bgr is not None:
                            h,w = bgr.shape[:2]
                            for a in manual_anns:
                                cid = cmap.get(a.cls_name, a.cls_id)
                                x1,y1,x2,y2 = a.bbox
                                cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h
                                bw=(x2-x1)/w;     bh=(y2-y1)/h
                                lines.append(
                                    f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    zf.writestr(f"labels/{src.stem}.txt", "\n".join(lines))
                    lbls += 1
                    self.prog.set(0.05 + 0.9*(idx+1)/total)

        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            self.prog.set(0); return

        self.prog.set(1.0)
        annotated_count = sum(1 for p in self.file_list
                              if any(a.manual for a in self.all_anns.get(str(p), [])))
        unannotated_count = len(self.file_list) - annotated_count - skipped
        size_mb = Path(save_path).stat().st_size / 1_000_000

        messagebox.showinfo("Exported",
            f"Manual-Only ZIP saved!\n\n"
            f"  Total images  : {imgs}\n"
            f"  Manual Annot. : {annotated_count} (with labels)\n"
            f"  Unannotated   : {unannotated_count} (empty labels)\n"
            f"  classes.txt   : {len(cls_list)} classes\n"
            f"  Skipped       : {skipped} (file not found)\n\n"
            f"Size: {size_mb:.1f} MB\n{save_path}")
        self.after(2000, lambda: self.prog.set(0))

    # ═══════════════════════════ RENDER ══════════════════════════════════════
    def _redraw(self):
        if self.orig is None: return
        cw=self.canvas.winfo_width(); ch=self.canvas.winfo_height()
        if cw<2 or ch<2: return
        h,w=self.orig.shape[:2]
        s=min(cw/w,ch/h,1.0); self.scale=s
        nw,nh=int(w*s),int(h*s)
        ox=(cw-nw)//2; oy=(ch-nh)//2; self.offset=(ox,oy)
        frame=cv2.resize(self.orig,(nw,nh),interpolation=cv2.INTER_LINEAR)
        anns=self._cur_anns()
        frame=draw_anns(frame,anns,scale=s,
                        show_conf=self.show_conf_v.get(),
                        hi_manual=self.show_man_v.get())
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(ox,oy,anchor="nw",image=img)
        self.canvas._ref=img
        self._refresh_det_list(anns)

    def _refresh_det_list(self,anns):
        self.det_lb.delete(0,tk.END)
        for i,a in enumerate(anns):
            tag="[M]" if a.manual else "[A]"
            cf="" if a.manual else f" {a.confidence:.2f}"
            self.det_lb.insert(tk.END,f"  {tag} [{i:02d}] {a.cls_name:<16}{cf}")
            self.det_lb.itemconfig(i,fg=MANUAL_COL if a.manual else cls_col(a.cls_id))

    def _upd_stats(self):
        anns=self._cur_anns()
        n=len(anns); m=sum(1 for a in anns if a.manual)
        tot=sum(len(v) for v in self.all_anns.values())
        done=sum(1 for v in self.all_anns.values() if v)
        pend=len(self.file_list)-done
        cust=" 🧠" if self.use_custom_v.get() else ""
        self.stats_lbl.configure(
            text=f"Here:{n}({m}M) | All:{tot} | ✅{done} | ⏳{pend}{cust}")
        self.frm_lbl.configure(text=f"{self.file_index+1}/{len(self.file_list)}"
                               if self.file_list else "0/0")
        if self.file_list:
            self.seek_sl.configure(to=max(1,len(self.file_list)-1))
            self.seek_v.set(self.file_index)

if __name__=="__main__":
    App().mainloop()
