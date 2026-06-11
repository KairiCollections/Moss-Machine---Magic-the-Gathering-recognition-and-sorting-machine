#!/usr/bin/env python3
"""
Enhanced GUI Interface for Card Scanner with Arduino Controls
Features: Live camera, CSV export, Arduino monitoring/control
"""
# ---------------------------------------------------------------------------
# Auto-install any missing required packages before importing them
# ---------------------------------------------------------------------------
def _ensure_packages():
    import sys, subprocess
    _REQUIRED = [
        "cv2:opencv-python",
        "numpy:numpy",
        "PIL:Pillow",
        "requests:requests",
        "qrcode:qrcode",
        "imagehash:imagehash",
        "truststore:truststore",
    ]
    missing = []
    for spec in _REQUIRED:
        mod, pkg = spec.split(":")
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    for pkg in missing:
        try:
            print(f"[startup] Installing {pkg}...", flush=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-input", pkg]
            )
        except Exception as e:
            print(f"[startup] Warning: could not install {pkg}: {e}", flush=True)
_ensure_packages()
# Inject Windows/macOS/Linux native certificate store so that HTTPS requests
# succeed even when the server's intermediate CA is missing from certifi.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
# ---------------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import os
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import time
import csv
from datetime import datetime
from optimized_scanner import OptimizedCardScanner
import qrcode
import io
import requests
import urllib.parse
import sqlite3
import glob
from error_codes import ERROR_CODES, lookup as _ec_lookup, format_log as _elog

# Plugin support
try:
    from plugins.loader import PluginManager
except Exception:
    PluginManager = None


class ScannerGUI:
    # Hard-coded mapping from display game name -> recognition DB numeric id
    HARD_CODE_GAME_IDS = {
                        "Magic: The Gathering": "1",
                        "YuGiOh": "2",
                        "Pokemon": "3",
                        "Epic": "7",
                        "WoW": "10",
                        "Cardfight Vanguard": "12",
                        "Force of Will": "13",
                        "Dice Masters": "14",
                        "Future Card BuddyFight": "15",
                        "Weiss Schwarz": "16",
                        "Dragon Ball Z TCG": "18",
                        "Final Fantasy TCG": "19",
                        "UniVersus": "20",
                        "Star Wars: Destiny": "21",
                        "Dragon Ball Super: Masters": "22",
                        "Dragoborne": "23",
                        "MetaX TCG": "25",
                        "Zombie World Order TCG": "31",
                        "The Caster Chronicles": "32",
                        "My Little Pony CCG": "33",
                        "Exodus TCG": "42",
                        "Lightseekers TCG": "43",
                        "Munchkin CCG": "48",
                        "Warhammer Age of Sigmar Champions TCG": "49",
                        "Transformers TCG": "51",
                        "Bakugan TCG": "52",
                        "Argent Saga TCG": "55",
                        "Flesh and Blood TCG": "56",
                        "Digimon Card Game": "57",
                        "Gate Ruler": "59",
                        "MetaZoo": "60",
                        "WIXOSS": "61",
                        "One Piece Card Game": "62",
                        "Disney Lorcana": "63",
                        "Battle Spirits Saga": "64",
                        "Shadowverse: Evolve": "65",
                        "Grand Archive TCG": "66",
                        "Akora TCG": "67",
                        "Kryptik TCG": "68",
                        "Sorcery: Contested Realm": "69",
                        "Alpha Clash": "70",
                        "Star Wars: Unlimited": "71",
                        "Dragon Ball Super: Fusion World": "72",
                        "Union Arena": "73",
                        "Elestrals": "75",
                        "Pokemon Japan": "76",
                        "Gundam Card Game": "77",
                        "hololive OFFICIAL CARD GAME": "78",
                        "Godzilla Card Game": "79",
                        "Riftbound: League of Legends Trading Card Game": "80",
                        "Waifu": "81",
    }
    def __init__(self, root):
        # Prune any hard-coded game mappings that correspond to games with load=0
        try:
            self._prune_hardcoded_map()
        except Exception:
            pass
        # Scan local recognition_data for available DBs
        try:
            self._scan_local_recognition_files()
        except Exception:
            self.available_game_ids = set()
        self.root = root
        self.root.title("Card Scanner Pro - Enhanced Control Interface")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#1e1e1e')
        
        # Configure default fonts for better clarity
        self.default_font = ('Segoe UI', 10)
        self.header_font = ('Segoe UI', 11, 'bold')
        self.title_font = ('Segoe UI', 12, 'bold')
        
        # Scanner & camera
        self.scanner = None
        self.camera = None
        self.running = False
        self.auto_scan = True
        self.auto_scan_var = tk.BooleanVar(value=True)
        self.last_scan_time = 0
        self.scan_cooldown = 2.0
        
        # Detection state
        self.current_frame = None
        self.detection_info = None
        # Prevent overlapping scans
        self._scan_in_progress = False
        # Exposure normalization state (to reduce flashing)
        self._last_v_median = None
        self._target_v = 120
        
        # Collection settings (for SKU support)
        self.collection_enabled = True
        self.default_condition = tk.StringVar(value="Near Mint")
        self.default_language = tk.StringVar(value="EN")
        self.default_foil = tk.BooleanVar(value=False)
        self.auto_save_mode = tk.BooleanVar(value=True)  # Auto-save without prompts
        self.export_format = tk.StringVar(value="TCGTraders")  # TCGTraders or TCGPlayer
        
        # Match backend settings (pHash only)
        # - PHASH_RGB: per-channel RGB pHash matching
        # - PHASH_GRAY: grayscale pHash matching
        self.vector_backend = tk.StringVar(value="PHASH_RGB")

        # Matching strictness controls (adjustable via GUI)
        self.match_threshold_var = tk.IntVar(value=90)
        
        # Arduino monitoring
        self.arduino_monitoring = False
        self.arduino_monitor_thread = None
        self.sensor_data = {'range': 0, 'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0, 'z_min': 0, 'z_max': 0, 'started': 0}
        self.machine_started = False  # Track if Arduino is in Started state
        # Camera canvas size (adjustable to make UI fit on smaller screens)
        self.canvas_width = 600
        self.canvas_height = 540

        # Display smoothing to reduce perceived flicker (blend successive frames)
        self.display_smooth_alpha = 0.1  # 0 = no smoothing, 1 = full hold previous frame
        self._last_display_frame = None

        # Compact UI toggle (helps fit everything on smaller screens)
        self.compact_mode_var = tk.BooleanVar(value=False)

        # Error log: list of (timestamp_str, code, message, fix)
        self.error_log = []
        self._error_badge_var = tk.StringVar(value="")

        self._setup_gui()
        # Prompt to download unified_card_database.db and load allowed game ids
        try:
            self._ensure_unified_db_prompt()
        except Exception:
            pass
        self._init_scanner()
        # Plugin manager: discover optional camera/recognition/arduino plugins
        self.plugin_manager = None
        self.camera_plugin = None
        self.recognition_plugin = None
        self.arduino_plugin = None
        if PluginManager is not None:
            try:
                self.plugin_manager = PluginManager()
                self.plugin_manager.discover_plugins()
                self.camera_plugin = self.plugin_manager.get_plugin('camera')
                self.recognition_plugin = self.plugin_manager.get_plugin('recognition')
                self.arduino_plugin = self.plugin_manager.get_plugin('arduino')
                self.log_status(f"Plugins loaded: {', '.join(self.plugin_manager.plugins.keys())}")
            except Exception as e:
                self.log_status(f"Plugin loader error: {e}", error=True)
                self.log_error('E108', str(e))
        # Start a lightweight poll to detect combobox changes on platforms where events may not fire
        try:
            self._last_game_var = self.game_var.get()
            self.root.after(500, self._poll_game_var)
        except Exception:
            pass
    def _setup_gui(self):
        """Setup main GUI with tabbed interface"""

        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Camera feed
        left_panel = tk.Frame(main_frame, bg='#1e1e1e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self._setup_camera_panel(left_panel)
        
        # Right: Tabbed control panel
        right_panel = tk.Frame(main_frame, bg='#1e1e1e', width=700)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Scanner Settings
        scanner_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(scanner_tab, text='📷 Scanner')
        self._setup_scanner_tab(scanner_tab)
        
        # Tab 2: Arduino Control
        arduino_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(arduino_tab, text='🤖 Arduino')
        self._setup_arduino_tab(arduino_tab)
        
        # Tab 3: Export & Stats
        export_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(export_tab, text='📊 Export')
        self._setup_export_tab(export_tab)
        
        # Tab 5: QR Codes
        qr_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(qr_tab, text='📱 Join Us')
        self._setup_qr_tab(qr_tab)

        # Tab 6: Downloads (server-hosted recognition DBs)
        downloads_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(downloads_tab, text='⬇️ Downloads')
        try:
            self._setup_downloads_tab(downloads_tab)
        except Exception as _e:
            tk.Label(downloads_tab, text=f"Downloads tab error: {_e}", bg='#1e1e1e', fg='#ff4444').pack(pady=20)

        # Tab 7: Tutorial / How To Use
        tutorial_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(tutorial_tab, text='📖 Tutorial')
        self._setup_tutorial_tab(tutorial_tab)

        # Tab 8: Troubleshooting
        self._troubleshooting_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(self._troubleshooting_tab, text='🔧 Troubleshoot')
        self._setup_troubleshooting_tab(self._troubleshooting_tab)
        
    def _setup_camera_panel(self, parent):
        """Setup camera feed panel"""
        tk.Label(parent, text="LIVE CAMERA FEED", bg='#1e1e1e', fg='#ffffff', 
                font=self.title_font).pack(pady=(0, 10))

        self.canvas = tk.Canvas(parent, width=self.canvas_width, height=self.canvas_height, bg='#000000', 
                               highlightthickness=3, highlightbackground='#4CAF50')
        self.canvas.pack()
        
        # Controls
        control_frame = tk.Frame(parent, bg='#1e1e1e')
        control_frame.pack(pady=15)
        
        self.start_btn = tk.Button(control_frame, text="▶ Start", command=self.start_camera,
                                   bg='#4CAF50', fg='white', font=self.header_font,
                                   padx=20, pady=10, cursor='hand2', relief=tk.FLAT,
                                   activebackground='#45a049')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="⏸ Stop", command=self.stop_camera,
                                  bg='#f44336', fg='white', font=('Arial', 11, 'bold'),
                                  padx=15, pady=8, cursor='hand2', state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Compact UI checkbox to reduce fonts/padding and shrink panels
        self.compact_cb = tk.Checkbutton(control_frame, text="Compact UI", variable=self.compact_mode_var,
                         command=self.apply_compact_mode, bg='#1e1e1e', fg='#fff',
                         selectcolor='#2b2b2b', font=('Arial', 9))
        self.compact_cb.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status indicator
        self.scan_mode_label = tk.Label(parent, text="Auto-Scan Mode: ON", bg='#1e1e1e', 
                                       fg='#4CAF50', font=('Arial', 10, 'bold'))
        self.scan_mode_label.pack(pady=5)
    
    def _setup_scanner_tab(self, parent):
        """Setup scanner settings tab"""
        canvas = tk.Canvas(parent, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg='#1e1e1e')
        
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # COLLECTION SETTINGS (SKU Support)
        self._section_label(frame, "COLLECTION & SKU SETTINGS")
        
        coll_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        coll_frame.pack(fill=tk.X, padx=20, pady=3)
        
        # Scan Mode
        mode_frame = tk.Frame(coll_frame, bg='#1a1a1a')
        mode_frame.pack(fill=tk.X, pady=3, padx=8)
        tk.Label(mode_frame, text="Scan Mode:", bg='#1a1a1a', fg='#fff', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=3)
        
        tk.Radiobutton(mode_frame, text="Auto-Scan", variable=self.auto_scan_var, value=True,
                      command=self.toggle_auto_scan, bg='#1a1a1a', fg='#fff', 
                      selectcolor='#404040', font=('Arial', 8)).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(mode_frame, text="Manual", variable=self.auto_scan_var, value=False,
                      command=self.toggle_auto_scan, bg='#1a1a1a', fg='#fff',
                      selectcolor='#404040', font=('Arial', 8)).pack(side=tk.LEFT, padx=8)
        
        # Default Condition
        cond_frame = tk.Frame(coll_frame, bg='#1a1a1a')
        cond_frame.pack(fill=tk.X, pady=3, padx=8)
        tk.Label(cond_frame, text="Condition:", bg='#1a1a1a', fg='#fff', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        conditions = ["Near Mint", "Lightly Played", "Moderately Played", "Heavily Played", "Damaged"]
        self.condition_combo = ttk.Combobox(cond_frame, textvariable=self.default_condition, 
                                            values=conditions, state='readonly', width=16, font=('Arial', 8))
        self.condition_combo.pack(side=tk.LEFT, padx=3)
        
        # Default Language
        lang_frame = tk.Frame(coll_frame, bg='#1a1a1a')
        lang_frame.pack(fill=tk.X, pady=3, padx=8)
        tk.Label(lang_frame, text="Language:", bg='#1a1a1a', fg='#fff', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        languages = ["EN", "JP", "FR", "DE", "IT", "ES", "PT", "KO", "CN"]
        self.language_combo = ttk.Combobox(lang_frame, textvariable=self.default_language,
                                          values=languages, state='readonly', width=16, font=('Arial', 8))
        self.language_combo.pack(side=tk.LEFT, padx=3)
        
        # Default Foil and Auto-Save
        checks_frame = tk.Frame(coll_frame, bg='#1a1a1a')
        checks_frame.pack(fill=tk.X, pady=3, padx=8)
        tk.Checkbutton(checks_frame, text="Default Foil", variable=self.default_foil,
                      bg='#1a1a1a', fg='#fff', selectcolor='#404040', 
                      font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checks_frame, text="Auto-Save (no prompts)", 
                      variable=self.auto_save_mode, bg='#1a1a1a', fg='#fff',
                      selectcolor='#404040', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Match Backend Selection (pHash only)
        self._section_label(frame, "MATCH BACKEND")
        backend_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        backend_frame.pack(fill=tk.X, padx=20, pady=3)
        
        backend_row = tk.Frame(backend_frame, bg='#1a1a1a')
        backend_row.pack(fill=tk.X, pady=5, padx=8)

        tk.Radiobutton(
            backend_row,
            text="pHash (RGB channels)",
            variable=self.vector_backend,
            value="PHASH_RGB",
            command=self.update_vector_backend,
            bg='#1a1a1a',
            fg='#fff',
            selectcolor='#404040',
            font=('Arial', 9),
        ).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(
            backend_row,
            text="pHash (Grayscale)",
            variable=self.vector_backend,
            value="PHASH_GRAY",
            command=self.update_vector_backend,
            bg='#1a1a1a',
            fg='#fff',
            selectcolor='#404040',
            font=('Arial', 9),
        ).pack(side=tk.LEFT, padx=8)
        
        # Backend status
        self.backend_status = tk.Label(backend_frame, text="✅ pHash (RGB channels)",
                                       bg='#1a1a1a', fg='#00ff00', font=('Arial', 8))
        self.backend_status.pack(pady=3)

        # Matching strictness controls
        match_frame = tk.Frame(backend_frame, bg='#1a1a1a')
        match_frame.pack(pady=6)

        tk.Label(match_frame, text="Match Threshold:", bg='#1a1a1a', fg='#fff', font=('Arial', 9)).pack(side=tk.LEFT)
        threshold_spin = tk.Spinbox(match_frame, from_=1, to=512, textvariable=self.match_threshold_var, width=6)
        threshold_spin.pack(side=tk.LEFT, padx=6)

        # Update scanner settings when values change (if scanner initialized)
        def _apply_match_settings(*args):
            try:
                if self.scanner:
                    self.scanner.scan_threshold = int(self.match_threshold_var.get())
            except Exception:
                pass

        # Trace changes
        self.match_threshold_var.trace_add('write', _apply_match_settings)
        
        # Sorting mode (dropdown)
        self._section_label(frame, "SORTING MODE")
        # internal key var used throughout the code (e.g. 'color', 'price', 'alpha')
        self.sorting_var = tk.StringVar(value="color")

        # Display labels mapped to internal keys
        sorting_options = [
            ("Color", "color"),
            ("Mana Value", "mana_value"),
            ("Set", "set"),
            ("Price", "price"),
            ("Type", "type"),
            ("Buy Mode", "buy"),
            ("Alpha (A-Z)", "alpha"),
            ("Rarity", "rarity"),
            ("Finish (Foil)", "finish")
        ]

        # Combobox shows friendly labels but we store internal keys in self.sorting_var
        display_labels = [label for label, key in sorting_options]
        label_to_key = {label: key for label, key in sorting_options}

        # Create combobox
        # Make sorting combobox half width to save horizontal space
        self.sorting_combo = ttk.Combobox(frame, values=display_labels, state='readonly', width=14)

        # Set combobox to current sorting_var value (find matching label)
        current_label = next((lbl for lbl, k in sorting_options if k == self.sorting_var.get()), display_labels[0])
        self.sorting_combo.set(current_label)
        self.sorting_combo.pack(padx=20, pady=5, anchor=tk.W)

        # When user selects a display label, update the internal sorting_var to the mapped key
        def _on_sort_selected(event=None):
            sel = self.sorting_combo.get()
            if sel in label_to_key:
                self.sorting_var.set(label_to_key[sel])

        self.sorting_combo.bind('<<ComboboxSelected>>', _on_sort_selected)
        
        # Filters - Compact side-by-side layout
        self._section_label(frame, "FILTERS")
        
        # Row 1: Game and Set filters
        filter_row1 = tk.Frame(frame, bg='#1e1e1e')
        filter_row1.pack(fill=tk.X, padx=20, pady=3)
        
        tk.Label(filter_row1, text="Game:", bg='#1e1e1e', fg='#aaa', font=('Arial', 9), width=8, anchor=tk.W).pack(side=tk.LEFT)
        self.game_var = tk.StringVar(value="All Games")
        self.game_combo = ttk.Combobox(filter_row1, textvariable=self.game_var, state='readonly', width=25)
        self.game_combo.pack(side=tk.LEFT, padx=5)
        # Apply button to force loading a selected game (helpful if events don't fire)
        self.apply_game_btn = tk.Button(filter_row1, text="Apply", command=self.on_game_change,
                        bg='#3a7be0', fg='white', font=('Arial', 8), padx=8)
        self.apply_game_btn.pack(side=tk.LEFT, padx=(6,0))
        self.game_combo.bind('<<ComboboxSelected>>', self.on_game_change)
        # Ensure changes to the underlying var also trigger the handler
        try:
            self.game_var.trace_add('write', lambda *args: self.on_game_change())
        except Exception:
            pass
        # Also bind some lower-level events in case the virtual event is not firing
        try:
            self.game_combo.bind('<ButtonRelease-1>', lambda e: self.on_game_change())
            self.game_combo.bind('<Return>', lambda e: self.on_game_change())
        except Exception:
            pass
        
        tk.Label(filter_row1, text="Set:", bg='#1e1e1e', fg='#aaa', font=('Arial', 9), width=6, anchor=tk.W).pack(side=tk.LEFT, padx=(15, 0))
        self.set_var = tk.StringVar(value="All Sets")
        self.set_combo = ttk.Combobox(filter_row1, textvariable=self.set_var, state='readonly', width=25)
        self.set_combo.pack(side=tk.LEFT, padx=5)
        
        # Row 2: Rarity and Foil filters
        filter_row2 = tk.Frame(frame, bg='#1e1e1e')
        filter_row2.pack(fill=tk.X, padx=20, pady=3)
        
        tk.Label(filter_row2, text="Rarity:", bg='#1e1e1e', fg='#aaa', font=('Arial', 9), width=8, anchor=tk.W).pack(side=tk.LEFT)
        self.rarity_var = tk.StringVar(value="All Rarities")
        self.rarity_combo = ttk.Combobox(filter_row2, textvariable=self.rarity_var, state='readonly', width=25)
        self.rarity_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(filter_row2, text="Foil:", bg='#1e1e1e', fg='#aaa', font=('Arial', 9), width=6, anchor=tk.W).pack(side=tk.LEFT, padx=(15, 0))
        self.foil_var = tk.StringVar(value="All Foil Types")
        self.foil_combo = ttk.Combobox(filter_row2, textvariable=self.foil_var, state='readonly', width=25)
        self.foil_combo.pack(side=tk.LEFT, padx=5)
        
        # Price threshold
        threshold_frame = tk.Frame(frame, bg='#1e1e1e')
        threshold_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(threshold_frame, text="Price Threshold ($):", bg='#1e1e1e', fg='#aaa', font=('Arial', 9)).pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="1000000")
        tk.Entry(threshold_frame, textvariable=self.threshold_var, width=15, 
                bg='#404040', fg='#fff', insertbackground='#fff').pack(side=tk.LEFT, padx=5)
        
        # Inventory
        self._section_label(frame, "INVENTORY")
        self.inventory_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="Track Inventory (Reject Duplicates)", variable=self.inventory_var,
                      command=self.toggle_inventory, bg='#1e1e1e', fg='#fff', selectcolor='#404040',
                      font=('Arial', 10), activebackground='#1e1e1e', activeforeground='#fff').pack(anchor=tk.W, padx=20, pady=5)
        
        # Status log
        self._section_label(frame, "STATUS LOG")
        status_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.BOTH, padx=20, pady=5, expand=True)
        
        self.status_text = tk.Text(status_frame, height=12, bg='#1a1a1a', fg='#00ff00',
                                   font=('Consolas', 8), wrap=tk.WORD, state=tk.DISABLED)
        status_scroll = tk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_arduino_tab(self, parent):
        """Setup Arduino control tab"""
        canvas = tk.Canvas(parent, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg='#1e1e1e')
        
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Connection
        self._section_label(frame, "CONNECTION")
        conn_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        conn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        conn_row = tk.Frame(conn_frame, bg='#1a1a1a')
        conn_row.pack(pady=8, padx=10)
        
        tk.Label(conn_row, text="Port:", bg='#1a1a1a', fg='#fff', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        self.serial_port_var = tk.StringVar(value="COM3")
        tk.Entry(conn_row, textvariable=self.serial_port_var, width=8, bg='#404040', fg='#fff', insertbackground='#fff').pack(side=tk.LEFT, padx=3)
        
        tk.Label(conn_row, text="Baud:", bg='#1a1a1a', fg='#fff', font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 3))
        self.baud_var = tk.StringVar(value="9600")
        tk.Entry(conn_row, textvariable=self.baud_var, width=8, bg='#404040', fg='#fff', insertbackground='#fff').pack(side=tk.LEFT, padx=3)
        
        self.connect_btn = tk.Button(conn_row, text="Connect", command=self.connect_arduino,
                                     bg='#FF9800', fg='white', font=('Arial', 9, 'bold'),
                                     padx=12, pady=4, cursor='hand2', relief=tk.FLAT)
        self.connect_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Start/Stop Machine Controls
        self._section_label(frame, "MACHINE STATE")
        state_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        state_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.machine_state_label = tk.Label(state_frame, text="● STOPPED", bg='#1a1a1a', 
                                            fg='#ff4444', font=('Arial', 10, 'bold'))
        self.machine_state_label.pack(pady=5)
        
        btn_row = tk.Frame(state_frame, bg='#1a1a1a')
        btn_row.pack(pady=5)
        
        self.start_machine_btn = tk.Button(btn_row, text="▶ START", command=self.start_machine,
                                          bg='#4CAF50', fg='white', font=('Arial', 9, 'bold'),
                                          padx=15, pady=5, cursor='hand2', relief=tk.FLAT)
        self.start_machine_btn.pack(side=tk.LEFT, padx=3)
        
        self.stop_machine_btn = tk.Button(btn_row, text="⏹ STOP", command=self.stop_machine,
                                         bg='#f44336', fg='white', font=('Arial', 9, 'bold'),
                                         padx=15, pady=5, cursor='hand2', state=tk.DISABLED, relief=tk.FLAT)
        self.stop_machine_btn.pack(side=tk.LEFT, padx=3)
        
        self.home_btn = tk.Button(btn_row, text="🏠 HOME", command=self.arduino_home,
                 bg='#9C27B0', fg='white', font=('Arial', 9, 'bold'),
                 padx=15, pady=5, cursor='hand2', relief=tk.FLAT)
        self.home_btn.pack(side=tk.LEFT, padx=3)
        
        # Sensors & Monitoring - Compact display
        self._section_label(frame, "SENSORS & MONITORING")
        sensor_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        sensor_frame.pack(fill=tk.X, padx=20, pady=5)
        
        sensor_row = tk.Frame(sensor_frame, bg='#1a1a1a')
        sensor_row.pack(pady=5)
        
        self.range_label = tk.Label(sensor_row, text="Range: -- mm", bg='#1a1a1a', fg='#00ff00', font=('Consolas', 9))
        self.range_label.pack(side=tk.LEFT, padx=10)
        
        self.monitor_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sensor_row, text="Live Monitor", variable=self.monitor_var,
                      command=self.toggle_arduino_monitoring, bg='#1a1a1a', fg='#fff',
                      selectcolor='#404040', font=('Arial', 9)).pack(side=tk.LEFT, padx=10)
        
        # Endstops - compact grid
        endstop_frame = tk.Frame(sensor_frame, bg='#1a1a1a')
        endstop_frame.pack(pady=3)
        
        self.endstop_labels = {}
        for i, (name, key) in enumerate([("X-", "x_min"), ("X+", "x_max"), ("Y-", "y_min"), 
                                         ("Y+", "y_max"), ("Z-", "z_min"), ("Z+", "z_max")]):
            lbl = tk.Label(endstop_frame, text=f"{name}:?", bg='#1a1a1a', fg='#888', font=('Consolas', 8), width=6)
            lbl.grid(row=i//6, column=i%6, padx=2, pady=2)
            self.endstop_labels[key] = lbl
        
        # Motor toggles - compact 2-column layout
        self._section_label(frame, "MOTORS & RELAYS")
        motor_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        motor_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.motor_vars = {}
        self.motor_checkboxes = {}
        # Motors enabled by default (LOW = enabled), Vacuums off, Lights on
        defaults = {
            "Xenable": True, "Yenable": True, "Zenable": True, 
            "E0enable": True, "E1enable": True,
            "Vacuum1": False, "Vacuum2": False, "Lights": True
        }
        
        motor_grid = tk.Frame(motor_frame, bg='#1a1a1a')
        motor_grid.pack(pady=5, padx=10)
        
        motor_names = ["Xenable", "Yenable", "Zenable", "E0enable", "E1enable", "Vacuum1", "Vacuum2", "Lights"]
        for i, name in enumerate(motor_names):
            var = tk.BooleanVar(value=defaults.get(name, False))
            self.motor_vars[name] = var
            cb = tk.Checkbutton(motor_grid, text=name, variable=var,
                          command=lambda n=name: self.toggle_motor(n), bg='#1a1a1a', fg='#fff',
                          selectcolor='#404040', font=('Arial', 8), width=10)
            cb.grid(row=i//4, column=i%4, sticky=tk.W, padx=3, pady=1)
            self.motor_checkboxes[name] = cb
        
        # Movement controls - compact
        self._section_label(frame, "MANUAL MOVEMENT")
        move_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        move_frame.pack(fill=tk.X, padx=20, pady=5)
        
        btn_frame = tk.Frame(move_frame, bg='#1a1a1a')
        btn_frame.pack(pady=5)
        
        self.move_steps_var = tk.StringVar(value="1")

        self.movement_buttons = []
        for row, axis in enumerate([('X', 'CalibrateX'), ('Y', 'CalibrateY')]):
            btn_minus = tk.Button(btn_frame, text=f"{axis[0]}-",
                     command=lambda a=axis[1]: self.send_arduino_command(
                         f"{a}1,{self.move_steps_var.get()}"),
                     bg='#607D8B', fg='white', font=('Arial', 8), padx=8, pady=3, relief=tk.FLAT)
            btn_minus.grid(row=row, column=0, padx=2, pady=1)
            self.movement_buttons.append(btn_minus)

            tk.Label(btn_frame, text=axis[0], bg='#1a1a1a', fg='#fff', font=('Arial', 9, 'bold'), width=3).grid(row=row, column=1, padx=5)

            btn_plus = tk.Button(btn_frame, text=f"{axis[0]}+",
                     command=lambda a=axis[1]: self.send_arduino_command(
                         f"{a}2,{self.move_steps_var.get()}"),
                     bg='#607D8B', fg='white', font=('Arial', 8), padx=8, pady=3, relief=tk.FLAT)
            btn_plus.grid(row=row, column=2, padx=2, pady=1)
            self.movement_buttons.append(btn_plus)

        # Steps input row
        steps_row = tk.Frame(move_frame, bg='#1a1a1a')
        steps_row.pack(pady=(2, 5))
        tk.Label(steps_row, text="Steps:", bg='#1a1a1a', fg='#aaa', font=('Arial', 9)).pack(side=tk.LEFT, padx=(8, 4))
        steps_entry = tk.Spinbox(steps_row, from_=1, to=9999, textvariable=self.move_steps_var,
                                  width=6, bg='#404040', fg='#fff', buttonbackground='#555',
                                  insertbackground='#fff', font=('Arial', 9))
        steps_entry.pack(side=tk.LEFT)
        tk.Label(steps_row, text="(per button press)", bg='#1a1a1a', fg='#666', font=('Arial', 8)).pack(side=tk.LEFT, padx=(6, 0))
        self.movement_buttons.append(steps_entry)  # disable during machine run
        
        # Parameters - compact 2-column layout
        self._section_label(frame, "PARAMETERS")
        params_container = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        params_container.pack(fill=tk.X, padx=20, pady=5)
        
        params_frame = tk.Frame(params_container, bg='#1a1a1a')
        params_frame.pack(pady=5, padx=5)
        
        self.param_vars = {}
        params = [
            ("Speed", "speed", "700"), ("Z Spd", "zspeed", "75"),
            ("Z E Spd", "zespeed", "120"), ("X Cal", "xcal", "350"),
            ("Y Cal", "ycal", "475"), ("Z Cal", "zcal", "140"),
            ("Pick", "pickup_thresh", "40"), ("Release", "release_thresh", "40"),
            ("Home", "hcc", "10"), ("Y Cor", "ycc", "1"), ("X Cor", "xcc", "0")
        ]
        
        for i, (label, key, default) in enumerate(params):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(params_frame, text=f"{label}:", bg='#1a1a1a', fg='#aaa', 
                    font=('Arial', 8), width=8, anchor=tk.W).grid(row=row, column=col, sticky=tk.W, padx=2, pady=1)
            
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            tk.Entry(params_frame, textvariable=var, width=6, bg='#404040', 
                    fg='#fff', insertbackground='#fff', font=('Arial', 8)).grid(row=row, column=col+1, padx=2, pady=1)
        
        # Parameter sync buttons
        btn_frame = tk.Frame(params_container, bg='#1a1a1a')
        btn_frame.pack(pady=5)
        
        self.upload_params_btn = tk.Button(btn_frame, text="📤 Upload", command=self.upload_params,
                 bg='#4CAF50', fg='white', font=('Arial', 8, 'bold'),
                 padx=10, pady=3, cursor='hand2', relief=tk.FLAT)
        self.upload_params_btn.pack(side=tk.LEFT, padx=2)
        
        self.fetch_params_btn = tk.Button(btn_frame, text="📥 Fetch", command=self.fetch_params,
                 bg='#2196F3', fg='white', font=('Arial', 8, 'bold'),
                 padx=10, pady=3, cursor='hand2', relief=tk.FLAT)
        self.fetch_params_btn.pack(side=tk.LEFT, padx=2)
    
    def _setup_export_tab(self, parent):
        """Setup export & statistics tab"""
        canvas = tk.Canvas(parent, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg='#1e1e1e')
        
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # CSV Export Format Selection
        self._section_label(frame, "EXPORT FORMAT")
        
        format_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=2)
        format_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(format_frame, text="Select export format:", bg='#1a1a1a', fg='#fff', 
                font=('Arial', 9, 'bold')).pack(pady=3, padx=10, anchor=tk.W)
        
        tk.Radiobutton(format_frame, text="TCGTraders (Our Website) - Quantity, SKU",
                      variable=self.export_format, value="TCGTraders", bg='#1a1a1a', fg='#fff',
                      selectcolor='#404040', font=('Arial', 9)).pack(anchor=tk.W, padx=20, pady=2)
        
        tk.Radiobutton(format_frame, text="TCGPlayer (TCGPlayer.com) - Quantity, Name, Set Code, Printing, Condition, Language",
                      variable=self.export_format, value="TCGPlayer", bg='#1a1a1a', fg='#fff',
                      selectcolor='#404040', font=('Arial', 9)).pack(anchor=tk.W, padx=20, pady=2)
        
        tk.Label(format_frame, text="💡 All exports are CSV format only",
                bg='#1a1a1a', fg='#ffaa00', font=('Arial', 8, 'italic')).pack(pady=5)
        
        # Export Buttons
        self._section_label(frame, "EXPORT COLLECTION")
        
        export_btn_frame = tk.Frame(frame, bg='#1e1e1e')
        export_btn_frame.pack(pady=8)
        
        tk.Button(export_btn_frame, text="💾 Export Collection", command=self.export_collection,
                 bg='#2196F3', fg='white', font=('Segoe UI', 9, 'bold'),
                 padx=12, pady=5, cursor='hand2', relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        
        tk.Button(export_btn_frame, text="🗑️ Clear Session", command=self.clear_collection_session,
                 bg='#FF9800', fg='white', font=('Segoe UI', 9, 'bold'),
                 padx=12, pady=5, cursor='hand2', relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        
        tk.Button(export_btn_frame, text="🗑️ Clear Master", command=self.clear_master_collection,
                 bg='#f44336', fg='white', font=('Segoe UI', 9, 'bold'),
                 padx=12, pady=5, cursor='hand2', relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        
        self.export_status_label = tk.Label(frame, text="", bg='#1e1e1e', fg='#4CAF50', font=('Arial', 9))
        self.export_status_label.pack(pady=5)
        
        # Statistics
        self._section_label(frame, "COLLECTION STATISTICS")
        stats_frame = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        stats_frame.pack(fill=tk.BOTH, padx=20, pady=5)
        
        self.stats_labels = {}
        for label in ["Total Cards Saved", "Session Cards", "Unique Cards", "Session Time"]:
            lbl_frame = tk.Frame(stats_frame, bg='#1a1a1a')
            lbl_frame.pack(fill=tk.X, pady=2)
            tk.Label(lbl_frame, text=f"{label}:", bg='#1a1a1a', fg='#888', font=('Arial', 9), width=18, anchor=tk.W).pack(side=tk.LEFT, padx=8)
            val_lbl = tk.Label(lbl_frame, text="0", bg='#1a1a1a', fg='#00ff00', font=('Arial', 9, 'bold'))
            val_lbl.pack(side=tk.LEFT)
            self.stats_labels[label] = val_lbl
        
        # Scan history table (enhanced with more columns)
        self._section_label(frame, "RECENT SCANS")
        table_frame = tk.Frame(frame, bg='#1e1e1e')
        table_frame.pack(fill=tk.BOTH, padx=20, pady=5, expand=True)
        
        columns = ("Time", "Card", "Game", "Set", "Condition", "Language", "Foil", "SKU")
        self.scan_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        # Set column widths
        # Reduce recent scans column widths by ~10% to save horizontal space
        # Reduce recent scans column widths by ~10% to save horizontal space
        col_widths = {"Time": 65, "Card": 146, "Game": 65, "Set": 49, "Condition": 65, "Language": 56, "Foil": 40, "SKU": 97}
        for col in columns:
            self.scan_table.heading(col, text=col)
            self.scan_table.column(col, width=col_widths.get(col, 100))
        
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.scan_table.yview)
        self.scan_table.configure(yscrollcommand=table_scroll.set)
        
        self.scan_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    
    def _setup_qr_tab(self, parent):
        """Setup QR codes for Discord and Website"""
        # Main frame
        frame = tk.Frame(parent, bg='#1e1e1e')
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(frame, text="🎮 Join Our Community", bg='#1e1e1e', fg='#ffffff',
                font=('Segoe UI', 16, 'bold')).pack(pady=(0, 20))
        
        # Container for both QR codes side by side
        qr_container = tk.Frame(frame, bg='#1e1e1e')
        qr_container.pack(expand=True)
        
        # Discord Section (LEFT)
        discord_frame = tk.Frame(qr_container, bg='#2b2b2b', relief=tk.RAISED, borderwidth=2)
        discord_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        tk.Label(discord_frame, text="Discord", bg='#2b2b2b', fg='#7289DA',
                font=('Segoe UI', 13, 'bold')).pack(pady=(10, 8))
        
        # Generate Discord QR code
        discord_qr = qrcode.QRCode(version=1, box_size=8, border=2)
        discord_qr.add_data("https://discord.com/invite/2gNWpV6UjW")
        discord_qr.make(fit=True)
        discord_img = discord_qr.make_image(fill_color="#7289DA", back_color="white")
        
        # Convert to PhotoImage
        discord_img = discord_img.resize((200, 200), Image.Resampling.LANCZOS)
        discord_photo = ImageTk.PhotoImage(discord_img)
        
        qr_label = tk.Label(discord_frame, image=discord_photo, bg='#2b2b2b')
        qr_label.image = discord_photo  # Keep reference
        qr_label.pack(pady=10)
        
        tk.Label(discord_frame, text="Join our server", 
                bg='#2b2b2b', fg='#cccccc', font=('Segoe UI', 9)).pack(pady=(0, 10))
        
        # Website Section (RIGHT)
        website_frame = tk.Frame(qr_container, bg='#2b2b2b', relief=tk.RAISED, borderwidth=2)
        website_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        tk.Label(website_frame, text="TCGTraders.app", bg='#2b2b2b', fg='#4CAF50',
                font=('Segoe UI', 13, 'bold')).pack(pady=(10, 8))
        
        # Generate Website QR code
        website_qr = qrcode.QRCode(version=1, box_size=8, border=2)
        website_qr.add_data("https://www.tcgtraders.app/")
        website_qr.make(fit=True)
        website_img = website_qr.make_image(fill_color="#4CAF50", back_color="white")
        
        # Convert to PhotoImage
        website_img = website_img.resize((200, 200), Image.Resampling.LANCZOS)
        website_photo = ImageTk.PhotoImage(website_img)
        
        qr_label2 = tk.Label(website_frame, image=website_photo, bg='#2b2b2b')
        qr_label2.image = website_photo  # Keep reference
        qr_label2.pack(pady=10)
        
        tk.Label(website_frame, text="Visit our site", 
                bg='#2b2b2b', fg='#cccccc', font=('Segoe UI', 9)).pack(pady=(0, 10))
        
        # Footer
        footer = tk.Label(frame, text="📱 Scan with your phone camera to visit", 
                         bg='#1e1e1e', fg='#888888', font=('Segoe UI', 9, 'italic'))
        footer.pack(pady=(30, 0))

    def _setup_downloads_tab(self, parent):
        """Setup Downloads tab for fetching recognition DBs from server"""
        frame = tk.Frame(parent, bg='#1e1e1e')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # ── top controls row ──────────────────────────────────────────────
        top = tk.Frame(frame, bg='#1e1e1e')
        top.pack(fill=tk.X, pady=(0, 4))

        tk.Label(top, text="Server:", bg='#1e1e1e', fg='#aaa', font=('Arial', 9)).pack(side=tk.LEFT)
        self.download_server_var = tk.StringVar(value="https://www.tcgtraders.app:443")
        tk.Entry(top, textvariable=self.download_server_var, width=44, bg='#404040', fg='#fff',
                 insertbackground='#fff', font=('Arial', 9)).pack(side=tk.LEFT, padx=6)
        self.allow_self_signed_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Bypass SSL", variable=self.allow_self_signed_var,
                       bg='#1e1e1e', fg='#aaa', selectcolor='#2b2b2b', font=('Arial', 8)).pack(side=tk.LEFT, padx=4)
        self.ca_bundle_var = tk.StringVar(value="")
        tk.Button(top, text="↺ Refresh", command=self._refresh_download_list,
                  bg='#2196F3', fg='white', font=('Arial', 9), padx=8).pack(side=tk.RIGHT)

        # ── unified DB row ────────────────────────────────────────────────
        db_row = tk.Frame(frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        db_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(db_row, text="unified_card_database.db", bg='#1a1a1a', fg='#fff',
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=8, pady=4)
        tk.Button(db_row, text="⬇ Download", bg='#4CAF50', fg='white', font=('Arial', 9),
                  command=lambda: self._start_download('unified_card_database.db')).pack(side=tk.RIGHT, padx=8, pady=4)

        # ── scrollable game grid ──────────────────────────────────────────
        pg_container = tk.Frame(frame, bg='#1e1e1e')
        pg_container.pack(fill=tk.BOTH, expand=True)

        pg_canvas = tk.Canvas(pg_container, bg='#1e1e1e', highlightthickness=0)
        pg_scroll = ttk.Scrollbar(pg_container, orient='vertical', command=pg_canvas.yview)
        self.games_frame = tk.Frame(pg_canvas, bg='#1e1e1e')
        self.games_frame.bind('<Configure>', lambda e: pg_canvas.configure(scrollregion=pg_canvas.bbox('all')))
        pg_canvas.create_window((0, 0), window=self.games_frame, anchor='nw')
        pg_canvas.configure(yscrollcommand=pg_scroll.set)
        pg_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pg_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        # mouse-wheel scrolling
        pg_canvas.bind('<Enter>', lambda e: pg_canvas.bind_all('<MouseWheel>',
            lambda ev: pg_canvas.yview_scroll(int(-1 * (ev.delta / 120)), 'units')))
        pg_canvas.bind('<Leave>', lambda e: pg_canvas.unbind_all('<MouseWheel>'))

        # ── status bar ────────────────────────────────────────────────────
        self.download_status_label = tk.Label(frame, text="", bg='#1e1e1e', fg='#4CAF50', font=('Arial', 9))
        self.download_status_label.pack(pady=(4, 0))

        try:
            self._refresh_download_list()
        except Exception:
            pass

    def _start_download(self, filename):
        """Start a download for a given filename (spawns thread)."""
        if not filename:
            return
        # create a small progress window
        try:
            prog_win = tk.Toplevel(self.root)
            prog_win.title(f"Downloading {filename}")
            prog_win.geometry("400x90")
            prog_win.transient(self.root)
            prog_win.grab_set()
            tk.Label(prog_win, text=f"Downloading {filename}...").pack(pady=(8,4))
            pb = ttk.Progressbar(prog_win, orient='horizontal', length=360, mode='determinate')
            pb.pack(padx=12, pady=(0,8))
            status_lbl = tk.Label(prog_win, text="Starting...", anchor='w')
            status_lbl.pack(fill='x', padx=12)
        except Exception:
            prog_win = None
            pb = None
            status_lbl = None

        thr = threading.Thread(target=self._download_file_thread, args=(filename, prog_win, pb, status_lbl), daemon=True)
        thr.start()

    def _refresh_download_list(self):
        """Populate game grid with one button per game (3-column grid)."""
        try:
            games_frame = self.games_frame
        except AttributeError:
            return  # tab not yet set up
        for child in games_frame.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass

        # Build game list ------------------------------------------------
        game_rows = []  # list of (display_name, game_id_str)
        allowed = set(getattr(self, 'allowed_game_ids', set()))

        scanner_games = self.scanner and getattr(self.scanner, 'games', None)
        if scanner_games:
            for game_key, info in sorted(scanner_games.items(), key=lambda x: x[0]):
                display = info.get('display_name') or game_key
                gid = str(info.get('id') or self.HARD_CODE_GAME_IDS.get(display, game_key))
                game_rows.append((display, gid))
        else:
            unified_path = getattr(self, 'unified_db_path', None) or os.path.join(
                os.path.dirname(__file__), 'recognition_data', 'unified_card_database.db')
            if os.path.exists(unified_path):
                try:
                    conn = sqlite3.connect(unified_path)
                    cur = conn.cursor()
                    cur.execute("PRAGMA table_info(games)")
                    cols = [r[1] for r in cur.fetchall()]
                    id_col = next((c for c in ('id', 'game_id', 'game') if c in cols), None)
                    name_col = next((c for c in ('name', 'display_name', 'game_name') if c in cols), None)
                    if id_col and name_col:
                        cur.execute(f"SELECT {id_col}, {name_col} FROM games WHERE load = 1 ORDER BY {name_col}")
                        for row in cur.fetchall():
                            game_rows.append((str(row[1]), str(row[0])))
                    conn.close()
                except Exception:
                    pass

        if allowed:
            game_rows = [(d, g) for d, g in game_rows if g in allowed]

        if not game_rows:
            tk.Label(games_frame, text="Download unified_card_database.db first, then click \u21ba Refresh",
                     bg='#1e1e1e', fg='#888', font=('Arial', 9), wraplength=500).grid(
                         row=0, column=0, columnspan=3, pady=20, padx=10)
            return

        # Render 3-column grid -------------------------------------------
        COLS = 3
        for idx, (display, gid) in enumerate(game_rows):
            row_i, col_i = divmod(idx, COLS)
            # check if already downloaded
            try:
                local = os.path.join(os.path.dirname(__file__), 'recognition_data', f'phash_cards_{gid}.db')
                have = os.path.exists(local)
            except Exception:
                have = False
            btn_bg = '#1a472a' if have else '#1e3a5f'
            btn_fg = '#88ff88' if have else '#ffffff'
            label = f"{display}\n{'downloaded' if have else 'pHash DB'}"
            try:
                btn = tk.Button(
                    games_frame,
                    text=label,
                    bg=btn_bg, fg=btn_fg,
                    font=('Arial', 8),
                    width=20, height=2,
                    wraplength=130,
                    justify=tk.CENTER,
                    relief=tk.RAISED, bd=1,
                    cursor='hand2',
                    command=(lambda fn=f'phash_cards_{gid}.db': self._start_download(fn))
                )
                btn.grid(row=row_i, column=col_i, padx=4, pady=4, sticky='nsew')
            except Exception:
                pass

        for c in range(COLS):
            try:
                games_frame.columnconfigure(c, weight=1)
            except Exception:
                pass

    def _prune_hardcoded_map(self):
        """Remove entries from HARD_CODE_GAME_IDS where games.load = 0 in the unified DB."""
        try:
            _db = os.path.join(os.path.dirname(__file__), 'recognition_data', 'unified_card_database.db')
            if not os.path.exists(_db):
                return
            conn = sqlite3.connect(_db)
            cur = conn.cursor()
            cur.execute("SELECT name FROM games WHERE load = 0")
            rows = cur.fetchall()
            conn.close()
            remove_names = {r[0] for r in rows}
            for name in list(self.HARD_CODE_GAME_IDS.keys()):
                if name in remove_names:
                    del self.HARD_CODE_GAME_IDS[name]
        except Exception:
            # don't fail GUI startup if DB not present or query fails
            return

    def _scan_local_recognition_files(self):
        """Scan local recognition_data directory for available recognition DBs and store numeric ids."""
        base = os.path.join(os.path.dirname(__file__), 'recognition_data')
        ids = set()
        names = set()
        if not os.path.isdir(base):
            self.available_game_ids = set()
            return
        for path in glob.glob(os.path.join(base, '*')):
            name = os.path.basename(path)
            # patterns: phash_cards_{id}.db (pHash-only mode)
            for prefix in ('phash_cards_',):
                if name.startswith(prefix):
                    rest = name[len(prefix):]
                    gid = rest.split('.')[0]
                    ids.add(gid)
                    names.add(gid.lower())
        # store both numeric ids and extracted name tokens (lowercased)
        self.available_game_ids = ids
        self.available_game_names = names

    def _ensure_unified_db_prompt(self):
        """Ensure the unified_card_database.db exists locally; prompt to download if missing."""
        base = os.path.join(os.path.dirname(__file__), 'recognition_data')
        os.makedirs(base, exist_ok=True)
        local = os.path.join(base, 'unified_card_database.db')
        try:
            if os.path.exists(local):
                self._load_allowed_game_ids_from_unified_db(local)
                return
            # ask user to download
            do = messagebox.askyesno("Download DB", "Local unified_card_database.db not found. Download from server now?")
            if not do:
                return
            # start non-blocking download using same UI path so progress is shown
            self._start_download('unified_card_database.db')
        except Exception:
            return

    def _download_unified_db_blocking(self, dest_path):
        """Download unified_card_database.db synchronously into dest_path. Returns True on success."""
        server = getattr(self, 'download_server_var', tk.StringVar(value='https://www.tcgtraders.app:443')).get().rstrip('/')
        # Try multiple endpoints in order: recognition_data, download fallback, root file
        endpoints = [
            'recognition_data/unified_card_database.db',
            'download/unified_card_database.db',
            'unified_card_database.db'
        ]
        # Determine SSL verification behavior
        verify_param = True
        ca_path = (getattr(self, 'ca_bundle_var', tk.StringVar(value='')).get() or '').strip()
        allow_self = bool(getattr(self, 'allow_self_signed_var', tk.BooleanVar(value=False)).get())
        if ca_path:
            if not os.path.exists(ca_path):
                messagebox.showerror("Download Error", f"CA bundle not found: {ca_path}")
                return False
            verify_param = ca_path
        elif allow_self:
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass
            verify_param = False

        tried = []
        try:
            self.download_status_label.config(text=f"Downloading unified DB...")
        except Exception:
            pass
        for ep in endpoints:
            url = urllib.parse.urljoin(server + '/', ep)
            tried.append(url)
            try:
                resp = requests.get(url, stream=True, timeout=120, verify=verify_param)
            except Exception as e:
                # network/SSL error — stop and show error
                try:
                    self.log_error('E701', str(e))
                    messagebox.showerror("Download Error", f"Error contacting {url}: {e}")
                    self.download_status_label.config(text="")
                except Exception:
                    pass
                return False

            if resp.status_code == 200:
                try:
                    with open(dest_path, 'wb') as fh:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                fh.write(chunk)
                    try:
                        self.download_status_label.config(text=f"Saved: {dest_path}")
                        messagebox.showinfo("Download Complete", f"Saved to {dest_path}")
                    except Exception:
                        pass
                    return True
                except Exception as e:
                    try:
                        self.log_error('E704', str(e))
                        messagebox.showerror("Save Error", str(e))
                        self.download_status_label.config(text="")
                    except Exception:
                        pass
                    return False
            # if 404, try next endpoint; otherwise report and stop
            if resp.status_code == 404:
                continue
            else:
                try:
                    self.log_error('E702', f'HTTP {resp.status_code} {url}')
                    messagebox.showerror("Download Failed", f"Server returned {resp.status_code} for {url}")
                    self.download_status_label.config(text="")
                except Exception:
                    pass
                return False

        # All endpoints tried and not found
        try:
            self.log_error('E702', 'all endpoints 404: ' + ', '.join(tried))
            messagebox.showerror("Download Failed", f"File not found on server. Tried:\n" + "\n".join(tried))
            self.download_status_label.config(text="")
        except Exception:
            pass
        return False

    def _load_allowed_game_ids_from_unified_db(self, path=None):
        """Load set of game ids where games.load = 1 from the unified DB at path."""
        candidates = []
        if path:
            candidates.append(path)
        # also try repository root fallback
        candidates.append(os.path.join(os.path.dirname(__file__), 'unified_card_database.db'))
        for p in candidates:
            if not os.path.exists(p):
                continue
            try:
                conn = sqlite3.connect(p)
                cur = conn.cursor()
                # detect column name for id
                cur.execute("PRAGMA table_info(games)")
                cols = [r[1] for r in cur.fetchall()]
                id_col = None
                for name in ('id', 'game_id', 'game'):
                    if name in cols:
                        id_col = name
                        break
                if not id_col:
                    conn.close()
                    continue
                q = f"SELECT {id_col} FROM games WHERE load = 1"
                cur.execute(q)
                rows = cur.fetchall()
                conn.close()
                ids = {str(r[0]) for r in rows}
                self.allowed_game_ids = ids
                # remember the unified DB path for later queries
                try:
                    self.unified_db_path = p
                except Exception:
                    self.unified_db_path = None
                return ids
            except Exception:
                continue
        # no DB found or query failed
        self.allowed_game_ids = set()
        self.unified_db_path = None
        return set()

    # listbox-based download helper removed; use per-file buttons instead

    def _download_file_thread(self, filename, prog_win=None, progress_bar=None, status_label=None):
        """Download file from server and save into recognition_data/. Supports progress updates to provided widgets."""
        server = self.download_server_var.get().rstrip('/')
        endpoints = [
            f"recognition_data/{filename}",
            f"download/{filename}",
            filename
        ]
        try:
            self.root.after(0, lambda: self.download_status_label.config(text=f"Downloading {filename}..."))
            # Determine SSL verification behavior
            verify_param = True
            ca_path = (self.ca_bundle_var.get() or '').strip()
            allow_self = bool(self.allow_self_signed_var.get())
            if ca_path:
                if not os.path.exists(ca_path):
                    self.root.after(0, lambda: messagebox.showerror("Download Error", f"CA bundle not found: {ca_path}"))
                    self.root.after(0, lambda: self.download_status_label.config(text=""))
                    return
                verify_param = ca_path
            elif allow_self:
                try:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                except Exception:
                    pass
                verify_param = False

            # try endpoints in order
            tried = []
            success = False
            dest_path = None
            for ep in endpoints:
                url = urllib.parse.urljoin(server + '/', ep)
                tried.append(url)
                try:
                    r = requests.get(url, stream=True, timeout=60, verify=verify_param)
                except Exception as e:
                    # network/SSL error: report and stop
                    self.root.after(0, lambda err=e: self.log_error('E701', str(err)))
                    self.root.after(0, lambda u=url, err=e: messagebox.showerror("Download Error", f"Error contacting {u}: {err}"))
                    self.root.after(0, lambda: self.download_status_label.config(text=""))
                    success = False
                    break

                if r.status_code == 200:
                    # prepare destination
                    dest_dir = os.path.join(os.path.dirname(__file__), 'recognition_data')
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, filename)
                    total = r.headers.get('content-length')
                    try:
                        total = int(total) if total else None
                    except Exception:
                        total = None

                    downloaded = 0
                    try:
                        with open(dest_path, 'wb') as fh:
                            for chunk in r.iter_content(chunk_size=65536):
                                if chunk:
                                    fh.write(chunk)
                                    downloaded += len(chunk)
                                    if progress_bar is not None and total:
                                        percent = int((downloaded / total) * 100)
                                        try:
                                            self.root.after(0, lambda p=percent: progress_bar.config(value=p))
                                            if status_label is not None:
                                                self.root.after(0, lambda d=downloaded, t=total: status_label.config(text=f"{d}/{t} bytes"))
                                        except Exception:
                                            pass
                        success = True
                    except Exception as e:
                        self.root.after(0, lambda err=e: self.log_error('E704', str(err)))
                        self.root.after(0, lambda err=e: messagebox.showerror("Save Error", str(err)))
                        success = False
                    break
                elif r.status_code == 404:
                    # try next endpoint
                    continue
                else:
                    self.root.after(0, lambda err=r.status_code: self.log_error('E702', f'HTTP {err}'))
                    self.root.after(0, lambda sc=r.status_code, u=url: messagebox.showerror("Download Failed", f"Server returned {sc} for {u}"))
                    success = False
                    break

            if not success and tried:
                # if none succeeded, inform user (unless already shown)
                if not any((t for t in tried if t.endswith(filename) and os.path.exists(os.path.join(os.path.dirname(__file__), 'recognition_data', filename)))):
                    self.root.after(0, lambda: self.log_error('E702', 'all endpoints failed: ' + ', '.join(tried)))
                    self.root.after(0, lambda: messagebox.showerror("Download Failed", f"File not found on server. Tried:\n" + "\n".join(tried)))
                    self.root.after(0, lambda: self.download_status_label.config(text=""))
                    if prog_win:
                        try:
                            prog_win.destroy()
                        except Exception:
                            pass
                    return

            # Only proceed with success handling if download actually succeeded
            if success and dest_path:
                dest_dir = os.path.join(os.path.dirname(__file__), 'recognition_data')
                # After saving, refresh local recognition DB list and UI
                try:
                    # If unified DB downloaded, load allowed game ids from it
                    if filename == 'unified_card_database.db':
                        try:
                            self._load_allowed_game_ids_from_unified_db(dest_path)
                        except Exception:
                            pass
                    # Rescan local recognition files and refresh downloads/dropdowns
                    try:
                        self._scan_local_recognition_files()
                    except Exception:
                        pass
                    try:
                        self.root.after(0, lambda: self._refresh_download_list())
                    except Exception:
                        pass
                    try:
                        self.root.after(0, lambda: self._populate_dropdowns())
                    except Exception:
                        pass
                except Exception:
                    pass

                self.root.after(0, lambda path=dest_path: self.download_status_label.config(text=f"Saved: {path}"))
                self.root.after(0, lambda path=dest_path: messagebox.showinfo("Download Complete", f"Saved to {path}"))
                # finish progress UI
                try:
                    if progress_bar is not None:
                        self.root.after(0, lambda: progress_bar.config(value=100))
                    if status_label is not None and os.path.exists(dest_path):
                        self.root.after(0, lambda path=dest_path: status_label.config(text=f"Saved {os.path.basename(path)}"))
                    if prog_win:
                        try:
                            prog_win.destroy()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            self.root.after(0, lambda err=e: self.log_error('E701', str(err)))
            self.root.after(0, lambda err=e: messagebox.showerror("Download Error", str(err)))
            self.root.after(0, lambda: self.download_status_label.config(text=""))
            if prog_win:
                try:
                    prog_win.destroy()
                except Exception:
                    pass
    
    def _setup_tutorial_tab(self, parent):
        """Setup Tutorial / How-To-Use tab with scrollable content."""
        outer = tk.Frame(parent, bg='#1e1e1e')
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        inner = tk.Frame(canvas, bg='#1e1e1e')

        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _on_mousewheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

        # --- helpers ---
        def h1(text):
            tk.Label(inner, text=text, bg='#1e1e1e', fg='#4CAF50',
                     font=('Segoe UI', 13, 'bold'), anchor='w').pack(fill=tk.X, padx=20, pady=(16, 3))

        def h2(text):
            tk.Label(inner, text=text, bg='#1e1e1e', fg='#2196F3',
                     font=('Segoe UI', 10, 'bold'), anchor='w').pack(fill=tk.X, padx=28, pady=(10, 2))

        def para(text):
            tk.Label(inner, text=text, bg='#1e1e1e', fg='#cccccc',
                     font=('Segoe UI', 9), anchor='w', justify='left',
                     wraplength=600).pack(fill=tk.X, padx=32, pady=2)

        def step(num, text):
            row = tk.Frame(inner, bg='#1e1e1e')
            row.pack(fill=tk.X, padx=32, pady=1)
            tk.Label(row, text=f"{num}.", bg='#1e1e1e', fg='#4CAF50',
                     font=('Segoe UI', 9, 'bold'), width=3, anchor='nw').pack(side=tk.LEFT)
            tk.Label(row, text=text, bg='#1e1e1e', fg='#cccccc',
                     font=('Segoe UI', 9), anchor='w', justify='left',
                     wraplength=560).pack(side=tk.LEFT, fill=tk.X)

        def tip(text):
            f = tk.Frame(inner, bg='#1a3020')
            f.pack(fill=tk.X, padx=20, pady=(4, 2))
            tk.Label(f, text=f"💡  {text}", bg='#1a3020', fg='#a5d6a7',
                     font=('Segoe UI', 9), anchor='w', justify='left',
                     wraplength=580).pack(fill=tk.X, padx=10, pady=6)

        def warn(text):
            f = tk.Frame(inner, bg='#2e2000')
            f.pack(fill=tk.X, padx=20, pady=(4, 2))
            tk.Label(f, text=f"⚠️  {text}", bg='#2e2000', fg='#ffe082',
                     font=('Segoe UI', 9), anchor='w', justify='left',
                     wraplength=580).pack(fill=tk.X, padx=10, pady=6)

        def sep():
            tk.Frame(inner, bg='#333333', height=1).pack(fill=tk.X, padx=20, pady=8)

        # -------------------------------------------------------
        h1("📖  Welcome to TCG Card Scanner")
        para("This application identifies physical trading cards using your webcam, matches them "
             "against a perceptual-hash (pHash) image database, and optionally controls an Arduino-based "
             "sorting machine to physically route cards into labelled bins.")

        sep()

        # PREPARATIONS
        h1("🔧  Preparations — What You Need")
        h2("Required Software & Hardware")
        step(1, "Python 3.9 or later installed on your system.")
        step(2, "All Python dependencies installed — open a terminal in the project folder and run:  "
                "pip install -r requirements.txt")
        step(3, "A USB webcam or compatible camera. Camera index 0 is used by default; "
                "change it in the Scanner tab if you have multiple cameras.")
        step(4, "At least one recognition database downloaded from the ⬇️ Downloads tab "
                "(see First-Time Setup below).")

        h2("Optional Hardware")
        step("A", "An Arduino-based card-sorting machine connected by USB serial. "
                  "Configure it in the 🤖 Arduino tab.")

        tip("You do not need the Arduino to use the scanner — scanning and exporting work entirely without it.")

        sep()

        # FIRST-TIME SETUP
        h1("🚀  First-Time Setup")

        h2("Step 1 — Download the Unified Database")
        step(1, "Open the ⬇️ Downloads tab.")
        step(2, "Check that the Server URL is correct (default: https://www.tcgtraders.app).")
        step(3, "Click 'Download' next to unified_card_database.db. "
                "This file lists every supported game and must be downloaded before individual game databases appear.")
        tip("The unified database only needs to be downloaded once, or whenever the supported game list changes on the server.")

        h2("Step 2 — Download a Game-Specific pHash Database")
        step(1, "After the unified DB finishes, the Downloads tab populates with a button for each supported game.")
        step(2, "Find your game (e.g. Magic: The Gathering, Pokémon, Yu-Gi-Oh!) and click its 'pHash DB' button.")
        step(3, "A progress window will appear. Larger games (e.g. MTG with 30 000+ cards) may take a few minutes.")
        step(4, "Once complete the database is saved to the  recognition_data/  folder next to this script and "
                "is available immediately — no restart required.")
        warn("If no games appear after downloading the unified DB, click 'Refresh List' on the Downloads tab.")

        sep()

        # SCANNING
        h1("📷  Scanning Cards")

        h2("Configure the Scanner Tab")
        step(1, "Game — Select the card game you are scanning from the dropdown.")
        step(2, "Match Backend — 'pHash RGB' is the default and most accurate. "
                "'pHash Grayscale' is slightly faster but less precise.")
        step(3, "Condition — Set the default card condition reported on export (NM, LP, MP, HP, DMG).")
        step(4, "Language — Select the language printed on the cards.")
        step(5, "Camera Index — Usually 0. Try 1 or 2 if your card camera is not the system default.")
        step(6, "Auto-Save — When checked, matched cards are logged automatically without a confirmation prompt.")

        h2("Start a Scan Session")
        step(1, "Click ▶ Start in the camera panel on the left side of the window.")
        step(2, "Place a card flat within the camera's view. The scanner detects the card outline automatically.")
        step(3, "When a match is found, a result appears showing the card name, set, rarity, and bin assignment.")

        tip("Even, diffuse lighting with no glare on the card surface gives the best results.")
        tip("Hold the card still for a moment — the scanner waits for a stable contour before running a match.")
        warn("If cards are consistently wrong, try re-downloading the pHash DB. "
             "The server database may have been updated since your last download.")

        sep()

        # ARDUINO
        h1("🤖  Arduino Sorter (Optional)")
        para("The Arduino tab controls a physical card-sorting machine that routes each identified card "
             "to a numbered bin based on game, set, rarity, foil, and/or price rules.")
        step(1, "Connect your Arduino sorter via USB, then select its COM port from the Port dropdown.")
        step(2, "Click 'Connect'. The status indicator turns green when the serial link is established.")
        step(3, "Bin assignments are determined automatically from the card's attributes and sent to the Arduino.")
        step(4, "Use the manual jog buttons (X/Y axis) in the Arduino tab to calibrate the home position.")
        tip("If the COM port is not listed, click the refresh icon next to the Port dropdown — the device may have connected after the program started.")

        sep()

        # EXPORT
        h1("📊  Exporting Your Collection")
        step(1, "Open the 📊 Export tab.")
        step(2, "Choose an export format: TCGTraders (for tcgtraders.app) or TCGPlayer CSV.")
        step(3, "Click 'Export' and choose a save location. A CSV file of your scanned inventory is written.")
        step(4, "The scan history table shows every card accepted in the current session.")
        tip("Exports include all cards accepted since the app was launched. Restart the app to begin a fresh session.")

        sep()

        # TROUBLESHOOTING
        h1("❓  Troubleshooting")
        step("Q", "Camera shows a black screen — make sure no other application is using the camera and the correct Camera Index is selected.")
        step("Q", "No card detected — ensure the card is well-lit, flat, and fully visible against a contrasting background.")
        step("Q", "Match accuracy is low — verify you have downloaded the correct game's pHash DB and selected the right game in the Scanner tab.")
        step("Q", "Download fails with SSL error — check the server URL for typos, or enable 'Bypass SSL' on the Downloads tab (only on a trusted local network).")
        step("Q", "'No games loaded' — download unified_card_database.db from the Downloads tab first.")
        step("Q", "Arduino not responding — confirm the COM port, verify the correct firmware is flashed, and ensure no other program (e.g. Arduino IDE Serial Monitor) has the port open.")

        # bottom padding
        tk.Frame(inner, bg='#1e1e1e', height=20).pack()

    # =========================================================================
    # TROUBLESHOOTING TAB
    # =========================================================================

    def _setup_troubleshooting_tab(self, parent):
        """Setup Troubleshooting tab: live error log + full error code reference."""
        outer = tk.Frame(parent, bg='#1e1e1e')
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Top: Recent Error Log ─────────────────────────────────────────
        log_frame = tk.Frame(outer, bg='#1a1a1a', relief=tk.SUNKEN, bd=1)
        log_frame.pack(fill=tk.X, padx=12, pady=(10, 4))

        hdr = tk.Frame(log_frame, bg='#1a1a1a')
        hdr.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(hdr, text="Recent Errors", bg='#1a1a1a', fg='#ff6666',
                 font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self._error_count_lbl = tk.Label(hdr, textvariable=self._error_badge_var,
                                          bg='#3a0000', fg='#ff9999',
                                          font=('Segoe UI', 9, 'bold'), padx=6)
        self._error_count_lbl.pack(side=tk.LEFT, padx=6)
        tk.Button(hdr, text="Clear", command=self._clear_error_log,
                  bg='#444', fg='#fff', font=('Arial', 8), padx=6, pady=2,
                  relief=tk.FLAT, cursor='hand2').pack(side=tk.RIGHT, padx=4)
        tk.Button(hdr, text="Copy", command=self._copy_error_log,
                  bg='#2196F3', fg='#fff', font=('Arial', 8), padx=6, pady=2,
                  relief=tk.FLAT, cursor='hand2').pack(side=tk.RIGHT, padx=4)

        err_cols = ("Time", "Code", "Title", "Detail", "Fix")
        self._err_table = ttk.Treeview(log_frame, columns=err_cols,
                                        show='headings', height=6)
        err_widths = {"Time": 70, "Code": 52, "Title": 180, "Detail": 260, "Fix": 280}
        for c in err_cols:
            self._err_table.heading(c, text=c)
            self._err_table.column(c, width=err_widths.get(c, 100), stretch=False)
        # Tag colours
        self._err_table.tag_configure('error',   foreground='#ff6666')
        self._err_table.tag_configure('warning', foreground='#ffcc44')

        err_vsb = ttk.Scrollbar(log_frame, orient='vertical',
                                 command=self._err_table.yview)
        err_hsb = ttk.Scrollbar(log_frame, orient='horizontal',
                                 command=self._err_table.xview)
        self._err_table.configure(yscrollcommand=err_vsb.set,
                                   xscrollcommand=err_hsb.set)
        self._err_table.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        err_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        err_hsb.pack(side=tk.BOTTOM, fill=tk.X, padx=8)

        # ── Bottom: Error Code Reference ──────────────────────────────────
        ref_outer = tk.Frame(outer, bg='#1e1e1e')
        ref_outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 10))

        ref_hdr = tk.Frame(ref_outer, bg='#1e1e1e')
        ref_hdr.pack(fill=tk.X, pady=(4, 2))
        tk.Label(ref_hdr, text="Error Code Reference", bg='#1e1e1e', fg='#4CAF50',
                 font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)

        # Search bar
        tk.Label(ref_hdr, text="Search:", bg='#1e1e1e', fg='#aaa',
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=(20, 4))
        self._ref_search_var = tk.StringVar()
        self._ref_search_var.trace_add('write', lambda *_: self._filter_ref_table())
        tk.Entry(ref_hdr, textvariable=self._ref_search_var, width=26,
                 bg='#404040', fg='#fff', insertbackground='#fff',
                 font=('Arial', 9)).pack(side=tk.LEFT)

        ref_cols = ("Code", "Title", "Description", "Fix")
        self._ref_table = ttk.Treeview(ref_outer, columns=ref_cols,
                                        show='headings', height=16)
        ref_widths = {"Code": 52, "Title": 200, "Description": 310, "Fix": 300}
        for c in ref_cols:
            self._ref_table.heading(c, text=c)
            self._ref_table.column(c, width=ref_widths.get(c, 120), stretch=False)
        self._ref_table.tag_configure('error',   foreground='#ff9999')
        self._ref_table.tag_configure('warning', foreground='#ffcc44')
        self._ref_table.tag_configure('cat',     foreground='#4CAF50',
                                       background='#1a2a1a', font=('Segoe UI', 9, 'bold'))

        ref_vsb = ttk.Scrollbar(ref_outer, orient='vertical',
                                 command=self._ref_table.yview)
        ref_hsb = ttk.Scrollbar(ref_outer, orient='horizontal',
                                 command=self._ref_table.xview)
        self._ref_table.configure(yscrollcommand=ref_vsb.set,
                                   xscrollcommand=ref_hsb.set)
        self._ref_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ref_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        ref_hsb.pack(side=tk.BOTTOM, fill=tk.X)

        self._populate_ref_table(ERROR_CODES)

    def _populate_ref_table(self, codes: dict, search: str = ""):
        """Fill the reference table, with optional search filter."""
        for row in self._ref_table.get_children():
            self._ref_table.delete(row)

        categories = {
            "E1": "── Database & Initialisation ──",
            "E2": "── Camera & Vision ──",
            "E3": "── Recognition / Matching ──",
            "E4": "── Serial / Arduino Communication ──",
            "E5": "── Arduino Hardware ──",
            "E6": "── Collection & Export ──",
            "E7": "── Network / Download ──",
        }

        s = search.lower()
        last_cat = None
        for code, entry in sorted(codes.items()):
            # Filter
            if s and s not in code.lower() \
                  and s not in entry['title'].lower() \
                  and s not in entry['description'].lower() \
                  and s not in entry['fix'].lower():
                continue
            # Category header row
            prefix = code[:2]
            if prefix != last_cat and prefix in categories:
                last_cat = prefix
                self._ref_table.insert('', 'end',
                    values=(prefix + "xx", categories[prefix], "", ""),
                    tags=('cat',))
            sev = entry.get('severity', 'error')
            self._ref_table.insert('', 'end',
                values=(code, entry['title'], entry['description'], entry['fix']),
                tags=(sev,))

    def _filter_ref_table(self):
        self._populate_ref_table(ERROR_CODES, self._ref_search_var.get())

    def _clear_error_log(self):
        self.error_log.clear()
        for row in self._err_table.get_children():
            self._err_table.delete(row)
        self._error_badge_var.set("")

    def _copy_error_log(self):
        """Copy recent error log to clipboard."""
        lines = ["\t".join(map(str, e)) for e in self.error_log]
        text = "\n".join(lines)
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except Exception:
            pass

    def log_error(self, code: str, detail: str = ""):
        """Log a structured error — updates the status log AND the Troubleshooting tab."""
        entry = _ec_lookup(code)
        title  = entry.get('title', 'Unknown Error')
        fix    = entry.get('fix', '')
        sev    = entry.get('severity', 'error')
        ts     = time.strftime("%H:%M:%S")

        # Append to in-memory log (keep last 200)
        self.error_log.append((ts, code, title, detail, fix))
        if len(self.error_log) > 200:
            self.error_log.pop(0)

        # Update badge
        count = len(self.error_log)
        self._error_badge_var.set(f"{count} error{'s' if count != 1 else ''}")

        # Update table on main thread
        def _insert():
            try:
                self._err_table.insert('', 0,
                    values=(ts, code, title, detail[:80], fix[:80]),
                    tags=(sev,))
                # Prune table to 200 rows
                children = self._err_table.get_children()
                if len(children) > 200:
                    self._err_table.delete(children[-1])
            except Exception:
                pass
        try:
            self.root.after(0, _insert)
        except Exception:
            pass

        # Also push to the status log with code prefix
        msg = f"[{code}] {title}"
        if detail:
            msg += f": {detail}"
        self.log_status(msg, error=True)

    # =========================================================================

    def _section_label(self, parent, text):
        """Create section header"""
        tk.Label(parent, text=text, bg='#1e1e1e', fg='#4CAF50',
                font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, padx=20, pady=(10, 5))
    
    def _init_scanner(self):
        """Initialize scanner with collection settings"""
        try:
            backend = self.vector_backend.get()
            use_grayscale_phash = backend == "PHASH_GRAY"
            # Always point the scanner at the copy in recognition_data/ so it
            # never accidentally opens or creates a blank DB in the cwd.
            _recog_dir = os.path.join(os.path.dirname(__file__), 'recognition_data')
            _db_path = os.path.join(_recog_dir, 'unified_card_database.db')
            self.scanner = OptimizedCardScanner(
                db_path=_db_path,
                max_workers=8, 
                cache_enabled=True,
                use_grayscale_phash=use_grayscale_phash,
                auto_vector_when_unfiltered=False,
                enable_collection=self.collection_enabled,
                default_condition=self.default_condition.get(),
                default_language=self.default_language.get(),
                default_foil=self.default_foil.get(),
                prompt_for_details=not self.auto_save_mode.get()
            )
            self.log_status(f"Scanner initialized: {len(self.scanner.games)} games")
            # Apply initial match settings from GUI
            try:
                self.scanner.scan_threshold = int(self.match_threshold_var.get())
            except Exception:
                pass
            self.update_vector_backend()
            self.log_status(f"Collection mode: {'Auto-save' if self.auto_save_mode.get() else 'Individual prompts'}")
            self.log_status(f"Defaults: {self.default_condition.get()}, {self.default_language.get()}, {'Foil' if self.default_foil.get() else 'Normal'}")
            self._populate_dropdowns()
            # Refresh downloads now that scanner/games are loaded
            try:
                self._refresh_download_list()
            except Exception:
                pass
            self.session_start = time.time()
            self._update_stats()
        except Exception as e:
            self.log_error('E101', str(e))
            messagebox.showerror("Scanner Error", f"Failed to initialize:\n{e}")
    
    def _populate_dropdowns(self):
        """Populate dropdowns"""
        if not self.scanner:
            return
        # Refresh local recognition files and only include games that have local recognition DBs
        self._scan_local_recognition_files()
        available_ids = set(getattr(self, 'available_game_ids', set()))
        available_names = set(getattr(self, 'available_game_names', set()))
        # only include games that are allowed by the unified DB (games.load = 1)
        allowed = set(getattr(self, 'allowed_game_ids', set()))
        # map display label -> scanner.games key for selection resolution
        self._display_to_game_key = {}
        game_labels = []
        for game_key, info in self.scanner.games.items():
            display = info.get('display_name') or game_key
            gid = str(info.get('id') or self.HARD_CODE_GAME_IDS.get(display, ''))
            display_token = (display or '').lower()
            key_token = str(game_key).lower()
            # Require that the game is allowed by the unified DB
            if allowed:
                if gid not in allowed:
                    continue
            # If we don't have an allowed list, fall back to including games
            # (still require a local recognition DB to be present; checked below)
            # Must also have a local recognition DB file present
            if not (gid in available_ids or display_token in available_names or key_token in available_names):
                continue
            game_labels.append(display)
            self._display_to_game_key[display] = game_key

        games = ["All Games"] + sorted(game_labels)
        self.game_combo['values'] = games
        # debug: log first few games present in combobox
        try:
            self.log_status(f"Combo games sample: {games[:10]}")
            self.log_status(f"Display->key map size: {len(self._display_to_game_key)}")
        except Exception:
            pass
        # ensure current selection is valid
        if self.game_var.get() not in games:
            self.game_var.set("All Games")
        # If a specific game is already selected, load its sets/rarities/foils
        try:
            cur = self.game_var.get()
            if cur and cur != "All Games":
                key = getattr(self, '_display_to_game_key', {}).get(cur)
                if key:
                    self._load_game_data(key)
        except Exception:
            pass
        self.set_combo['values'] = ["All Sets"]
        self.rarity_combo['values'] = ["All Rarities"]
        self.foil_combo['values'] = ["All Foil Types"]
        self.log_status(f"Loaded {len(games)-1} games")
    
    def on_game_change(self, event=None):
        """Handle game selection"""
        selected = self.game_var.get()
        self.log_status(f"Dropdown changed -> selected: {selected}")
        # refresh local file list in case something changed
        self._scan_local_recognition_files()
        if selected == "All Games":
            # Build allowed list from unified DB and local files
            allowed_ids = set(getattr(self, 'allowed_game_ids', set()))
            local_ids = set(getattr(self, 'available_game_ids', set()))
            allowed = []
            if allowed_ids:
                for key, info in self.scanner.games.items():
                    gid = str(info.get('id') or self.HARD_CODE_GAME_IDS.get(info.get('display_name') or key, ''))
                    display_token = (info.get('display_name') or key).lower()
                    if gid in allowed_ids and (gid in local_ids or display_token in set(getattr(self, 'available_game_names', set()))):
                        allowed.append(key)
            # set active games
            try:
                self.scanner.set_active_games(allowed)
            except Exception:
                self.scanner.active_games = allowed
            self.log_status(f"Game filter: All (restricted to {len(allowed)} local games)")
        else:
            # resolve display label to internal game key
            game_key = getattr(self, '_display_to_game_key', {}).get(selected)
            self.log_status(f"Resolved display->key mapping: {selected} -> {game_key}")
            if game_key:
                try:
                    self.scanner.set_active_games([game_key])
                except Exception:
                    self.scanner.active_games = [game_key]
                self.log_status(f"Game filter: {selected}")
                self._load_game_data(game_key)
                return
            # fallback: try to match by display_name or key name
            for k, info in self.scanner.games.items():
                display = info.get('display_name') or k
                if display == selected or k == selected:
                    try:
                        self.scanner.set_active_games([k])
                    except Exception:
                        self.scanner.active_games = [k]
                    self.log_status(f"Game filter: {selected}")
                    self._load_game_data(k)
                    return
    
    def _load_game_data(self, game_name):
        """Load sets/rarities/foils for game"""
        try:
            # Resolve game_name (may be display label or internal key)
            game_key = None
            if game_name in self.scanner.games:
                game_key = game_name
            else:
                # try mapped display -> key
                game_key = getattr(self, '_display_to_game_key', {}).get(game_name)
                if not game_key:
                    for k, info in self.scanner.games.items():
                        if (info.get('display_name') or k) == game_name:
                            game_key = k
                            break
            if not self.scanner:
                self.log_error('E106', 'game_data load: scanner is None')
                self.log_status(f"Game '{game_name}' not found in database", error=True)
                return

            # Resolve numeric game id
            gid = str(self.scanner.games[game_key]['id'])
            # First try to load sets from unified DB if available
            unified_path = getattr(self, 'unified_db_path', None)
            sets = []
            if unified_path and os.path.exists(unified_path):
                try:
                    uconn = sqlite3.connect(unified_path)
                    ucur = uconn.cursor()
                    ucur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sets'")
                    if ucur.fetchone():
                        sets = ucur.execute("SELECT DISTINCT code, name FROM sets WHERE game = ? ORDER BY name", (gid,)).fetchall()
                    uconn.close()
                except Exception:
                    sets = []
            # fallback: load set codes from per-game cards table (cards_{id}.set_code)
            if not sets:
                try:
                    cursor = self.scanner.get_connection()
                    table = self.scanner.games[game_key]['table']
                    try:
                        rows = cursor.execute(f"SELECT DISTINCT set_code FROM {table} WHERE set_code IS NOT NULL ORDER BY set_code").fetchall()
                        sets = [(r[0], r[0]) for r in rows]
                    except Exception:
                        sets = []
                except Exception:
                    sets = []

            self.set_combo['values'] = ["All Sets"] + [f"{c} - {n}" if c and n and c != n else (c or n) for c, n in sets]
            self.set_var.set("All Sets")

            # Rarities: prefer unified DB 'cards' table if available, otherwise use per-game table
            rarities = []
            if unified_path and os.path.exists(unified_path):
                try:
                    uconn = sqlite3.connect(unified_path)
                    ucur = uconn.cursor()
                    ucur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cards'")
                    if ucur.fetchone():
                        ucur.execute("PRAGMA table_info(cards)")
                        cols = [r[1] for r in ucur.fetchall()]
                        if 'rarity' in cols:
                            rarities = ucur.execute("SELECT DISTINCT rarity FROM cards WHERE game = ? AND rarity IS NOT NULL ORDER BY rarity", (gid,)).fetchall()
                    uconn.close()
                except Exception:
                    rarities = []

            if not rarities:
                try:
                    cursor = self.scanner.get_connection()
                    table = self.scanner.games[game_key]['table']
                    rarities = cursor.execute(f"SELECT DISTINCT rarity FROM {table} WHERE rarity IS NOT NULL ORDER BY rarity").fetchall()
                except Exception:
                    rarities = []

            self.rarity_combo['values'] = ["All Rarities"] + [r[0] for r in rarities]
            self.rarity_var.set("All Rarities")

            # Foil: hardcode simple options
            self.foil_combo['values'] = ["All Foil Types", "Foil", "Normal"]
            self.foil_var.set("All Foil Types")
        except Exception as e:
            self.log_status(f"Error loading game data: {e}", error=True)
            self.log_error('E105', str(e))

    def _poll_game_var(self):
        try:
            cur = self.game_var.get()
            if getattr(self, '_last_game_var', None) != cur:
                self._last_game_var = cur
                try:
                    self.log_status(f"[poll] detected game change -> {cur}")
                except Exception:
                    pass
                try:
                    self.on_game_change()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.root.after(500, self._poll_game_var)
        except Exception:
            pass
    
    def toggle_auto_scan(self):
        """Toggle auto-scan mode"""
        self.auto_scan = self.auto_scan_var.get()
        mode_text = "Auto-Scan Mode: ON" if self.auto_scan else "Manual Mode: ON"
        self.scan_mode_label.config(text=mode_text, fg='#4CAF50' if self.auto_scan else '#2196F3')
        self.log_status(f"Scan mode: {mode_text}")
    
    def toggle_inventory(self):
        """Toggle inventory tracking"""
        if self.scanner:
            enabled = self.inventory_var.get()
            self.scanner.enable_inventory_tracking(enabled)
            self.log_status(f"Inventory tracking {'ON' if enabled else 'OFF'}")
    
    def update_vector_backend(self):
        """Update match backend (pHash only)"""
        backend = self.vector_backend.get()

        if backend == "PHASH_RGB":
            self.backend_status.config(text="✅ pHash (RGB channels)", fg='#00ff00')
            self.log_status("[Match] Using pHash (RGB channels)")
            if self.scanner:
                self.scanner.use_resnet50 = False
                self.scanner.use_grayscale_phash = False
                self.scanner.use_vector = False
                self.scanner.auto_vector_when_unfiltered = False
                self.scanner.vector_searcher = None

        elif backend == "PHASH_GRAY":
            self.backend_status.config(text="✅ pHash (Grayscale)", fg='#00ff00')
            self.log_status("[Match] Using pHash (Grayscale)")
            if self.scanner:
                self.scanner.use_grayscale_phash = True
                self.scanner.use_vector = False
                self.scanner.auto_vector_when_unfiltered = False
                self.scanner.vector_searcher = None
        else:
            # Fallback to RGB pHash if an unknown backend value is present
            try:
                self.vector_backend.set("PHASH_RGB")
            except Exception:
                pass
            self.backend_status.config(text="✅ pHash (RGB channels)", fg='#00ff00')
            if self.scanner:
                self.scanner.use_resnet50 = False
                self.scanner.use_grayscale_phash = False
                self.scanner.use_vector = False
                self.scanner.auto_vector_when_unfiltered = False
                self.scanner.vector_searcher = None
    
    def connect_arduino(self):
        """Connect to Arduino"""
        if not self.scanner:
            self.log_status("Scanner not initialized", error=True)
            return

        port = (self.serial_port_var.get() if hasattr(self, 'serial_port_var') else '').strip()
        baud_raw = (self.baud_var.get() if hasattr(self, 'baud_var') else '').strip()

        if not port:
            self.log_status("Please enter a serial port (e.g., COM3)", error=True)
            messagebox.showwarning("Arduino", "Please enter a serial port (e.g., COM3)")
            return

        try:
            baud = int(baud_raw or '9600')
        except Exception:
            baud = 9600

        # Close any existing connection
        try:
            if getattr(self.scanner, 'ser', None):
                try:
                    self.scanner.ser.close()
                except Exception:
                    pass
                self.scanner.ser = None
        except Exception:
            pass

        self.scanner.serial_port = port
        self.scanner.baud_rate = baud
        self.log_status(f"Connecting to Arduino on {port} @ {baud}...")

        # Disable button while connecting to prevent double-clicks
        try:
            self.connect_btn.configure(text="Connecting…", state=tk.DISABLED, bg='#FF9800')
        except Exception:
            pass

        def _do_connect():
            ok = False
            try:
                if getattr(self, 'arduino_plugin', None):
                    try:
                        ok = bool(self.arduino_plugin.connect(port, baud))
                    except Exception:
                        ok = False

                if not ok:
                    try:
                        ok = bool(self.scanner.init_serial())
                    except Exception as e:
                        self.root.after(0, lambda err=e: self.log_error('E405', str(err)))
                        ok = False
            except Exception as e:
                self.root.after(0, lambda err=e: self.log_error('E405', str(err)))
                ok = False

            def _on_done(success):
                if success:
                    self.log_status("✓ Arduino connected")
                    try:
                        self.connect_btn.configure(text="Connected", state=tk.DISABLED, bg='#4CAF50')
                    except Exception:
                        pass
                    self.machine_started = False
                    self.machine_state_label.configure(text="● STOPPED", fg='#ff4444')
                    self.start_machine_btn.configure(state=tk.NORMAL)
                    self.stop_machine_btn.configure(state=tk.DISABLED)
                    self._update_controls_state()
                else:
                    self.log_error('E403')
                    try:
                        self.connect_btn.configure(text="Connect", state=tk.NORMAL, bg='#FF9800')
                    except Exception:
                        pass
                    messagebox.showerror(
                        "Arduino",
                        "Failed to connect to Arduino.\n\n"
                        "- Verify the COM port\n"
                        "- Verify baud rate\n"
                        "- Ensure Arduino is plugged in\n"
                        "- Ensure pyserial is installed"
                    )

            self.root.after(0, lambda: _on_done(ok))

        threading.Thread(target=_do_connect, daemon=True).start()

    def start_machine(self):
        """Start the machine - disable manual controls"""
        if not self.scanner or not getattr(self.scanner, 'ser', None):
            self.log_status("Arduino not connected", error=True)
            return

        response = self._send_arduino("StartMachine")
        if response and "OK" in response:
            self.machine_started = True
            self.machine_state_label.configure(text="● STARTED", fg='#44ff44')
            self.start_machine_btn.configure(state=tk.DISABLED)
            self.stop_machine_btn.configure(state=tk.NORMAL)
            self._update_controls_state()
            self.log_status("✓ Machine STARTED - Manual controls disabled")
        else:
            self.log_status("✗ Failed to start machine", error=True)
            self.log_error('E406', 'StartMachine no OK')
            messagebox.showerror("Error", "Failed to start machine")

    def arduino_home(self):
        """Home the machine (manual control)"""
        if self.machine_started:
            self.log_status("⚠ Cannot home while machine is started", error=True)
            messagebox.showwarning("Machine Started", "Stop the machine first to use Home")
            return

        if not self.scanner or not getattr(self.scanner, 'ser', None):
            self.log_status("Arduino not connected", error=True)
            return

        response = self._send_arduino("HomeButton")
        self.log_status("→ HomeButton")
        if response:
            self.log_status(f"← {response}")
        else:
            self.log_status("No response from Arduino", error=True)

    
    def stop_machine(self):
        """Stop the machine - enable manual controls"""
        if not self.scanner or not self.scanner.ser:
            self.log_status("Arduino not connected", error=True)
            return
        
        response = self._send_arduino("StopMachine")
        if response and "OK" in response:
            self.machine_started = False
            self.machine_state_label.configure(text="● STOPPED", fg='#ff4444')
            self.start_machine_btn.configure(state=tk.NORMAL)
            self.stop_machine_btn.configure(state=tk.DISABLED)
            self._update_controls_state()
            self.log_status("✓ Machine STOPPED - Manual controls enabled")
        else:
            self.log_status("✗ Failed to stop machine", error=True)
            self.log_error('E406', 'StopMachine no OK')
            messagebox.showerror("Error", "Failed to stop machine")
    
    def _update_controls_state(self):
        """Enable/disable controls based on machine state"""
        # When Started: disable all manual controls
        # When Stopped: enable all manual controls
        state = tk.DISABLED if self.machine_started else tk.NORMAL
        
        # Disable/enable manual controls
        self.home_btn.configure(state=state)
        
        for checkbox in self.motor_checkboxes.values():
            checkbox.configure(state=state)
        
        for button in self.movement_buttons:
            button.configure(state=state)
        
        self.upload_params_btn.configure(state=state)
        self.fetch_params_btn.configure(state=state)
    
    def toggle_motor(self, name):
        """Toggle motor/relay"""
        if self.machine_started:
            self.log_status("⚠ Cannot control motors while machine is started", error=True)
            self.log_error('E508', f'toggle_motor:{name}')
            # Revert checkbox
            self.motor_vars[name].set(not self.motor_vars[name].get())
            return
        
        if not self.scanner or not self.scanner.ser:
            self.log_status("Arduino not connected", error=True)
            return
        
        state = 1 if self.motor_vars[name].get() else 0
        response = self._send_arduino(f"SetMotor,{name},{state}")
        
        if response and "OK" in response:
            self.log_status(f"✓ {name}: {'ON' if state else 'OFF'}")
        elif response and "MachineStarted" in response:
            self.log_status(f"⚠ Machine is started - stop first", error=True)
            self.motor_vars[name].set(not state)
        else:
            self.log_status(f"✗ {name} control failed", error=True)
            # Revert checkbox if failed
            self.motor_vars[name].set(not state)
    
    def send_arduino_command(self, cmd):
        """Send command to Arduino"""
        # Check if manual control command while machine is started
        # Commands may carry a step suffix e.g. "CalibrateX1,10" — compare base name only
        _cmd_base = cmd.split(',')[0]
        if self.machine_started and _cmd_base in ["HomeButton", "CalibrateX1", "CalibrateX2", "CalibrateY1", "CalibrateY2"]:
            self.log_status("⚠ Cannot send manual commands while machine is started", error=True)
            self.log_error('E508', f'cmd={_cmd_base}')
            messagebox.showwarning("Machine Started", "Stop the machine first to use manual controls")
            return
        
        if (getattr(self, 'arduino_plugin', None) and self.arduino_plugin.is_connected()) or (self.scanner and getattr(self.scanner, 'ser', None)):
            response = self._send_arduino(cmd)
            self.log_status(f"→ {cmd}")
            if response:
                self._handle_arduino_response(response, cmd)
                if "MachineStarted" in response:
                    messagebox.showwarning("Machine Started", "Machine must be STOPPED for this command")
        else:
            self.log_status("Arduino not connected", error=True)
            self.log_error('E402', 'send_arduino_command')

    def _handle_arduino_response(self, response: str, context: str = ""):
        """Parse an Arduino response; log error codes if the response contains one."""
        if not response:
            return
        self.log_status(f"  \u2190 Arduino: {response}")
        # New structured format: <Error,EXXX,detail>
        if response.startswith("Error,E"):
            parts = response.split(",", 2)
            code = parts[1] if len(parts) > 1 else "E500"
            detail = parts[2] if len(parts) > 2 else context
            self.log_error(code, detail)
        # Legacy flat tags
        elif "MachineStarted" in response:
            self.log_error("E508", context or "command rejected")
        elif "MachineNotStarted" in response:
            self.log_error("E507", context or "tray command sent while stopped")
        elif "NotAtHome" in response:
            self.log_error("E509", context or "params sent while not at home")

    def _send_arduino(self, cmd):
        """Route command to Arduino plugin if available, else fallback to scanner serial."""
        try:
            if getattr(self, 'arduino_plugin', None) and self.arduino_plugin.is_connected():
                try:
                    return self.arduino_plugin.send(cmd)
                except Exception:
                    return None

            if self.scanner and getattr(self.scanner, 'ser', None):
                try:
                    return self.scanner.send_to_arduino(cmd)
                except Exception:
                    return None
        except Exception:
            return None
        return None
    
    def toggle_arduino_monitoring(self):
        """Start/stop Arduino monitoring"""
        if self.monitor_var.get():
            if not (self.scanner and getattr(self.scanner, 'ser', None)):
                # Allow plugin-based Arduino connections
                if not (getattr(self, 'arduino_plugin', None) and self.arduino_plugin.is_connected()):
                    self.log_status("Connect Arduino first", error=True)
                    self.monitor_var.set(False)
                    return
            self.arduino_monitoring = True
            self.arduino_monitor_thread = threading.Thread(target=self._arduino_monitor_loop, daemon=True)
            self.arduino_monitor_thread.start()
            self.log_status("Arduino monitoring started")
        else:
            self.arduino_monitoring = False
            self.log_status("Arduino monitoring stopped")
    
    def upload_params(self):
        """Upload all parameters to Arduino"""
        if not self.scanner or not self.scanner.ser:
            self.log_status("Arduino not connected", error=True)
            self.log_error('E402', 'upload_params')
            return
        
        if self.machine_started:
            self.log_status("⚠ Cannot upload params while machine is started", error=True)
            self.log_error('E508', 'upload_params')
            messagebox.showwarning("Machine Started", 
                "Machine must be STOPPED to upload parameters.\n\nClick 'Stop Machine' first.")
            return
        
        param_mapping = {
            "speed": "speed",
            "zspeed": "zspeed", 
            "zespeed": "zespeed",
            "xcal": "xcal",
            "ycal": "ycal",
            "zcal": "zcal",
            "pickup_thresh": "pickup_thresh",
            "release_thresh": "release_thresh",
            "hcc": "hcc",
            "ycc": "ycc",
            "xcc": "xcc"
        }
        
        success_count = 0
        for gui_key, arduino_key in param_mapping.items():
            try:
                value = self.param_vars[gui_key].get()
                response = self._send_arduino(f"SetParam,{arduino_key},{value}")
                
                if response and "OK" in response:
                    success_count += 1
                elif response and "MachineStarted" in response:
                    self.log_status("⚠ Machine is started - stop first", error=True)
                    self.log_error('E508', f'upload_params:{gui_key}')
                    messagebox.showwarning("Machine Started",
                        "Machine must be STOPPED to upload parameters.\n\nClick 'Stop Machine' first.")
                    return
                elif response and "NotAtHome" in response:
                    self.log_status("⚠ Arduino not at home position - params rejected", error=True)
                    self.log_error('E509', f'upload_params:{gui_key}')
                    messagebox.showwarning("Not At Home",
                        "Arduino must be at home position to accept parameter changes.\n\nClick 'Home Machine' first.")
                    return
                else:
                    self.log_status(f"✗ Failed to set {gui_key}", error=True)
                    self.log_error('E406', f'SetParam:{gui_key}')
                
                time.sleep(0.05)  # Small delay between commands
            except Exception as e:
                self.log_status(f"✗ Error setting {gui_key}: {e}", error=True)
                self.log_error('E405', f'SetParam:{gui_key}:{e}')
        
        if success_count == len(param_mapping):
            self.log_status(f"✓ Uploaded {success_count} parameters to Arduino")
            messagebox.showinfo("Success", f"Uploaded {success_count} parameters to Arduino")
        else:
            self.log_status(f"⚠ Uploaded {success_count}/{len(param_mapping)} parameters", error=True)
            self.log_error('E407', f'{success_count}/{len(param_mapping)} params uploaded')
    
    def fetch_params(self):
        """Fetch current parameters from Arduino"""
        if not self.scanner or not self.scanner.ser:
            self.log_status("Arduino not connected", error=True)
            self.log_error('E402', 'fetch_params')
            return
        
        try:
            response = self._send_arduino("QueryParams")
            
            if response and "Params" in response:
                # Parse response: <Params,speed=700,zspeed=75,...>
                parts = response.replace("<", "").replace(">", "").split(",")
                
                param_mapping = {
                    "speed": "speed",
                    "zspeed": "zspeed",
                    "zespeed": "zespeed",
                    "xcal": "xcal",
                    "ycal": "ycal",
                    "zcal": "zcal",
                    "pickup_thresh": "pickup_thresh",
                    "release_thresh": "release_thresh",
                    "hcc": "hcc",
                    "ycc": "ycc",
                    "xcc": "xcc"
                }
                
                fetched = 0
                for part in parts[1:]:  # Skip "Params"
                    if "=" in part:
                        key, value = part.split("=")
                        if key in param_mapping:
                            gui_key = param_mapping[key]
                            self.param_vars[gui_key].set(value)
                            fetched += 1
                
                self.log_status(f"✓ Fetched {fetched} parameters from Arduino")
                messagebox.showinfo("Success", f"Fetched {fetched} parameters from Arduino")
            else:
                self.log_status("✗ Failed to fetch parameters", error=True)
                self.log_error('E406', 'QueryParams no valid response')
        except Exception as e:
            self.log_status(f"✗ Fetch error: {e}", error=True)
            self.log_error('E405', str(e))
    def _arduino_monitor_loop(self):
        """Monitor Arduino sensors"""
        while self.arduino_monitoring:
            if self.scanner and self.scanner.ser:
                try:
                    # Query sensor data from Arduino
                    response = self._send_arduino("QuerySensors")
                    
                    if response and "Sensors" in response:
                        # Parse response: <Sensors,range=45,xmin=0,xmax=1,ymin=0,ymax=1,zmin=0,zmax=1,home=1,started=0>
                        parts = response.replace("<", "").replace(">", "").split(",")
                        for part in parts[1:]:  # Skip "Sensors"
                            if "=" in part:
                                key, value = part.split("=")
                                if key == "range":
                                    self.sensor_data['range'] = int(value)
                                    self.range_label.configure(text=f"Range Sensor: {value} mm")
                                elif key == "started":
                                    # Update machine state from Arduino
                                    started_state = int(value) == 1
                                    if started_state != self.machine_started:
                                        self.machine_started = started_state
                                        if started_state:
                                            self.machine_state_label.configure(text="● STARTED", fg='#44ff44')
                                            self.start_machine_btn.configure(state=tk.DISABLED)
                                            self.stop_machine_btn.configure(state=tk.NORMAL)
                                        else:
                                            self.machine_state_label.configure(text="● STOPPED", fg='#ff4444')
                                            self.start_machine_btn.configure(state=tk.NORMAL)
                                            self.stop_machine_btn.configure(state=tk.DISABLED)
                                        self._update_controls_state()
                                elif key in self.sensor_data:
                                    val = int(value)
                                    self.sensor_data[key] = val
                                    if key in self.endstop_labels:
                                        label = self.endstop_labels[key]
                                        status = "●" if val == 1 else "○"
                                        color = "#00ff00" if val == 1 else "#888888"
                                        name = key.replace("_", "-").upper()
                                        label.configure(text=f"{name}: {status}", fg=color)
                except Exception as e:
                    self.log_status(f"Monitor error: {e}", error=True)
                    self.log_error('E405', f'monitor_loop:{e}')
            
            time.sleep(0.5)
    
    # Old CSV export methods removed - now using collection manager
    
    def _add_to_history(self, card_info, bin_result, sorting_mode):
        """Add scan to history and save to collection"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = card_info.get('name', 'Unknown')
        game = card_info.get('game', 'Unknown')
        set_code = card_info.get('set', 'N/A')
        
        # Save to collection using current default settings
        saved_entry = None
        if self.scanner and self.scanner.collection_manager:
            try:
                saved_entry = self.scanner.save_to_collection(
                    card_info,
                    quantity=1,
                    condition=self.default_condition.get(),
                    language=self.default_language.get(),
                    is_foil=self.default_foil.get()
                )
            except Exception as e:
                self.log_error('E603', str(e))
        
        # Extract data from saved entry or use defaults
        if saved_entry:
            condition = saved_entry.get('condition', 'N/A')
            language = saved_entry.get('language', 'N/A')
            is_foil = saved_entry.get('is_foil', False)
            sku = saved_entry.get('sku', 'N/A')
        else:
            condition = self.default_condition.get()
            language = self.default_language.get()
            is_foil = self.default_foil.get()
            sku = 'N/A'
        
        # Add to table (enhanced with more columns)
        self.scan_table.insert('', 0, values=(
            timestamp.split()[1],  # Time only
            name[:25],             # Card name (truncated)
            game[:12],             # Game
            set_code[:8],          # Set
            condition[:12],        # Condition
            language,              # Language
            'Yes' if is_foil else 'No',  # Foil
            sku                    # SKU
        ))
        
        self._update_stats()
    
    def export_collection(self):
        """Export collection using scanner's collection manager"""
        if not self.scanner or not self.scanner.collection_manager:
            messagebox.showerror("Error", "Collection manager not initialized")
            self.log_error('E601', 'export_collection')
            return
        
        try:
            format_type = 'tcgtraders' if self.export_format.get() == 'TCGTraders' else 'tcgplayer'
            
            # Export collection
            results = self.scanner.export_collection(format_type=format_type, by_game=False)
            
            if results:
                exported_file = results.get(format_type, 'Unknown')
                stats = self.scanner.collection_manager.get_stats()
                total = stats.get('total_scans', 0)
                
                self.export_status_label.config(
                    text=f"✓ Exported {total} cards to {exported_file}",
                    fg='#4CAF50'
                )
                self.log_status(f"✓ Collection exported: {exported_file}")
                
                messagebox.showinfo("Export Complete", 
                    f"Successfully exported {total} cards to:\n{exported_file}\n\nFormat: {self.export_format.get()}")
            else:
                messagebox.showwarning("Export Warning", "No cards to export")
        except Exception as e:
            self.log_error('E602', str(e))
            messagebox.showerror("Export Error", str(e))
    
    def clear_collection_session(self):
        """Clear current collection session"""
        if not self.scanner or not self.scanner.collection_manager:
            messagebox.showerror("Error", "Collection manager not initialized")
            self.log_error('E601', 'clear_collection_session')
            return
        
        if messagebox.askyesno("Clear Session", "Clear current session?\n\nThis will not affect the master collection."):
            try:
                self.scanner.collection_manager.clear_session()
                
                # Clear scan table
                for item in self.scan_table.get_children():
                    self.scan_table.delete(item)
                
                self.export_status_label.config(text="Session cleared", fg='#ffaa00')
                self.log_status("✓ Collection session cleared")
                self._update_stats()
            except Exception as e:
                self.log_status(f"✗ Clear session failed: {e}", error=True)
                self.log_error('E601', str(e))
                messagebox.showerror("Error", str(e))
    
    def clear_master_collection(self):
        """Clear entire master collection"""
        if not self.scanner or not self.scanner.collection_manager:
            messagebox.showerror("Error", "Collection manager not initialized")
            self.log_error('E601', 'clear_master_collection')
            return
        
        if messagebox.askyesno("Clear Master Collection", 
                              "⚠️ WARNING ⚠️\n\nThis will DELETE ALL cards from the master collection!\n\nThis action cannot be undone.\n\nAre you sure?"):
            try:
                # Clear master collection
                self.scanner.collection_manager.master_collection = {'cards': [], 'metadata': {'created': datetime.now().isoformat()}}
                self.scanner.collection_manager._save_collection(
                    self.scanner.collection_manager.master_collection, 
                    self.scanner.collection_manager.master_file
                )
                
                # Also clear session
                self.scanner.collection_manager.clear_session()
                
                # Clear scan table
                for item in self.scan_table.get_children():
                    self.scan_table.delete(item)
                
                self.export_status_label.config(text="Master collection cleared", fg='#ff4444')
                self.log_status("✓ Master collection cleared")
                self._update_stats()
                
                messagebox.showinfo("Success", "Master collection has been cleared.")
            except Exception as e:
                self.log_status(f"✗ Clear master collection failed: {e}", error=True)
                self.log_error('E601', str(e))
                messagebox.showerror("Error", str(e))
    
    def _update_stats(self):
        """Update statistics display using collection manager"""
        if self.scanner and self.scanner.collection_manager:
            stats = self.scanner.collection_manager.get_stats()
            
            total_scans = stats.get('total_scans', 0)
            session_scans = stats.get('session_scans', 0)
            
            self.stats_labels["Total Cards Saved"].configure(text=str(total_scans))
            self.stats_labels["Session Cards"].configure(text=str(session_scans))
            
            # Count unique cards by SKU
            if total_scans > 0:
                unique = len(set(card.get('sku') for card in self.scanner.collection_manager.master_collection.get('cards', []) if card.get('sku')))
            else:
                unique = 0
            self.stats_labels["Unique Cards"].configure(text=str(unique))
        else:
            # Fallback to scan history
            total = len(self.scan_table.get_children())
            self.stats_labels["Total Cards Saved"].configure(text=str(total))
            self.stats_labels["Session Cards"].configure(text=str(total))
            self.stats_labels["Unique Cards"].configure(text="N/A")
        
        if hasattr(self, 'session_start'):
            elapsed = int(time.time() - self.session_start)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.stats_labels["Session Time"].configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Schedule next update
        self.root.after(1000, self._update_stats)
    
    # clear_history removed - use clear_collection_session instead
    
    def start_camera(self):
        """Start camera"""
        if self.running:
            return
        # If a camera plugin is available, prefer it
        if getattr(self, 'camera_plugin', None):
            try:
                ok = bool(self.camera_plugin.open(0))
            except Exception as e:
                ok = False
                self.log_status(f"Camera plugin open failed: {e}", error=True)

            if not ok:
                self.log_status("✗ Camera plugin failed to open, falling back to OpenCV", error=True)
                self.camera_plugin = None

        if not getattr(self, 'camera_plugin', None):
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.log_error('E201', f'Camera index {self.camera_index_var.get() if hasattr(self, "camera_index_var") else 0}')
                messagebox.showerror("Error", "Cannot open camera")
                return
            # Try to reduce internal camera buffering to lower latency (may be ignored on some platforms)
            try:
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            try:
                # Request a reasonable FPS if supported
                self.camera.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass
            # Try to stabilize camera exposure/whitebalance where supported (only on Linux)
            try:
                import platform
                if platform.system().lower() == 'linux':
                    # Disable auto exposure where supported
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    # Set a reasonable exposure value (may be camera-dependent)
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)
                    # Disable auto white balance
                    self.camera.set(cv2.CAP_PROP_AUTO_WB, 0)
                    # Optional: set gain lower
                    self.camera.set(cv2.CAP_PROP_GAIN, 0)
            except Exception:
                pass
        
        self.running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.log_status("✓ Camera started")
        threading.Thread(target=self._video_loop, daemon=True).start()
    
    def stop_camera(self):
        """Stop camera"""
        self.running = False
        try:
            self._last_display_frame = None
        except Exception:
            pass
        # Release the camera immediately so any blocking camera.read() in the
        # background thread returns (False, None) right away — on Windows,
        # release() unblocks a pending read(). Null out self.camera first so the
        # background thread's cleanup won't double-release.
        if getattr(self, 'camera_plugin', None):
            plugin = self.camera_plugin
            self.camera_plugin = None
            try:
                plugin.close()
            except Exception:
                pass
        else:
            cam = getattr(self, 'camera', None)
            self.camera = None
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.log_status("⏸ Camera stopped")
    
    def _video_loop(self):
        """Video processing loop"""
        while self.running:
            if getattr(self, 'camera_plugin', None):
                try:
                    ret, frame = self.camera_plugin.read()
                except Exception:
                    ret = False
                    frame = None
            else:
                cam = getattr(self, 'camera', None)
                if cam is None:
                    break  # stop_camera released the camera
                try:
                    ret, frame = cam.read()
                except Exception:
                    ret = False
                    frame = None
            if not ret:
                break
            
            self.current_frame = frame.copy()
            display_frame = frame.copy()
            
            # Normalize brightness to reduce flashing before detection
            # Display-normalized frame to reduce flashing, but use raw frame for detection
            try:
                proc_frame = self._normalize_brightness(frame)
            except Exception:
                proc_frame = frame

            # Use original/raw frame for contour detection to match the Original recognition pipeline
            card_approx = self.scanner._find_card_contour(frame)
            
            if card_approx is not None:
                cv2.drawContours(display_frame, [card_approx], -1, (0, 255, 0), 3)
                cv2.putText(display_frame, "CARD DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Only start a scan if cooldown has passed and no scan in progress.
                if self.auto_scan and (time.time() - self.last_scan_time) > self.scan_cooldown and not getattr(self, '_scan_in_progress', False):
                    # Show immediate UI feedback that scanning will occur
                    try:
                        self.detection_info = {'status': 'scanning', 'message': 'Scanning...'}
                    except Exception:
                        pass
                    # Set in-progress flag and schedule the heavy work shortly so the UI can update first
                    try:
                        self._scan_in_progress = True
                        self.root.after(10, lambda f=frame.copy(), a=card_approx.copy(): threading.Thread(target=self._run_scan_thread, args=(f, a), daemon=True).start())
                    except Exception:
                        # Fallback to immediate thread start
                        try:
                            threading.Thread(target=self._run_scan_thread, args=(frame.copy(), card_approx.copy()), daemon=True).start()
                        except Exception:
                            pass
            else:
                cv2.putText(display_frame, "NO CARD", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if self.detection_info:
                self._draw_detection_info(display_frame, self.detection_info)

            # Temporal smoothing: blend with previous displayed frame to reduce flicker
            try:
                if getattr(self, '_last_display_frame', None) is not None and getattr(self, 'display_smooth_alpha', 0) > 0:
                    try:
                        prev = self._last_display_frame.astype('float32')
                        cur = display_frame.astype('float32')
                        a = float(self.display_smooth_alpha)
                        blended = cv2.addWeighted(prev, a, cur, 1.0 - a, 0)
                        display_frame = blended.astype('uint8')
                    except Exception:
                        pass
                try:
                    self._last_display_frame = display_frame.copy()
                except Exception:
                    self._last_display_frame = None
            except Exception:
                pass

            try:
                # Schedule canvas update on the main thread to avoid Tk threading issues
                self.root.after(0, lambda f=display_frame: self._update_canvas(f))
            except Exception:
                # Fallback: call directly (older behavior)
                self._update_canvas(display_frame)

            time.sleep(0.03)

        # Loop exited — release any remaining resources (stop_camera may have already done this).
        if getattr(self, 'camera_plugin', None):
            plugin = self.camera_plugin
            self.camera_plugin = None
            try:
                plugin.close()
            except Exception:
                pass
        else:
            cam = getattr(self, 'camera', None)
            self.camera = None
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
        # Ensure buttons are in the correct state regardless of why the loop stopped.
        try:
            self.root.after(0, lambda: [
                self.start_btn.configure(state=tk.NORMAL),
                self.stop_btn.configure(state=tk.DISABLED),
            ])
        except Exception:
            pass

    def _normalize_brightness(self, frame):
        """Normalize brightness using HSV median scaling to reduce flashes between frames."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            median_v = int(cv2.medianBlur(v, 5).mean())
            if self._last_v_median is None:
                self._last_v_median = median_v
            # Smooth median to avoid oscillation
            alpha = 0.1
            self._last_v_median = int(alpha * median_v + (1 - alpha) * self._last_v_median)
            if self._last_v_median <= 0:
                return frame
            factor = float(self._target_v) / float(self._last_v_median)
            # Apply scaling and clip
            v = cv2.multiply(v.astype('float32'), factor)
            v = np.clip(v, 0, 255).astype('uint8')
            hsv2 = cv2.merge([h, s, v])
            out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
            return out
        except Exception:
            return frame

    def _run_scan_thread(self, frame, card_approx):
        """Run the heavy scan work in a background thread. GUI updates should be scheduled via root.after inside called functions."""
        try:
            try:
                self._perform_scan(frame, card_approx)
            except Exception as e:
                # Schedule a thread-safe log
                try:
                    self.root.after(0, lambda: self.log_status(f"Scan thread error: {e}", error=True))
                    self.root.after(0, lambda err=e: self.log_error('E304', str(err)))
                except Exception:
                    pass
        finally:
            # ensure flag cleared on completion
            try:
                self._scan_in_progress = False
            except Exception:
                pass
            # Update last scan time AFTER the scan and any Arduino handshake completes
            try:
                self.last_scan_time = time.time()
            except Exception:
                pass
    
    def manual_scan(self):
        """Manual scan"""
        if not self.running or self.current_frame is None:
            self.log_status("Start camera first", error=True)
            return
        
        card_approx = self.scanner._find_card_contour(self.current_frame)
        if card_approx is not None:
            self._perform_scan(self.current_frame, card_approx)
        else:
            self.log_status("✗ No card detected", error=True)
            self.log_error('E203', 'manual_scan')
    
    def _perform_scan(self, frame, card_approx):
        """Scan card"""
        try:
            scan_image_path = None
            try:
                warped = self.scanner._get_perspective_corrected_card(frame, card_approx)
                if warped is not None:
                    # Ensure vertical orientation
                    h, w = warped.shape[:2]
                    if w > h:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                        h, w = warped.shape[:2]

                    # Final sanity: vertical required
                    if h > w:
                        scans_dir = os.path.join('Collection', 'ScanImages')
                        os.makedirs(scans_dir, exist_ok=True)
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        # product_id may not be known yet; fill after recognition if needed
                        tmp_name = f"scan_{ts}_{int(time.time()*1000)}.jpg"
                        scan_image_path = os.path.join(scans_dir, tmp_name)
                        cv2.imwrite(scan_image_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            except Exception as _e:
                scan_image_path = None

            card_info = None
            # Allow recognition plugin to handle recognition; fallback to built-in scanner
            if getattr(self, 'recognition_plugin', None):
                try:
                    card_info = self.recognition_plugin.recognize(frame, card_approx, self.scanner)
                except Exception as e:
                    self.log_status(f"Recognition plugin error: {e}", error=True)
                    self.log_error('E305', str(e))

            if not card_info:
                card_info = self.scanner._process_card_from_contour(frame, card_approx)
            
            if not card_info:
                self.detection_info = {'status': 'error', 'message': 'Not recognized', 'bin': 'RejectCard'}
                self.log_error('E301')
                return

            # Attach scan image path + backend to card_info for persistence
            try:
                if scan_image_path and os.path.exists(scan_image_path):
                    # If filename lacks product id, rename now that we know it
                    pid = str(card_info.get('product_id') or card_info.get('UniqueID') or '').strip()
                    if pid and ('scan_' in os.path.basename(scan_image_path)) and (pid not in os.path.basename(scan_image_path)):
                        try:
                            base = os.path.dirname(scan_image_path)
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            new_path = os.path.join(base, f"scan_{ts}_{pid}.jpg")
                            # Avoid overwrite
                            if os.path.exists(new_path):
                                new_path = os.path.join(base, f"scan_{ts}_{pid}_{int(time.time()*1000)}.jpg")
                            os.replace(scan_image_path, new_path)
                            scan_image_path = new_path
                        except Exception:
                            pass

                    card_info['scan_image_path'] = scan_image_path
                    # Store which backend is active
                    try:
                        backend = self.vector_backend.get() if hasattr(self, 'vector_backend') else None
                    except Exception:
                        backend = None
                    if backend == 'PHASH_RGB':
                        card_info['scan_backend'] = 'pHash (RGB)'
                    elif backend == 'PHASH_GRAY':
                        card_info['scan_backend'] = 'pHash (Grayscale)'
            except Exception:
                pass
            
            if self.scanner.track_inventory:
                card_info = self.scanner.check_inventory(card_info)
                if card_info == "RejectCard":
                    self.detection_info = {'status': 'reject', 'message': 'In inventory', 'bin': 'RejectCard'}
                    self.log_status("⊗ Already in inventory")
                    if getattr(self.scanner, 'ser', None) or (getattr(self, 'arduino_plugin', None) and self.arduino_plugin.is_connected()):
                        self._send_arduino("RejectCard")
                    return
            
            sorting_mode = self.sorting_var.get()
            try:
                threshold = float(self.threshold_var.get())
            except:
                threshold = 1000000
            
            bin_result = self.scanner.get_bin_number(card_info, sorting_mode, threshold)
            
            self.detection_info = {'status': 'success', 'card': card_info, 'bin': bin_result, 'sorting_mode': sorting_mode}
            
            name = card_info.get('name', 'Unknown')
            game = card_info.get('game', 'Unknown')
            confidence = card_info.get('confidence', 0)
            price = card_info.get('market_price')
            
            self.log_status(f"✓ {name} ({game}) {confidence:.1f}% → {bin_result}")
            
            # Add to history
            self._add_to_history(card_info, bin_result, sorting_mode)
            
            if getattr(self.scanner, 'ser', None) or (getattr(self, 'arduino_plugin', None) and self.arduino_plugin.is_connected()):
                response = self._send_arduino(bin_result)
                if response:
                    self._handle_arduino_response(response, bin_result)
        
        except Exception as e:
            self.log_error('E304', str(e))
    
    def _draw_detection_info(self, frame, info):
        """Draw info on frame"""
        # Defensive: `info` may be None or may not contain all expected keys.
        if not info:
            return

        status = info.get('status')
        bin_val = info.get('bin', None)

        if status == 'success':
            card = info.get('card', {}) or {}
            y = 70
            cv2.putText(frame, f"Card: {card.get('name', 'Unknown')}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(frame, f"Game: {card.get('game', 'Unknown')}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30
            try:
                conf = float(card.get('confidence', 0))
            except Exception:
                conf = 0.0
            cv2.putText(frame, f"Confidence: {conf:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30
            try:
                phash_conf = float(card.get('phash_confidence', 0))
            except Exception:
                phash_conf = 0.0
            cv2.putText(frame, f"pHash: {phash_conf:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y += 30
            try:
                mser_score = float(card.get('mser_score', 0)) * 100.0
            except Exception:
                mser_score = 0.0
            cv2.putText(frame, f"MSER: {mser_score:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y += 30
            if card.get('market_price') is not None:
                try:
                    price = float(card.get('market_price'))
                    cv2.putText(frame, f"Price: ${price:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 30
                except Exception:
                    pass
            if bin_val is not None:
                cv2.putText(frame, f"BIN: {bin_val}", (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            msg = (info.get('message') or 'Error').upper()
            cv2.putText(frame, msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if bin_val is not None:
                cv2.putText(frame, f"BIN: {bin_val}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    def _update_canvas(self, frame):
        """Update canvas with double-buffered image item to reduce flicker."""
        h, w = frame.shape[:2]
        scale = min(self.canvas_width / w, self.canvas_height / h)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        x_offset = (self.canvas_width - int(w * scale)) // 2
        y_offset = (self.canvas_height - int(h * scale)) // 2

        # Reuse a single image item on the canvas to avoid deleting/creating each frame
        try:
            if not hasattr(self, '_canvas_image_id') or self._canvas_image_id is None:
                self._canvas_image_id = self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
            else:
                try:
                    self.canvas.coords(self._canvas_image_id, x_offset, y_offset)
                    self.canvas.itemconfig(self._canvas_image_id, image=photo)
                except Exception:
                    # If item was deleted or invalid, recreate
                    self._canvas_image_id = self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        except Exception:
            # As a last resort, clear and draw
            try:
                self.canvas.delete("all")
            except Exception:
                pass
            self._canvas_image_id = self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)

        # Keep a reference to prevent GC
        self.canvas.image = photo
    
    def log_status(self, message, error=False):
        """Log to status window"""
        # Ensure UI updates happen on the main thread
        def _append():
            try:
                self.status_text.configure(state=tk.NORMAL)
                timestamp = time.strftime("%H:%M:%S")
                self.status_text.tag_config("error", foreground="#ff4444")
                self.status_text.tag_config("normal", foreground="#00ff00")
                self.status_text.insert(tk.END, f"[{timestamp}] {message}\n", "error" if error else "normal")
                self.status_text.see(tk.END)
                self.status_text.configure(state=tk.DISABLED)
            except Exception:
                try:
                    print(f"[{time.strftime('%H:%M:%S')}] {message}")
                except Exception:
                    pass

        try:
            self.root.after(0, _append)
        except Exception:
            _append()

    def apply_compact_mode(self):
        """Apply compact UI settings to reduce fonts, paddings and panel sizes."""
        try:
            compact = bool(self.compact_mode_var.get())
        except Exception:
            compact = False

        if compact:
            # Smaller canvas for faster resize and smaller layout
            self.canvas_width = 480
            self.canvas_height = 360
            # Slightly smaller fonts
            self.default_font = ('Segoe UI', 9)
            self.header_font = ('Segoe UI', 10, 'bold')
            self.title_font = ('Segoe UI', 11, 'bold')
            self.scan_cooldown = max(0.5, self.scan_cooldown)  # keep responsive
        else:
            self.canvas_width = 600
            self.canvas_height = 540
            self.default_font = ('Segoe UI', 10)
            self.header_font = ('Segoe UI', 11, 'bold')
            self.title_font = ('Segoe UI', 12, 'bold')

        # Apply to existing widgets where possible
        try:
            if getattr(self, 'canvas', None):
                self.canvas.config(width=self.canvas_width, height=self.canvas_height)
        except Exception:
            pass

        try:
            # Update Start/Stop button paddings for compactness
            if getattr(self, 'start_btn', None):
                if compact:
                    self.start_btn.config(padx=12, pady=6, font=('Segoe UI', 10, 'bold'))
                    self.stop_btn.config(padx=10, pady=5, font=('Segoe UI', 9, 'bold'))
                else:
                    self.start_btn.config(padx=20, pady=10, font=self.header_font)
                    self.stop_btn.config(padx=15, pady=8, font=('Arial', 11, 'bold'))
        except Exception:
            pass
    
    
    def on_close(self):
        """Cleanup"""
        self.running = False
        self.arduino_monitoring = False
        if self.camera:
            self.camera.release()
        if self.scanner:
            self.scanner.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ScannerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == '__main__':
    main()

