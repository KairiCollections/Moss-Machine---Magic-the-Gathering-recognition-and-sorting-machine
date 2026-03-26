#!/usr/bin/env python3
"""
Optimized Universal Card Scanner
High-performance version with multi-threading, early termination, and hash indexing

Performance optimizations:
1. Multi-threaded game scanning (parallel processing)
2. Early termination when exact match found
3. Progressive threshold tightening
4. Batch SQL queries with LIMIT
5. In-memory hash caching for frequently scanned games
6. Distance pre-filtering (quick rejection of bad matches)
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
        "imagehash:imagehash",
        "truststore:truststore",
    ]
    for pkg in (pkg for mod, pkg in (s.split(":") for s in _REQUIRED)
               if not __import__('importlib').util.find_spec(mod)):
        try:
            print(f"[startup] Installing {pkg}...", flush=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-input", pkg]
            )
        except Exception as e:
            print(f"[startup] Warning: could not install {pkg}: {e}", flush=True)
_ensure_packages()
# Inject Windows/macOS/Linux native certificate store so HTTPS requests
# succeed even when the server's intermediate CA is missing from certifi.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
# ---------------------------------------------------------------------------
try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import sqlite3
from pathlib import Path
import imagehash
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
try:
    import serial
except Exception:
    serial = None
import os
import json

# Import collection manager
from card_collection_manager import CardCollectionManager
from mtg_symbol_recognizer import (
    load_set_symbols,
    match_symbol_template,
)

def download_database(db_path="unified_card_database.db"):
    """Download database from server if it doesn't exist locally.

    Accepts any filename; attempts to download from configured servers
    under the `/download/<filename>` endpoint.
    """
    # If absolute path provided, use the basename for remote path but save to the full path
    target_path = Path(db_path)
    filename = target_path.name

    if target_path.exists():
        return True

    print(f"[!] Database not found at {db_path}")
    print("[*] Attempting to download from server...")

    servers = [
        "https://www.tcgtraders.app"
    ]

    for base in servers:
        server_url = f"{base}/download/{filename}"
        try:
            print(f"[*] Trying {server_url}...")
            import urllib.request
            import ssl

            if server_url.startswith("http://"):
                context = None
            else:
                context = ssl._create_unverified_context()

            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    print(f"\r[*] Downloading: {percent}%", end='', flush=True)

            if context:
                urllib.request.urlretrieve(server_url, str(target_path), reporthook=reporthook, context=context)
            else:
                urllib.request.urlretrieve(server_url, str(target_path), reporthook=reporthook)

            print(f"\n[+] Successfully downloaded database from {server_url}")
            return True
        except Exception as e:
            print(f"\n[!] Failed to download from {server_url}: {e}")
            continue

    print("[!] Could not download database from any server")
    return False

class OptimizedCardScanner:
    def __init__(self, db_path="unified_card_database.db", max_workers=8, cache_enabled=True, 
                 serial_port=None, baud_rate=9600, use_grayscale_phash=False,
                 auto_vector_when_unfiltered=True, enable_collection=True,
                 default_condition='Near Mint', default_language='EN', default_foil=False,
                 prompt_for_details=True,
                 enable_mser_scoring=True, mser_weight=0.15):
        """Initialize optimized scanner
        
        Args:
            default_condition: Default condition for saved cards (Near Mint, Lightly Played, etc.)
            default_language: Default language code (EN, JP, FR, etc.)
            default_foil: Default foil status (True/False)
            prompt_for_details: If False, use defaults without prompting
        """
        self.db_path = db_path
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        
        # Collection default settings
        self.default_condition = default_condition
        self.default_language = default_language
        self.default_foil = default_foil
        self.prompt_for_details = prompt_for_details
        
        # Download database if it doesn't exist
        if not download_database(db_path):
            raise FileNotFoundError(f"Database not found and could not be downloaded: {db_path}")
        
        # Main database connection
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Thread-local storage for database connections
        self.local = threading.local()
        
        # Hash cache for popular games (Magic, Pokemon, YuGiOh)
        self.hash_cache = {}
        
        # Collection manager for saving scans
        self.collection_enabled = enable_collection
        self.collection_manager = None
        if enable_collection:
            try:
                self.collection_manager = CardCollectionManager()
                print("[+] Collection manager initialized")
            except Exception as e:
                print(f"[!] Collection manager disabled: {e}")
                self.collection_enabled = False
        
        # Load games list and find actual table names from database
        self.games = {}
        
        # Check if game_table_mapping exists for fast lookups
        try:
            mapping_exists = self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='game_table_mapping'"
            ).fetchone()
        except:
            mapping_exists = None
        
        # Get game list
        try:
            games_data = self.cursor.execute("SELECT id, name, display_name, total_cards FROM games").fetchall()
        except sqlite3.OperationalError:
            rows = self.cursor.execute("SELECT id, name, total_cards FROM games").fetchall()
            # normalize to 4-tuple with display_name == name
            games_data = [(r[0], r[1], r[1], r[2] if len(r) > 2 else 0) for r in rows]
        
        # If game_table_mapping exists, use it for accurate lookups
        if mapping_exists:
            mapping = {}
            for row in self.cursor.execute("SELECT game_name, table_name FROM game_table_mapping").fetchall():
                mapping[row[0]] = row[1]
            
            for game_id, name, display_name, total_cards in games_data:
                if name in mapping:
                    self.games[name] = {
                        'id': game_id,
                        'display_name': display_name,
                        'total_cards': total_cards,
                        'table': mapping[name]
                    }
        else:
            # Fallback: use cards_{id} pattern directly from games.id
            for game_id, name, display_name, total_cards in games_data:
                table_name = f'cards_{game_id}'
                # Verify table exists
                if self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone():
                    self.games[name] = {
                        'id': game_id,
                        'display_name': display_name,
                        'total_cards': total_cards,
                        'table': table_name
                    }
        
        print(f"[+] Loaded {len(self.games)} games")
        print(f"[+] Max workers: {max_workers}")
        print(f"[+] Hash caching: {'enabled' if cache_enabled else 'disabled'}")
        
        # Default: scan all games
        self.active_games = list(self.games.keys())

        
        # Serial communication for Arduino
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.start_marker = 60  # '<'
        self.end_marker = 62    # '>'
        
        # Inventory tracking
        self.inventory_file = "Collection/Collection.txt"
        self.track_inventory = False

        # Matching parameters (can be adjusted by GUI at runtime)
        self.scan_threshold = 40
        self.quick_filter_max = 80
        
        # Performance stats
        self.stats = {
            'scans': 0,
            'total_time': 0,
            'cards_checked': 0,
            'cache_hits': 0
        }
        # Matching backend (pHash only; vector/ORB disabled)
        self.use_vector = False
        self.use_resnet50 = False
        self.use_orb = False
        self._orb = None
        self.use_grayscale_phash = use_grayscale_phash
        self.auto_vector_when_unfiltered = False
        self.vector_searcher = None
        # MSER-based quality scoring (input-only signal, blended into confidence)
        self.enable_mser_scoring = enable_mser_scoring
        self.mser_weight = float(mser_weight)

        # MTG set symbol verification
        self.enable_set_symbol_check = True
        self.symbol_min_confidence = 0.50
        self._mtg_symbols = None
        self._mtg_roi_config = None

    def get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.local.cursor = self.local.conn.cursor()
        return self.local.cursor
    
    # =====================================================================
    # SERIAL COMMUNICATION (Arduino Interface)
    # =====================================================================
    
    def init_serial(self):
        """Initialize serial connection to Arduino"""
        if not self.serial_port:
            print("[!] Serial port not configured")
            return False
        
        try:
            # Set a read timeout so recv operations don't block forever
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=5)
            print(f"[+] Serial port {self.serial_port} opened (Baudrate: {self.baud_rate})")
            self.wait_for_arduino()
            return True
        except Exception as e:
            print(f"[!] Failed to open serial port: {e}")
            return False
    
    def send_to_arduino(self, send_str):
        """Send data to Arduino with proper encoding"""
        if not self.ser or not send_str:
            return None

        try:
            # Write the command
            self.ser.write(send_str.encode('utf-8'))
            print(f"[->] Sent to Arduino: {send_str}")

            # Wait for one delimited response from Arduino (uses recv_from_arduino)
            resp = self.recv_from_arduino()
            if resp:
                print(f"[<-] Arduino: {resp}")
            else:
                print("[!] No response received from Arduino (timeout)")

            return resp
        except Exception as e:
            print(f"[!] Error sending to Arduino: {e}")
            return None
    
    def recv_from_arduino(self):
        """Receive data from Arduino using start/end markers"""
        if not self.ser:
            return ""
        ck = ""
        started = False
        start_b = bytes([self.start_marker])
        end_b = bytes([self.end_marker])

        while True:
            b = self.ser.read(1)
            if not b:
                # timeout or no data
                break

            if not started:
                if b == start_b:
                    started = True
                # ignore until start marker
                continue

            # started == True
            if b == end_b:
                break

            try:
                ck += b.decode('utf-8', errors='replace')
            except Exception:
                # ignore decode errors
                pass

        return ck
    
    def wait_for_arduino(self):
        """Wait for Arduino ready signal"""
        if not self.ser:
            return
        
        msg = ""
        while "Arduino is ready" not in msg:
            while self.ser.in_waiting == 0:
                pass
            msg = self.recv_from_arduino()
            if msg:
                print(f"[<-] Arduino: {msg}")
    
    # =====================================================================
    # SORTING & BIN ASSIGNMENT
    # =====================================================================
    
    @staticmethod
    def is_basic_land(name):
        """Check if card is a basic land"""
        return name.lower() in {"plains", "island", "swamp", "mountain", "forest", "wastes"}
    
    @staticmethod
    def is_land_card(types):
        """Check if card has 'land' type"""
        if isinstance(types, str):
            return "land" in types.lower()
        elif isinstance(types, list):
            return any("land" in t.lower() for t in types)
        return False
    
    def get_bin_color(self, card_info):
        """Get bin based on card color"""
        types = card_info.get("type", "") or card_info.get("types", [])
        
        if self.is_land_card(types):
            name = card_info.get("name", "") or card_info.get("card_name", "")
            return "Basic land" if self.is_basic_land(name) else "Nonbasic land"
        
        # Try different color field names
        colors = (card_info.get("colors") or 
                 card_info.get("color") or 
                 card_info.get("Colors") or [])
        
        if isinstance(colors, str):
            colors = [colors]
        
        if not colors or len(colors) == 0:
            return "Colorless"
        
        return "Multicolor" if len(colors) > 1 else colors[0]
    
    def get_bin_mana(self, card_info):
        """Get bin based on mana value (CMC)"""
        mv = (card_info.get("cmc") or 
              card_info.get("mana_value") or 
              card_info.get("convertedManaCost") or 0)
        
        try:
            mv = int(float(mv))
        except:
            mv = 0
        
        if mv <= 1:
            return "One"
        elif mv <= 8:
            return str(mv).capitalize()
        else:
            return "RejectCard"
    
    def get_bin_set(self, card_info):
        """Get bin based on set code"""
        set_code = (card_info.get("set") or 
                   card_info.get("set_code") or 
                   card_info.get("setCode") or "???").lower()
        
        types = card_info.get("type", "") or card_info.get("types", [])
        
        if isinstance(types, str) and "token" in types.lower():
            return "token"
        elif isinstance(types, list) and any("token" in t.lower() for t in types):
            return "token"
        
        return set_code if set_code != "???" else "RejectCard"
    
    def get_bin_price(self, card_info, threshold=1000000):
        """Get bin based on price"""
        price_str = (card_info.get("market_price") or 
                    card_info.get("price") or 
                    card_info.get("Price") or "null")
        
        if price_str == "null" or price_str is None:
            return "RejectCard"
        
        try:
            price = float(str(price_str).strip('$'))
        except (ValueError, TypeError):
            return "RejectCard"
        
        # Price binning
        bins = {
            0.02: "tray1",
            0.05: "tray7",
            0.10: "tray14",
            0.25: "tray18",
            0.50: "tray21",
            1.0: "tray24",
            2.0: "tray25",
            4.0: "tray26",
            8.0: "tray27",
            16.0: "tray28",
            32.0: "tray29",
            64.0: "tray30",
            128.0: "tray31",
            float('inf'): "tray32"
        }
        
        for upper_limit, bin_name in bins.items():
            if price <= upper_limit and price <= threshold:
                return bin_name
        
        return "RejectCard"
    
    def get_bin_type(self, card_info):
        """Get bin based on card type"""
        types = card_info.get("type", "") or card_info.get("types", [])
        
        if isinstance(types, str):
            types = [types]
        
        type_mapping = {
            "creature": "creature",
            "artifact": "artifact",
            "enchantment": "enchantment",
            "instant": "instant",
            "sorcery": "sorcery",
            "battle": "battle",
            "planeswalker": "planeswalker",
            "land": "land",
            "token": "token"
        }
        
        for card_type in types:
            card_type_lower = str(card_type).lower()
            for key, value in type_mapping.items():
                if key in card_type_lower:
                    return value
        
        return "RejectCard"
    
    def get_bin_alpha(self, card_info):
        """Get bin based on first letter of card name (A-Z alphabetical)"""
        name = (card_info.get("name") or 
                card_info.get("card_name") or 
                card_info.get("Name") or "")
        
        if not name:
            return "RejectCard"
        
        # Get first character and convert to uppercase
        first_char = name.strip()[0].upper()
        
        # Check if it's a letter A-Z
        if first_char.isalpha():
            return first_char
        else:
            # Numbers and symbols go to a special bin
            return "0-9"
    
    def get_bin_rarity(self, card_info):
        """Get bin based on card rarity"""
        rarity = (card_info.get("rarity") or 
                 card_info.get("Rarity") or "").lower()
        
        if not rarity:
            return "RejectCard"
        
        # Standard rarity bins
        rarity_mapping = {
            "common": "Common",
            "uncommon": "Uncommon",
            "rare": "Rare",
            "mythic": "Mythic",
            "mythic rare": "Mythic",
            "special": "Special",
            "bonus": "Bonus",
            "promo": "Promo",
            "token": "Token"
        }
        
        for key, value in rarity_mapping.items():
            if key in rarity:
                return value
        
        return "RejectCard"
    
    def get_bin_finish(self, card_info):
        """Get bin based on foil/nonfoil finish"""
        # Check various foil field names across different games
        finishes = card_info.get("finishes") or card_info.get("finish") or []
        is_foil = card_info.get("foil") or card_info.get("isFoil") or False
        
        # Handle different data structures
        if isinstance(finishes, list):
            has_foil = any("foil" in str(f).lower() for f in finishes)
            has_nonfoil = any("nonfoil" in str(f).lower() or "normal" in str(f).lower() for f in finishes)
            
            if has_foil and has_nonfoil:
                return "Both"
            elif has_foil:
                return "Foil"
            elif has_nonfoil:
                return "Nonfoil"
        
        # Simple boolean check
        if isinstance(is_foil, bool):
            return "Foil" if is_foil else "Nonfoil"
        
        # String check
        if isinstance(is_foil, str):
            return "Foil" if is_foil.lower() in ["true", "yes", "foil"] else "Nonfoil"
        
        return "Nonfoil"  # Default to nonfoil if unknown
    
    def get_bin_number(self, card_info, mode, threshold=1000000):
        """
        Get bin number for card based on sorting mode
        Modes: color, mana_value, set, price, type, buy, alpha, rarity, finish
        """
        if not card_info or card_info == "RejectCard":
            return "RejectCard"
        
        if mode == "color":
            return self.get_bin_color(card_info)
        elif mode == "mana_value":
            return self.get_bin_mana(card_info)
        elif mode == "set":
            return self.get_bin_set(card_info)
        elif mode == "price":
            return self.get_bin_price(card_info, 1000000)
        elif mode == "type":
            return self.get_bin_type(card_info)
        elif mode == "buy":
            return self.get_bin_price(card_info, threshold)
        elif mode == "alpha":
            return self.get_bin_alpha(card_info)
        elif mode == "rarity":
            return self.get_bin_rarity(card_info)
        elif mode == "finish":
            return self.get_bin_finish(card_info)
        else:
            return "RejectCard"
    
    # =====================================================================
    # INVENTORY TRACKING
    # =====================================================================
    
    def enable_inventory_tracking(self, enable=True):
        """Enable or disable inventory tracking"""
        self.track_inventory = enable
        if enable:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.inventory_file), exist_ok=True)
            # Create file if it doesn't exist
            if not os.path.exists(self.inventory_file):
                open(self.inventory_file, 'w').close()
            print(f"[+] Inventory tracking enabled: {self.inventory_file}")
    
    def check_inventory(self, card_info):
        """
        Check if card is already in inventory
        Returns "RejectCard" if found, otherwise adds it and returns card_info
        """
        if not self.track_inventory:
            return card_info
        
        card_str = str(card_info)
        
        try:
            # Read existing inventory
            with open(self.inventory_file, 'r') as f:
                if card_str in f.read():
                    print('[*] Card already in inventory - rejecting')
                    return "RejectCard"
            
            # Add to inventory
            with open(self.inventory_file, 'a') as f:
                f.write(card_str + "\n")
            
            print('[+] Card added to inventory')
            return card_info
        
        except Exception as e:
            print(f"[!] Inventory check error: {e}")
            return card_info
    

    def set_active_games(self, game_names):
        """Limit scanning to specific games"""
        self.active_games = [g for g in game_names if g in self.games]
        print(f"[+] Active games: {len(self.active_games)}")
    
    def preload_cache(self, games=['Magic', 'Pokemon', 'YuGiOh']):
        """Preload hash cache for specified games (WARNING: memory intensive)"""
        if not self.cache_enabled:
            return
        
        print("\n[*] Preloading hash cache...")
        for game_name in games:
            if game_name not in self.games:
                continue
            
            print(f"    Loading {game_name}...", end='', flush=True)
            start = time.time()
            
            table = self.games[game_name]['table']
            try:
                # Use optimized SELECT with only needed columns for faster loading
                needed_cols = "product_id, name, number, r_phash, g_phash, b_phash, set_code, rarity, subTypeName, market_price, low_price"
                try:
                    query = f"SELECT {needed_cols} FROM {table} WHERE r_phash IS NOT NULL"
                    rows = self.cursor.execute(query).fetchall()
                    colnames = [d[0] for d in self.cursor.description]
                    # convert to list of dicts for robust column-name access
                    cards = [dict(zip(colnames, row)) for row in rows]
                except sqlite3.OperationalError:
                    print(f" Table not found")
                    cards = []

                self.hash_cache[game_name] = cards
                
                elapsed = time.time() - start
                print(f" {len(cards):,} cards loaded in {elapsed:.2f}s")
            except sqlite3.OperationalError:
                print(f" Table not found")
    
    def compute_phash(self, image):
        """Compute perceptual hash (256-bit, hash_size=16).

        Behavior aligned with `scripts/recompute_phashes_square.py`:
        - Rotate image 90deg if width > height
        - Resize to fit within a 1024x1024 square preserving aspect ratio
        - Paste centered onto a white 1024x1024 background (letterbox)
        - Compute per-channel pHash with hash_size=16 and normalize to 64-hex
        Originals are not modified.
        """
        if isinstance(image, np.ndarray):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Rotate to portrait if necessary
        if pil_image.width > pil_image.height:
            pil_image = pil_image.transpose(Image.ROTATE_90)

        size = 1024
        w, h = pil_image.size
        if w == 0 or h == 0:
            return None, None, None

        scale = min(float(size) / w, float(size) / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im_resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
        bg = Image.new('RGB', (size, size), (255, 255, 255))
        offset = ((size - new_w) // 2, (size - new_h) // 2)
        bg.paste(im_resized, offset)
        sq = bg

        def _norm(phash_obj):
            if not phash_obj:
                return None
            s = str(phash_obj)
            if s.startswith('0x'):
                s = s[2:]
            s = s.lower()
            return s.ljust(64, '0')[:64]

        if self.use_grayscale_phash:
            try:
                h = imagehash.phash(sq.convert('L'), hash_size=16)
                nh = _norm(h)
                return nh, nh, nh
            except Exception:
                return None, None, None

        r, g, b = sq.split()
        try:
            r_h = _norm(imagehash.phash(r, hash_size=16))
            g_h = _norm(imagehash.phash(g, hash_size=16))
            b_h = _norm(imagehash.phash(b, hash_size=16))
        except Exception:
            return None, None, None

        return r_h, g_h, b_h

    def _compute_mser_score(self, image):
        """Compute a lightweight MSER quality score for the input image (0..1)."""
        if not self.enable_mser_scoring or cv2 is None:
            return None

        try:
            if isinstance(image, Image.Image):
                img = np.array(image)
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img
            elif isinstance(image, np.ndarray):
                if image.ndim == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
            else:
                return None

            h, w = gray.shape[:2]
            if h < 50 or w < 50:
                return None

            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            if not regions:
                return 0.0

            img_area = float(h * w)
            # Estimate region area mass
            area_sum = 0.0
            for p in regions:
                if p is None or len(p) < 5:
                    continue
                try:
                    area_sum += float(cv2.contourArea(p.reshape(-1, 1, 2)))
                except Exception:
                    continue

            region_count = len(regions)
            area_ratio = area_sum / img_area if img_area > 0 else 0.0

            # Normalize metrics into 0..1 range (tuned for card-sized crops)
            def _norm(val, vmin, vmax):
                if vmax <= vmin:
                    return 0.0
                return max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))

            # Typical MSER counts on card crops are in the hundreds
            count_score = _norm(region_count, 120, 1200)
            # Area mass ratio for text/edges tends to be small but non-zero
            area_score = _norm(area_ratio, 0.008, 0.18)

            score = 0.7 * count_score + 0.3 * area_score
            return max(0.0, min(1.0, score))

        except Exception:
            return None

    def hamming_distance(self, hash1, hash2):
        """Fast Hamming distance calculation"""
        if not hash1 or not hash2:
            return 999
        
        try:
            return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
        except:
            return 999
    
    def quick_filter(self, hash1, hash2, max_dist=None):
        """
        Quick rejection filter using single channel
        Returns True if worth checking all channels
        """
        if max_dist is None:
            max_dist = getattr(self, 'quick_filter_max', 80)
        dist = self.hamming_distance(hash1, hash2)
        return dist <= max_dist

    
    def scan_game(self, game_name, r_hash, g_hash, b_hash, threshold, found_exact,
                  set_filter=None, foil_type_filter=None, rarity_filter=None):
        """
        Scan a single game (runs in thread)
        Returns list of matches from this game
        
        Args:
            set_filter: List of set codes to filter by (None = all sets)
            foil_type_filter: List of foil types (subTypeName values) to filter by (None = all types)
            rarity_filter: List of rarity codes to filter by (None = all rarities)
        """
        if found_exact.is_set():
            return []  # Another thread found exact match, abort
        
        cursor = self.get_connection()
        matches = []
        
        game_info = self.games[game_name]
        table = game_info['table']
        
        try:
            # Always load pHash data from per-game pHash DB (no fallbacks)
            if game_name in self.hash_cache:
                cards = self.hash_cache[game_name]
                self.stats['cache_hits'] += 1
            else:
                gid = str(game_info.get('id'))
                phash_db = Path(__file__).parent / 'recognition_data' / f"phash_cards_{gid}.db"
                if not phash_db.exists():
                    return []
                ph_conn = sqlite3.connect(str(phash_db))
                ph_cur = ph_conn.cursor()
                ph_cur.execute("PRAGMA table_info(cards)")
                ph_cols = [r[1] for r in ph_cur.fetchall()]
                wanted = [c for c in ('product_id', 'r_phash', 'g_phash', 'b_phash', 'grayscale_phash') if c in ph_cols]
                if 'product_id' not in wanted:
                    ph_conn.close()
                    return []
                ph_cur.execute(f"SELECT {', '.join(wanted)} FROM cards")
                rows = ph_cur.fetchall()
                ph_conn.close()
                cards = [dict(zip(wanted, row)) for row in rows]
                if self.cache_enabled:
                    self.hash_cache[game_name] = cards
            
            # Scan cards with early termination
            match_count = 0
            max_matches_per_game = 50  # Stop after finding 50 good matches per game to speed up multi-game scans
            
            candidate_ids = []
            candidate_rows = []
            for card in cards:
                if found_exact.is_set():
                    break  # Exact match found by another thread
                
                # Early exit if we have enough matches from this game
                if match_count >= max_matches_per_game:
                    break

                # Card is a dict (from pHash DB)
                product_id = card.get('product_id')
                if product_id is not None:
                    product_id = str(product_id)
                name = 'Unknown'
                number = None
                if self.use_grayscale_phash:
                    gray = card.get('grayscale_phash')
                    card_r = gray
                    card_g = gray
                    card_b = gray
                else:
                    card_r = card.get('r_phash')
                    card_g = card.get('g_phash')
                    card_b = card.get('b_phash')
                set_code = None
                rarity = None
                subTypeName = None
                market_price = None
                low_price = None

                # Quick filter on R channel first (cheapest check)
                if not self.quick_filter(r_hash, card_r, threshold * 3):
                    continue

                # Calculate full distance
                dist_r = self.hamming_distance(r_hash, card_r)
                dist_g = self.hamming_distance(g_hash, card_g)
                dist_b = self.hamming_distance(b_hash, card_b)

                avg_distance = (dist_r + dist_g + dist_b) / 3.0

                if avg_distance <= threshold:
                    confidence = max(0, 100 - (avg_distance / 256 * 100))

                    # Price fallback: prefer market_price/price, then low_price variants
                    price_value = None
                    try:
                        if market_price is not None and float(market_price) > 0:
                            price_value = float(market_price)
                        elif low_price is not None and float(low_price) > 0:
                            price_value = float(low_price)
                    except Exception:
                        price_value = None

                    candidate_ids.append(product_id)
                    candidate_rows.append({
                        'product_id': product_id,
                        'game': game_info['display_name'],
                        'name': name,
                        'number': number,
                        'set': set_code,
                        'rarity': rarity,
                        'foil_type': subTypeName,
                        'market_price': price_value,
                        'distance': avg_distance,
                        'confidence': confidence,
                        'dist_r': dist_r,
                        'dist_g': dist_g,
                        'dist_b': dist_b
                    })
                    
                    match_count += 1

                    # Exact match found!
                    if avg_distance == 0:
                        found_exact.set()
                        break
            
            # Enrich candidates with unified metadata
            meta_map = {}
            if candidate_ids and table:
                try:
                    placeholders = ','.join(['?' for _ in candidate_ids])
                    cur = cursor
                    cur.execute(
                        f"SELECT product_id, name, number, set_code, set_name, rarity, subTypeName, market_price, low_price, color "
                        f"FROM {table} WHERE product_id IN ({placeholders})",
                        candidate_ids
                    )
                    for row in cur.fetchall():
                        meta_map[str(row[0])] = {
                            'name': row[1],
                            'number': row[2],
                            'set_code': row[3],
                            'set_name': row[4],
                            'rarity': row[5],
                            'subTypeName': row[6],
                            'market_price': row[7],
                            'low_price': row[8],
                            'color': row[9]
                        }
                except Exception:
                    meta_map = {}

            for m in candidate_rows:
                pid = m.get('product_id')
                if pid is not None:
                    pid = str(pid)
                meta = meta_map.get(pid, {})
                if meta:
                    m['name'] = meta.get('name') or m['name']
                    m['number'] = meta.get('number') or m['number']
                    m['set'] = meta.get('set_code') or m.get('set')
                    m['set_name'] = meta.get('set_name')
                    m['rarity'] = meta.get('rarity') or m.get('rarity')
                    m['foil_type'] = meta.get('subTypeName') or m.get('foil_type')
                    if meta.get('color') is not None:
                        m['color'] = meta.get('color')
                    price_value = None
                    try:
                        if meta.get('market_price') is not None and float(meta.get('market_price')) > 0:
                            price_value = float(meta.get('market_price'))
                        elif meta.get('low_price') is not None and float(meta.get('low_price')) > 0:
                            price_value = float(meta.get('low_price'))
                    except Exception:
                        price_value = m.get('market_price')
                    m['market_price'] = price_value

                # Apply filters after metadata
                if set_filter and m.get('set'):
                    if m.get('set').upper() not in [s.upper() for s in set_filter]:
                        continue
                if foil_type_filter and m.get('foil_type'):
                    if m.get('foil_type') not in foil_type_filter:
                        continue
                if rarity_filter and m.get('rarity'):
                    if m.get('rarity').upper() not in [r.upper() for r in rarity_filter]:
                        continue

                matches.append(m)

            self.stats['cards_checked'] += len(cards)
        
        except sqlite3.OperationalError:
            pass
        
        return matches
    
    def scan_card(self, image, threshold=None, top_n=10, set_filter=None, foil_type_filter=None, rarity_filter=None, game_filter=None):
        """
        Multi-threaded card scanning with early termination
        
        Args:
            set_filter: List of set codes to filter by (None = all sets)
            foil_type_filter: List of foil types (subTypeName values) to filter by (None = all types)
            rarity_filter: List of rarity codes to filter by (None = all rarities)
            game_filter: Specific game name to filter by (None = all active games)
        """
        start_time = time.time()
        
        # pHash-based scanning (vector/embedding disabled)
        # Compute phash if not already computed (ResNet50 path skips this)
        if not 'r_hash' in locals():
            r_hash, g_hash, b_hash = self.compute_phash(image)

        # Optional MSER-based quality score
        mser_score = self._compute_mser_score(image)
        
        # Event for early termination when exact match found
        found_exact = threading.Event()
        
        all_matches = []
        
        # Determine effective threshold (use instance default if not provided)
        if threshold is None:
            threshold = getattr(self, 'scan_threshold', 40)

        # Submit game scans to thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.scan_game, 
                    game_name, 
                    r_hash, 
                    g_hash, 
                    b_hash, 
                    threshold,
                    found_exact,
                    set_filter,
                    foil_type_filter,
                    rarity_filter
                ): game_name 
                for game_name in self.active_games
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                matches = future.result()
                all_matches.extend(matches)
                
                # If exact match found, cancel remaining tasks
                if found_exact.is_set():
                    break
        
        # Sort by distance
        # Deduplicate matches by product_id (faster than complex tuple key)
        deduped = {}
        for m in all_matches:
            pid = m.get('product_id')
            if pid:
                if pid in deduped:
                    # Keep the one with smaller distance
                    if m['distance'] < deduped[pid]['distance']:
                        deduped[pid] = m
                else:
                    deduped[pid] = m
            else:
                # Fallback for entries without product_id
                key = (m['game'], m['name'], m['number'], m['set'])
                if key in deduped:
                    if m['distance'] < deduped[key]['distance']:
                        deduped[key] = m
                else:
                    deduped[key] = m

        all_matches = list(deduped.values())
        all_matches.sort(key=lambda x: x['distance'])

        # Blend MSER into confidence as a total score
        weight_mser = max(0.0, min(1.0, float(self.mser_weight)))
        weight_phash = max(0.0, 1.0 - weight_mser)

        for m in all_matches:
            phash_conf = float(m.get('confidence', 0) or 0)
            total_conf = (
                phash_conf * weight_phash +
                (mser_score or 0) * 100.0 * weight_mser
            )
            m['phash_confidence'] = phash_conf
            m['mser_score'] = mser_score
            m['confidence'] = total_conf
        
        # Update stats
        elapsed = time.time() - start_time
        self.stats['scans'] += 1
        self.stats['total_time'] += elapsed
        
        return all_matches[:top_n], elapsed
    
    def scan_from_file(self, image_path, threshold=10, top_n=10, set_filter=None, foil_type_filter=None, rarity_filter=None, game_filter=None):
        """Scan from image file"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[!] Failed to load image: {image_path}")
            return [], 0
        
        return self.scan_card(image, threshold, top_n, set_filter, foil_type_filter, rarity_filter, game_filter)
    
    def adaptive_scan(self, image, max_threshold=20, target_matches=5, set_filter=None, foil_type_filter=None, rarity_filter=None):
        """
        Adaptive threshold scanning
        Starts strict, gradually relaxes until target matches found
        """
        print("\n[*] Adaptive scan mode...")
        
        for threshold in range(5, max_threshold + 1, 2):
            matches, elapsed = self.scan_card(image, threshold=threshold, top_n=target_matches, set_filter=set_filter, foil_type_filter=foil_type_filter, rarity_filter=rarity_filter)
            
            print(f"    Threshold {threshold}: {len(matches)} matches in {elapsed:.2f}s")
            
            if len(matches) >= target_matches:
                return matches, elapsed
            
            if matches and matches[0]['distance'] == 0:
                # Exact match found
                return matches, elapsed
        
        # Return best we found
        return matches, elapsed
    
    def print_stats(self):
        """Print performance statistics"""
        print("\n" + "=" * 80)
        print("PERFORMANCE STATISTICS")
        print("=" * 80)
        print(f"Total scans: {self.stats['scans']}")
        if self.stats['scans'] > 0:
            print(f"Average time: {self.stats['total_time'] / self.stats['scans']:.2f}s")
            print(f"Average cards checked: {self.stats['cards_checked'] / self.stats['scans']:.0f}")
            print(f"Cache hit rate: {100 * self.stats['cache_hits'] / max(self.stats['scans'], 1):.1f}%")
        print("=" * 80)
        
        # Collection stats if enabled
        if self.collection_manager:
            print("\n")
            self.collection_manager.print_summary()
    
    def save_to_collection(self, card_info, quantity=None, condition=None, language=None, is_foil=None, prompt=None):
        """
        Save a scanned card to the collection
        
        Args:
            card_info: Card data from scan results
            quantity: Number of copies (uses default 1 if None)
            condition: Card condition (uses default if None)
            language: Language code (uses default if None)
            is_foil: Whether card is foil (uses default if None)
            prompt: Override prompt_for_details setting (True/False/None)
        
        Returns:
            Card entry if saved, None if collection disabled
        """
        if not self.collection_manager:
            print("[!] Collection manager not initialized")
            return None
        
        # Use defaults if not specified
        if quantity is None:
            quantity = 1
        if condition is None:
            condition = self.default_condition
        if language is None:
            language = self.default_language
        if is_foil is None:
            is_foil = self.default_foil
        
        try:
            entry = self.collection_manager.add_card(
                card_info, 
                quantity=quantity,
                condition=condition,
                language=language,
                is_foil=is_foil
            )
            print(f"[+] Saved to collection: {entry['name']} ({entry['sku']})")
            return entry
        except Exception as e:
            print(f"[!] Error saving to collection: {e}")
            return None
    
    def export_collection(self, format_type='both', by_game=False):
        """
        Export the collection
        
        Args:
            format_type: 'tcgtraders', 'tcgplayer', or 'both'
            by_game: Whether to separate exports by game
        
        Returns:
            Exported file path(s)
        """
        if not self.collection_manager:
            print("[!] Collection manager not initialized")
            return None
        
        if by_game:
            return self.collection_manager.export_by_game(format_type)
        else:
            results = {}
            if format_type in ['tcgtraders', 'both']:
                results['tcgtraders'] = self.collection_manager.export_tcgtraders_csv()
            if format_type in ['tcgplayer', 'both']:
                results['tcgplayer'] = self.collection_manager.export_tcgplayer_text()
            return results
    
    def close(self):
        """Close database and serial connections"""
        self.conn.close()
        if hasattr(self.local, 'conn'):
            self.local.conn.close()
        if self.ser:
            self.ser.close()
            print("[+] Serial connection closed")
    
    # =====================================================================
    # WEBCAM/REALTIME CAPTURE MODE
    # =====================================================================

    def run_realtime_mode(self, sorting_mode="color", threshold=1000000):
        """
        Run scanner in realtime webcam mode (phash-based identification)
        Automatically scans cards when detected
        """
        print("\n" + "=" * 80)
        print("REALTIME CARD SCANNER MODE")
        print("=" * 80)
        print(f"Sorting mode: {sorting_mode}")
        if sorting_mode in ["buy", "price"]:
            print(f"Price threshold: ${threshold:,.2f}")
        print(f"Inventory tracking: {self.track_inventory}")
        print(f"Collection mode: {'Enabled' if self.collection_manager else 'Disabled'}")
        print(f"Arduino: {'Connected' if self.ser else 'Disabled'}")
        print("\nKeyboard shortcuts:")
        print("  'q' - Quit")
        print("  's' - Save last scanned card to collection")
        print("  'c' - View collection stats")
        print("=" * 80)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[!] Cannot open webcam")
            return
        
        frame_count = 0
        total_processing_time = 0
        last_scanned_card = None  # Store last scanned card for 's' shortcut
        
        try:
            while True:
                frame_start = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    print("[!] Failed to grab frame")
                    break
                
                # Show camera feed
                cv2.imshow("Camera Feed", frame)
                
                display_frame = frame.copy()
                
                # Find card contour
                card_approx = self._find_card_contour(frame)
                
                if card_approx is not None:
                    # Scan the card using phash
                    card_info = self._process_card_from_contour(frame, card_approx)
                    
                    if card_info:
                        # Store for keyboard shortcut
                        last_scanned_card = card_info
                        
                        # Check inventory if enabled
                        if self.track_inventory:
                            card_info = self.check_inventory(card_info)
                        
                        if card_info != "RejectCard":
                            # Get bin assignment
                            bin_result = self.get_bin_number(card_info, sorting_mode, threshold)
                            
                            # Display result
                            self._handle_recognized_card(display_frame, card_info, bin_result)
                            
                            # Send to Arduino
                            if self.ser:
                                self.send_to_arduino(bin_result)
                        else:
                            self._handle_unrecognized_card(display_frame, card_approx, "In inventory")
                    else:
                        self._handle_unrecognized_card(display_frame, card_approx, "Card not found in database")
                    
                    cv2.imshow("Detected Card", display_frame)
                else:
                    cv2.imshow("Detected Card", display_frame)
                
                # Stats
                frame_count += 1
                frame_time = time.time() - frame_start
                total_processing_time += frame_time
                avg_time = total_processing_time / frame_count
                
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"[*] Frame {frame_count} | Time: {frame_time:.3f}s | Avg: {avg_time:.3f}s")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s') and last_scanned_card and self.collection_manager:
                    # Quick save last scanned card
                    print("\n[*] Saving to collection:")
                    print(f"    Card: {last_scanned_card.get('name', 'Unknown')}")
                    
                    if self.prompt_for_details:
                        # Prompt for each detail
                        print("    Enter details (or press Enter for defaults):")
                        
                        qty = input("    Quantity (1): ").strip() or "1"
                        cond = input("    Condition (NM/LP/MP/HP/DMG) [NM]: ").strip().upper() or "NM"
                        lang = input("    Language (EN/JP/FR/etc) [EN]: ").strip().upper() or "EN"
                        foil = input("    Foil? (Y/N) [N]: ").strip().upper() == "Y"
                        
                        cond_map = {
                            'NM': 'Near Mint',
                            'LP': 'Lightly Played', 
                            'MP': 'Moderately Played',
                            'HP': 'Heavily Played',
                            'DMG': 'Damaged'
                        }
                        
                        self.save_to_collection(
                            last_scanned_card,
                            quantity=int(qty),
                            condition=cond_map.get(cond, 'Near Mint'),
                            language=lang,
                            is_foil=foil
                        )
                    else:
                        # Auto-save with defaults (no prompts)
                        self.save_to_collection(last_scanned_card)
                        print(f"    Saved with defaults: {self.default_condition}, {self.default_language}, {'Foil' if self.default_foil else 'Normal'}")
                    
                    print("[+] Card saved! Press any key to continue scanning...\n")
                    cv2.waitKey(0)
                
                elif key == ord('c') and self.collection_manager:
                    # Show collection stats
                    print("\n" + "=" * 60)
                    stats = self.collection_manager.get_stats()
                    print(f"Collection Statistics:")
                    print(f"  Total cards: {stats['total_cards']}")
                    print(f"  Unique cards: {stats['unique_cards']}")
                    print(f"  Session cards: {stats['session_cards']}")
                    if stats['by_game']:
                        print(f"  By game:")
                        for game, count in stats['by_game'].items():
                            print(f"    {game}: {count}")
                    print("=" * 60)
                    print("[*] Press any key to continue scanning...\n")
                    cv2.waitKey(0)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n[+] Processed {frame_count} frames")
            if frame_count > 0:
                print(f"[+] Average processing time: {total_processing_time/frame_count:.3f}s")
    
    def _find_card_contour(self, frame):
        """
        Find card contour using the internal detection pipeline.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = None
            max_area = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > 10000 and area > max_area:
                        max_area = area
                        best_contour = approx
            return best_contour
        except Exception:
            return None
        return None
    
    def _process_card_from_contour(self, frame, card_approx):
        """
        Extract and process card from contour
        Returns card info dict or None
        """
        try:
            # Use the internal perspective correction
            warped = self._get_perspective_corrected_card(frame, card_approx)

            if warped is None:
                return None

            # Crop using internal WIDTH/HEIGHT (don't force a square)
            crop_w, crop_h = 745, 1043

            # Ensure we don't index out of bounds if the warped image is smaller
            h, w = warped.shape[:2]
            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)

            cropped = warped[:crop_h, :crop_w]

            # Convert to PIL for hashing (original uses per-channel pHash)
            rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Compute per-channel pHashes and print live debug output
            try:
                r_hash, g_hash, b_hash = self.compute_phash(pil_image)
                print(f"phashes: r={r_hash} g={g_hash} b={b_hash}")
                try:
                    from pathlib import Path
                    import time
                    outdir = Path(__file__).parent / 'debug_crops'
                    outdir.mkdir(parents=True, exist_ok=True)
                    shortname = f"{int(time.time())}_{(r_hash or '')[:8]}_{(g_hash or '')[:8]}_{(b_hash or '')[:8]}.png"
                    pil_image.save(outdir / shortname)
                except Exception as _e:
                    print(f"[!] Failed to save debug crop: {_e}")
            except Exception:
                r_hash = g_hash = b_hash = None

            # Scan the card using existing scan path which computes pHash identically
            # Use current scanner threshold (GUI can update this live)
            scan_threshold = getattr(self, 'scan_threshold', 40)
            matches, _ = self.scan_card(pil_image, threshold=scan_threshold, top_n=3)

            if matches and len(matches) > 0:
                card_info = matches[0]

                # Secondary MTG set symbol verification
                game_name = (card_info.get('game') or '').lower()
                if self.enable_set_symbol_check and 'magic' in game_name:
                    card_name = card_info.get('name')
                    phash_set = card_info.get('set')
                    resolved_set, resolved_conf, conf_map = self._select_set_by_symbol(cropped, card_name)
                    if resolved_set:
                        if phash_set and phash_set in conf_map:
                            phash_conf = conf_map.get(phash_set, 0.0)
                            if resolved_set != phash_set and (resolved_conf - phash_conf) < 0.08:
                                print(
                                    f"[!] Symbol match close; keeping pHash set {phash_set} "
                                    f"(symbol best={resolved_set} {resolved_conf:.3f}, phash={phash_conf:.3f})"
                                )
                            else:
                                card_info['set'] = resolved_set
                                print(f"[+] Symbol verification matched set: {resolved_set}")
                        else:
                            card_info['set'] = resolved_set
                            print(f"[+] Symbol verification matched set: {resolved_set}")
                    else:
                        print("[!] Symbol verification failed; rejecting match")
                        return None

                return card_info

            return None

        except Exception as e:
            print(f"[!] Error processing card: {e}")
            return None
    
    def _get_perspective_corrected_card(self, frame, card_approx):
        """Get perspective-corrected card image"""
        try:
            # Define card dimensions
            width = 745
            height = 1043
            
            # Get corner points
            pts = card_approx.reshape(4, 2)
            pts = sorted(pts, key=lambda point: point[1])
            top_two, bottom_two = pts[:2], pts[2:]
            top_left, top_right = sorted(top_two, key=lambda point: point[0])
            bottom_left, bottom_right = sorted(bottom_two, key=lambda point: point[0])
            rect = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            
            # Destination points
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (width, height))

            # Rotate to portrait if needed
            h, w = warped.shape[:2]
            if w > h:
                for _ in range(3):
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    h, w = warped.shape[:2]
                    if w <= h:
                        break
            
            return warped
        
        except Exception as e:
            print(f"[!] Perspective correction error: {e}")
            return None

    def _load_mtg_roi_config(self):
        if self._mtg_roi_config is not None:
            return self._mtg_roi_config
        config_path = Path(__file__).parent / 'mtg_set_roi_locations.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self._mtg_roi_config = json.load(f)
                    return self._mtg_roi_config
            except Exception:
                pass
        self._mtg_roi_config = {
            "sets": {},
            "no_symbol_sets": []
        }
        return self._mtg_roi_config

    def _verify_mtg_set_symbol(self, image_bgr, set_code, card_name=None):
        """Return True if symbol matches set_code, False if mismatch at high confidence, or None if skipped."""
        if image_bgr is None or image_bgr.size == 0 or not set_code:
            return None

        set_code = str(set_code).upper()

        config = self._load_mtg_roi_config()
        if set_code in config.get('no_symbol_sets', []):
            return None

        prior = config.get('sets', {}).get(set_code)
        if not prior:
            return None

        if self._mtg_symbols is None:
            self._mtg_symbols = load_set_symbols()

        # Normalize promo-style set codes (e.g., PPELD -> ELD) if needed
        if (not prior) or (set_code not in self._mtg_symbols):
            normalized = self._normalize_mtg_set_code(set_code, config)
            if normalized and normalized != set_code:
                print(f"[!] Normalized set code for symbol check: {set_code} -> {normalized}")
                set_code = normalized
                prior = config.get('sets', {}).get(set_code)

        if (not prior) or (set_code not in self._mtg_symbols):
            resolved = self._resolve_symbol_set_from_name(card_name, config)
            if resolved and resolved != set_code:
                print(f"[!] Resolved symbol set from name: {set_code} -> {resolved}")
                set_code = resolved
                prior = config.get('sets', {}).get(set_code)

        if not self._mtg_symbols or set_code not in self._mtg_symbols:
            return None

        h, w = image_bgr.shape[:2]
        px1 = max(0, int(prior[0] * w))
        py1 = max(0, int(prior[1] * h))
        px2 = min(w, int(prior[2] * w))
        py2 = min(h, int(prior[3] * h))
        if px2 <= px1 or py2 <= py1:
            return None

        roi = image_bgr[py1:py2, px1:px2]
        match = match_symbol_template(
            roi,
            self._mtg_symbols,
            true_set_code=set_code,
            set_only=True,
            search_all_sets=False,
            quiet=True,
            allow_ml_set=False,
            allow_ml=False
        )

        if not match:
            return None

        sym_set = match.get('set_code')
        sym_conf = float(match.get('confidence', 0.0))

        if sym_conf >= self.symbol_min_confidence and sym_set and sym_set != set_code:
            print(f"[!] Symbol mismatch: predicted={set_code} symbol={sym_set} conf={sym_conf:.3f}")
            return False

        return True

    def _select_set_by_symbol(self, image_bgr, card_name):
        """Match symbol against all printings of a card name; return (set, conf, conf_map)."""
        if image_bgr is None or image_bgr.size == 0 or not card_name:
            return None, 0.0, {}

        name = str(card_name).strip()
        if not name:
            return None, 0.0, {}

        config = self._load_mtg_roi_config()
        if self._mtg_symbols is None:
            self._mtg_symbols = load_set_symbols()

        try:
            cur = self.get_connection()
            rows = cur.execute(
                "SELECT DISTINCT set_code FROM cards_1 WHERE name = ?",
                (name,)
            ).fetchall()
        except Exception:
            return None, 0.0, {}

        if not rows:
            return None, 0.0, {}

        candidate_sets = []
        for (set_code,) in rows:
            if not set_code:
                continue
            sc = str(set_code).upper()
            if sc in config.get('no_symbol_sets', []):
                continue
            if sc in config.get('sets', {}) and sc in (self._mtg_symbols or {}):
                candidate_sets.append(sc)

        if not candidate_sets:
            return None, 0.0, {}

        h, w = image_bgr.shape[:2]
        best_set = None
        best_conf = 0.0
        conf_map = {}
        for sc in candidate_sets:
            prior = config.get('sets', {}).get(sc)
            if not prior:
                continue
            px1 = max(0, int(prior[0] * w))
            py1 = max(0, int(prior[1] * h))
            px2 = min(w, int(prior[2] * w))
            py2 = min(h, int(prior[3] * h))
            if px2 <= px1 or py2 <= py1:
                continue

            roi = image_bgr[py1:py2, px1:px2]
            match = match_symbol_template(
                roi,
                self._mtg_symbols,
                true_set_code=sc,
                set_only=True,
                search_all_sets=False,
                quiet=True,
                allow_ml_set=False,
                allow_ml=False
            )

            if not match:
                continue
            conf = float(match.get('confidence', 0.0))
            conf_map[sc] = conf
            if conf > best_conf:
                best_conf = conf
                best_set = sc

        if best_set and best_conf >= self.symbol_min_confidence:
            return best_set, best_conf, conf_map

        return None, 0.0, conf_map

    def _resolve_symbol_set_from_name(self, card_name, config):
        """Pick a base set code for symbol check based on card name and ROI/template availability."""
        if not card_name:
            return None

        name = str(card_name).strip()
        if not name:
            return None

        try:
            cur = self.get_connection()
            rows = cur.execute(
                "SELECT set_code, COUNT(*) as cnt FROM cards_1 WHERE name = ? GROUP BY set_code ORDER BY cnt DESC",
                (name,)
            ).fetchall()
        except Exception:
            return None

        if not rows:
            return None

        for set_code, _ in rows:
            if not set_code:
                continue
            sc = str(set_code).upper()
            if sc in config.get('sets', {}) and sc in (self._mtg_symbols or {}):
                return sc

        return None

    def _normalize_mtg_set_code(self, set_code, config):
        """Try to map promo-style set codes to base set codes that have ROI/templates."""
        candidates = []
        sc = str(set_code).upper()

        # Common promo prefixes
        if sc.startswith("PP") and len(sc) > 2:
            candidates.append(sc[2:])
        if sc.startswith("P") and len(sc) > 1:
            candidates.append(sc[1:])
        if sc.startswith("PR") and len(sc) > 2:
            candidates.append(sc[2:])

        # Remove non-alnum chars
        alnum = "".join(ch for ch in sc if ch.isalnum())
        if alnum != sc:
            candidates.append(alnum)

        # Dedup while preserving order
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        for cand in candidates:
            if cand in config.get('sets', {}) and cand in (self._mtg_symbols or {}):
                return cand

        return None
    
    def _handle_unrecognized_card(self, display_frame, card_approx, reason="Unknown"):
        """Display unrecognized card with reason"""
        cv2.drawContours(display_frame, [card_approx], -1, (0, 0, 255), 2)
        cv2.putText(display_frame, "Unrecognized Card", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Reason: {reason}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, "Bin: RejectCard", (10, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"[!] Card unrecognized: {reason}")
    
    def _handle_recognized_card(self, display_frame, card_info, bin_result):
        """Display recognized card with info"""
        # Draw card info
        y_pos = 30
        line_height = 25
        
        name = card_info.get('name') or card_info.get('card_name') or 'Unknown'
        game = card_info.get('game', 'Unknown')
        set_code = card_info.get('set', 'N/A')
        number = card_info.get('number', 'N/A')
        confidence = card_info.get('confidence', 0)
        price = card_info.get('market_price')
        
        cv2.putText(display_frame, f"Card: {name}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_height
        
        cv2.putText(display_frame, f"Game: {game}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(display_frame, f"Set: {set_code} #{number}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(display_frame, f"Confidence: {confidence:.1f}%", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height
        
        if price:
            cv2.putText(display_frame, f"Price: ${price:.2f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height
        
        y_pos += 15
        
        # Bin number
        bin_color = (0, 255, 0) if bin_result != "RejectCard" else (0, 0, 255)
        cv2.putText(display_frame, f"Bin: {bin_result}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bin_color, 2)
        
        print(f"[+] Recognized: {name} ({game}) -> Bin: {bin_result}")

def print_sorting_options():
    """Print available sorting modes"""
    print("\n" + "=" * 60)
    print("SORTING MODES:")
    print("=" * 60)
    print("1 - Color:      Sort by card color (W/U/B/R/G/Multicolor/Colorless/Land)")
    print("2 - CMC:        Sort by converted mana cost (1-8+)")
    print("3 - Set:        Sort by set code")
    print("4 - Price:      Sort by price (all price ranges)")
    print("5 - Type:       Sort by card type (Creature/Instant/etc)")
    print("6 - Buy mode:   Sort by price with custom threshold")
    print("=" * 60)

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Universal Card Scanner')
    parser.add_argument('image', nargs='?', help='Path to card image')
    parser.add_argument('--cache', action='store_true', help='Preload hash cache for Magic, Pokemon, YuGiOh')
    parser.add_argument('--game', '-g', action='append', help='Limit to specific game(s). Can be used multiple times. Example: -g Magic -g Pokemon')
    parser.add_argument('--set', '-s', action='append', help='Limit to specific set code(s). Can be used multiple times. Example: -s LEA -s LEB')
    parser.add_argument('--foil-type', '-f', action='append', help='Filter by foil type (subTypeName). Can be used multiple times. Example: -f Foil -f Normal')
    parser.add_argument('--rarity', '-r', action='append', help='Filter by rarity. Can be used multiple times. Example: -r M -r R')
    parser.add_argument('--list-games', action='store_true', help='List all available games and exit')
    parser.add_argument('--list-foil-types', metavar='GAME', help='List all foil types (subTypeName values) for a specific game and exit')
    parser.add_argument('--list-rarities', metavar='GAME', help='List all rarities for a specific game and exit')
    parser.add_argument('--list-sets', metavar='GAME', help='List all sets for a specific game and exit')
    parser.add_argument('--threshold', '-t', type=int, default=40, help='Match threshold (default: 40, lower = stricter)')
    parser.add_argument('--top', '-n', type=int, default=10, help='Number of top matches to show (default: 10)')
    parser.add_argument('--min-confidence', '-c', type=float, default=85.0, help='Minimum confidence (percent) for Method 1 to be considered reasonable (default: 85.0)')
    
    # NEW OPTIONS
    parser.add_argument('--realtime', action='store_true', help='Run in realtime webcam mode')
    parser.add_argument('--serial-port', help='Serial port for Arduino (e.g., COM3)')
    parser.add_argument('--baud-rate', type=int, default=9600, help='Serial baud rate (default: 9600)')
    parser.add_argument('--track-inventory', action='store_true', help='Enable inventory tracking (reject duplicates)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with menus (like original scanner)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("OPTIMIZED UNIVERSAL CARD SCANNER")
    print("=" * 80)
    
    # Initialize with 8 worker threads
    scanner = OptimizedCardScanner(
        max_workers=8,
        cache_enabled=True,
        serial_port=args.serial_port,
        baud_rate=args.baud_rate,
    )
    
    # Enable inventory tracking if requested
    if args.track_inventory:
        scanner.enable_inventory_tracking(True)
    
    # Initialize serial if port specified
    if args.serial_port:
        scanner.init_serial()
    
    # List games and exit
    if args.list_games:
        print("\nAvailable games (317 total):")
        print(f"{'ID':>4}  {'Game Name'}")
        print("-" * 80)
        for name, info in sorted(scanner.games.items(), key=lambda x: x[1]['id']):
            try:
                print(f"{info['id']:4d}. {info['display_name']}")
            except UnicodeEncodeError:
                # Handle special characters that can't be printed in console
                safe_name = info['display_name'].encode('ascii', 'replace').decode('ascii')
                print(f"{info['id']:4d}. {safe_name}")
        scanner.close()
        return
    
    # List foil types for a specific game and exit
    if args.list_foil_types:
        matched_game = None
        # Try to interpret argument as a numeric game ID first
        try:
            game_id = int(args.list_foil_types)
            for name, info in scanner.games.items():
                if info.get('id') == game_id:
                    matched_game = (name, info)
                    break
        except ValueError:
            # Try exact name or display name match
            target = args.list_foil_types.lower()
            for name, info in scanner.games.items():
                if name.lower() == target or (info.get('display_name') or '').lower() == target:
                    matched_game = (name, info)
                    break

        # If still not found, try substring matches
        if not matched_game:
            target = args.list_foil_types.lower()
            matched_games = [(n, i) for n, i in scanner.games.items() if target in n.lower() or target in (i.get('display_name') or '').lower()]
            if len(matched_games) == 1:
                matched_game = matched_games[0]
            elif len(matched_games) > 1:
                print(f"\n[!] Multiple games found matching '{args.list_foil_types}':")
                for name, info in matched_games:
                    print(f"    ID {info['id']}: {info['display_name']}")
                print(f"\nPlease use a specific game ID: --list-foil-types <ID>")
                scanner.close()
                return

        if not matched_game:
            print(f"\n[!] No game found matching: {args.list_foil_types}")
            print("    Use --list-games to see available games with IDs")
            scanner.close()
            return

        game_name, game_info = matched_game
        table = game_info['table']

        print(f"\n{'=' * 80}")
        print(f"Foil Types for: {game_info['display_name']} (Game ID: {game_info['id']})")
        print('=' * 80)

        # Query distinct subTypeName values
        try:
            cursor = scanner.get_connection()
            query = f"SELECT DISTINCT subTypeName FROM {table} WHERE subTypeName IS NOT NULL ORDER BY subTypeName"
            foil_types = cursor.execute(query).fetchall()

            if foil_types:
                print(f"\nFound {len(foil_types)} foil type(s):\n")
                for (foil_type,) in foil_types:
                    print(f"  - {foil_type}")
            else:
                print("\nNo foil types found (subTypeName column may not exist or is empty)")

        except sqlite3.OperationalError as e:
            print(f"\n[!] Error accessing foil types: {e}")
            print("    The subTypeName column may not exist for this game")

        scanner.close()
        return
    
    # List rarities for a specific game and exit
    if args.list_rarities:
        # Try to parse as game ID first
        matched_game = None
        
        try:
            game_id = int(args.list_rarities)
            # Find game by ID
            for name, info in scanner.games.items():
                if info['id'] == game_id:
                    matched_game = (name, info)
                    break
        except ValueError:
            # Not a number, try matching by name (case-insensitive, partial match)
            matched_games = [(name, info) for name, info in scanner.games.items() 
                            if args.list_rarities.lower() in name.lower()]
            if matched_games:
                if len(matched_games) == 1:
                    matched_game = matched_games[0]
                else:
                    print(f"\n[!] Multiple games found matching '{args.list_rarities}':")
                    for name, info in matched_games:
                        print(f"    ID {info['id']}: {info['display_name']}")
                    print(f"\nPlease use a specific game ID: --list-rarities <ID>")
                    scanner.close()
                    return
        
        if not matched_game:
            print(f"\n[!] No game found matching: {args.list_rarities}")
            print("    Use --list-games to see available games with IDs")
            scanner.close()
            return
        
        game_name, game_info = matched_game
        table = game_info['table']
        
        print(f"\n{'=' * 80}")
        print(f"Rarities for: {game_info['display_name']} (Game ID: {game_info['id']})")
        print('=' * 80)
        
        # Query distinct rarity values with counts
        try:
            cursor = scanner.get_connection()
            query = f"SELECT rarity, COUNT(*) as count FROM {table} WHERE rarity IS NOT NULL GROUP BY rarity ORDER BY rarity"
            rarities = cursor.execute(query).fetchall()
            
            if rarities:
                print(f"\nFound {len(rarities)} rarity type(s):\n")
                for rarity, count in rarities:
                    print(f"  {rarity:4s} - {count:,} cards")
            else:
                print("\nNo rarities found (rarity column may not exist or is empty)")
        
        except sqlite3.OperationalError as e:
            print(f"\n[!] Error accessing rarities: {e}")
            print("    The rarity column may not exist for this game")
        
        scanner.close()
        return
    
    # List sets for a specific game and exit
    if args.list_sets:
        # Try to parse as game ID first
        game_id = None
        matched_game = None
        
        try:
            game_id = int(args.list_sets)
            # Find game by ID
            for name, info in scanner.games.items():
                if info['id'] == game_id:
                    matched_game = (name, info)
                    break
        except ValueError:
            # Not a number, try matching by name (case-insensitive, partial match)
            matched_games = [(name, info) for name, info in scanner.games.items() 
                            if args.list_sets.lower() in name.lower()]
            if matched_games:
                if len(matched_games) == 1:
                    matched_game = matched_games[0]
                else:
                    print(f"\n[!] Multiple games found matching '{args.list_sets}':")
                    for name, info in matched_games:
                        print(f"    ID {info['id']}: {info['display_name']}")
                    print(f"\nPlease use a specific game ID: --list-sets <ID>")
                    scanner.close()
                    return
        
        if not matched_game:
            print(f"\n[!] No game found matching: {args.list_sets}")
            print("    Use --list-games to see available games with IDs")
            scanner.close()
            return
        
        game_name, game_info = matched_game
        
        print(f"\n{'=' * 80}")
        print(f"Sets for: {game_info['display_name']} (Game ID: {game_info['id']})")
        print('=' * 80)
        
        # Query sets table using game name
        try:
            cursor = scanner.get_connection()
            query = """
                SELECT id, name, code, total_cards
                FROM sets
                WHERE game = ?
                ORDER BY name
            """
            sets = cursor.execute(query, (game_name,)).fetchall()
            
            if sets:
                print(f"\nFound {len(sets)} sets:\n")
                print(f"{'ID':>6}  {'Set Name':60s}  Cards")
                print("-" * 80)
                for set_id, name, code, count in sets:
                    card_count = f"({count:,} cards)" if count else ""
                    print(f"{set_id:6d}. {name:60s} {card_count}")
            else:
                print("\nNo sets found in database")
        
        except sqlite3.OperationalError as e:
            print(f"\n[!] Error accessing sets data: {e}")
        
        scanner.close()
        return
    
    # =====================================================================
    # INTERACTIVE MODE
    # =====================================================================
    if args.interactive or args.realtime:
        # Show main menu
        while True:
            print("\n" + "=" * 80)
            print("CARD SCANNER - MAIN MENU")
            print("=" * 80)
            print("1 - Start Realtime Scanning (Webcam)")
            print("2 - Scan Single Image File")
            print("3 - View Collection Statistics")
            print("4 - Export Collection")
            print("5 - Clear Current Session")
            print("6 - Collection Settings")
            print("7 - Exit")
            print("=" * 80)
            
            # Show current settings
            if scanner.collection_manager:
                print(f"\nCurrent Settings:")
                print(f"  Condition: {scanner.default_condition}")
                print(f"  Language: {scanner.default_language}")
                print(f"  Foil: {'Yes' if scanner.default_foil else 'No'}")
                print(f"  Prompt Mode: {'Individual prompts' if scanner.prompt_for_details else 'Auto-save with defaults'}")
            
            menu_choice = input("\nSelect option: ").strip()
            
            if menu_choice == "1":
                # Realtime scanning mode
                print_sorting_options()
                choice = input("\nEnter the number of the sorting method: ").strip()
                sorting_modes = {
                    "1": "color",
                    "2": "mana_value",
                    "3": "set",
                    "4": "price",
                    "5": "type",
                    "6": "buy"
                }
                sorting_mode = sorting_modes.get(choice, "color")
                
                # Get price threshold for buy mode
                threshold = 1000000
                if sorting_mode in ["buy", "price"]:
                    threshold_input = input("Enter a price threshold: ").strip()
                    try:
                        threshold = float(threshold_input)
                    except:
                        threshold = 1000000
                
                # Ask about inventory tracking
                if choice in ["3", "6"]:
                    track_inv = input("Track inventory (reject duplicates)? (Y/N): ").strip().upper()
                    if track_inv == "Y":
                        scanner.enable_inventory_tracking(True)
                
                # Optionally preload cache
                if args.cache:
                    cache_games = ['Magic', 'Pokemon', 'YuGiOh']
                    scanner.preload_cache(cache_games)
                
                # Run realtime mode
                print("\n[*] Starting realtime scanner...")
                print("[*] Press 'q' to return to menu")
                if scanner.prompt_for_details:
                    print("[*] Press 's' to save card (with prompts)")
                else:
                    print("[*] Press 's' to save card (auto-save with defaults)")
                scanner.run_realtime_mode(sorting_mode=sorting_mode, 
                                        threshold=threshold)
                
            elif menu_choice == "2":
                # Scan single file
                filepath = input("\nEnter image path: ").strip()
                if os.path.exists(filepath):
                    print(f"\n[*] Scanning {filepath}...")
                    matches, elapsed = scanner.scan_from_file(filepath, threshold=args.threshold, top_n=args.top)
                    
                    if matches:
                        print(f"\n[+] Found {len(matches)} matches in {elapsed:.2f}s:\n")
                        for i, match in enumerate(matches, 1):
                            print(f"{i}. {match['name']} ({match['set_code']}) - Distance: {match['distance']}")
                        
                        # Ask to save to collection
                        save = input("\nSave to collection? (Y/N): ").strip().upper()
                        if save == "Y" and matches and scanner.collection_manager:
                            card = matches[0]  # Use best match
                            
                            if scanner.prompt_for_details:
                                # Prompt for details
                                qty = input("Quantity (default 1): ").strip() or "1"
                                cond = input("Condition (NM/LP/MP/HP/DMG, default NM): ").strip().upper() or "NM"
                                lang = input("Language (EN/JP/FR/etc, default EN): ").strip().upper() or "EN"
                                foil = input("Foil? (Y/N, default N): ").strip().upper() == "Y"
                                
                                cond_map = {'NM': 'Near Mint', 'LP': 'Lightly Played', 'MP': 'Moderately Played', 
                                           'HP': 'Heavily Played', 'DMG': 'Damaged'}
                                
                                scanner.save_to_collection(
                                    card, 
                                    quantity=int(qty),
                                    condition=cond_map.get(cond, 'Near Mint'),
                                    language=lang,
                                    is_foil=foil
                                )
                            else:
                                # Auto-save with defaults
                                scanner.save_to_collection(card)
                                print(f"[+] Saved with defaults: {scanner.default_condition}, {scanner.default_language}, {'Foil' if scanner.default_foil else 'Normal'}")
                    else:
                        print("\n[!] No matches found")
                else:
                    print(f"\n[!] File not found: {filepath}")
            
            elif menu_choice == "3":
                # View stats
                scanner.print_stats()
            
            elif menu_choice == "4":
                # Export collection
                if not scanner.collection_manager:
                    print("\n[!] Collection manager not initialized")
                    continue
                
                print("\n" + "=" * 60)
                print("EXPORT OPTIONS")
                print("=" * 60)
                print("1 - Export for TCGTraders (our website)")
                print("2 - Export for TCGPlayer")
                print("3 - Export both formats")
                print("4 - Export by game (separate files)")
                print("=" * 60)
                
                export_choice = input("\nSelect export format: ").strip()
                
                if export_choice == "1":
                    result = scanner.export_collection(format_type='tcgtraders', by_game=False)
                    print(f"\n[+] Exported to: {result.get('tcgtraders')}")
                elif export_choice == "2":
                    result = scanner.export_collection(format_type='tcgplayer', by_game=False)
                    print(f"\n[+] Exported to: {result.get('tcgplayer')}")
                elif export_choice == "3":
                    result = scanner.export_collection(format_type='both', by_game=False)
                    print(f"\n[+] TCGTraders: {result.get('tcgtraders')}")
                    print(f"[+] TCGPlayer: {result.get('tcgplayer')}")
                elif export_choice == "4":
                    print("\nExport format:")
                    print("1 - TCGTraders")
                    print("2 - TCGPlayer")
                    print("3 - Both")
                    fmt_choice = input("Select format: ").strip()
                    fmt_map = {'1': 'tcgtraders', '2': 'tcgplayer', '3': 'both'}
                    fmt = fmt_map.get(fmt_choice, 'both')
                    
                    results = scanner.export_collection(format_type=fmt, by_game=True)
                    print(f"\n[+] Exported {len(results)} files:")
                    for name, path in results.items():
                        print(f"  {name}: {path}")
            
            elif menu_choice == "5":
                # Clear session
                if scanner.collection_manager:
                    confirm = input("\nClear current session? This will not affect master collection (Y/N): ").strip().upper()
                    if confirm == "Y":
                        scanner.collection_manager.clear_session()
                else:
                    print("\n[!] Collection manager not initialized")
            
            elif menu_choice == "6":
                # Collection Settings
                if not scanner.collection_manager:
                    print("\n[!] Collection manager not initialized")
                    continue
                
                print("\n" + "=" * 80)
                print("COLLECTION SETTINGS")
                print("=" * 80)
                print(f"Current Settings:")
                print(f"  1. Condition: {scanner.default_condition}")
                print(f"  2. Language: {scanner.default_language}")
                print(f"  3. Foil: {'Yes' if scanner.default_foil else 'No'}")
                print(f"  4. Prompt Mode: {'Individual prompts' if scanner.prompt_for_details else 'Auto-save with defaults'}")
                print("  5. Return to main menu")
                print("=" * 80)
                
                setting_choice = input("\nSelect setting to change: ").strip()
                
                if setting_choice == "1":
                    print("\nCondition Options:")
                    print("  1 - Near Mint (NM)")
                    print("  2 - Lightly Played (LP)")
                    print("  3 - Moderately Played (MP)")
                    print("  4 - Heavily Played (HP)")
                    print("  5 - Damaged (DMG)")
                    cond_choice = input("Select condition: ").strip()
                    cond_map = {
                        '1': 'Near Mint',
                        '2': 'Lightly Played',
                        '3': 'Moderately Played',
                        '4': 'Heavily Played',
                        '5': 'Damaged'
                    }
                    if cond_choice in cond_map:
                        scanner.default_condition = cond_map[cond_choice]
                        print(f"[+] Default condition set to: {scanner.default_condition}")
                
                elif setting_choice == "2":
                    print("\nCommon Language Codes:")
                    print("  EN - English")
                    print("  JP - Japanese")
                    print("  FR - French")
                    print("  DE - German")
                    print("  IT - Italian")
                    print("  ES - Spanish")
                    print("  PT - Portuguese")
                    print("  KO - Korean")
                    print("  CN - Chinese")
                    lang = input("\nEnter language code (2 letters): ").strip().upper()
                    if len(lang) == 2:
                        scanner.default_language = lang
                        print(f"[+] Default language set to: {scanner.default_language}")
                    else:
                        print("[!] Invalid language code (must be 2 letters)")
                
                elif setting_choice == "3":
                    foil_choice = input("\nDefault to Foil? (Y/N): ").strip().upper()
                    scanner.default_foil = (foil_choice == "Y")
                    print(f"[+] Default foil set to: {'Yes' if scanner.default_foil else 'No'}")
                
                elif setting_choice == "4":
                    print("\nPrompt Mode Options:")
                    print("  1 - Individual prompts (ask for details on each card)")
                    print("  2 - Auto-save with defaults (no prompts, faster)")
                    mode_choice = input("Select mode: ").strip()
                    if mode_choice == "1":
                        scanner.prompt_for_details = True
                        print("[+] Prompt mode set to: Individual prompts")
                    elif mode_choice == "2":
                        scanner.prompt_for_details = False
                        print("[+] Prompt mode set to: Auto-save with defaults")
                        print(f"    Cards will be saved as: {scanner.default_condition}, {scanner.default_language}, {'Foil' if scanner.default_foil else 'Normal'}")
            
            elif menu_choice == "7":
                # Exit
                print("\n[*] Exiting...")
                scanner.print_stats()
                scanner.close()
                return
            
            else:
                print("\n[!] Invalid option")
    
    # =====================================================================
    # COMMAND LINE MODE (Original functionality)
    # =====================================================================
    
    # Filter by games if specified
    if args.game:
        valid_games = []
        for game_filter in args.game:
            # Try to parse as game ID first
            try:
                game_id = int(game_filter)
                # Find game by ID
                matched = [name for name, info in scanner.games.items() if info['id'] == game_id]
                valid_games.extend(matched)
            except ValueError:
                # Match game names (case-insensitive, partial match)
                matched = [name for name in scanner.games.keys() 
                          if game_filter.lower() in name.lower()]
                valid_games.extend(matched)
        
        if valid_games:
            scanner.set_active_games(valid_games)
            print(f"\n[+] Filtering to {len(valid_games)} game(s):")
            for game in valid_games:
                print(f"    - ID {scanner.games[game]['id']}: {scanner.games[game]['display_name']}")
        else:
            print(f"\n[!] No games matched filter: {args.game}")
            print("    Use --list-games to see available games with IDs")
            scanner.close()
            return

    # Optionally preload cache for top games (uses ~500MB RAM)
    if args.cache:
        cache_games = ['Magic', 'Pokemon', 'YuGiOh']
        # Only cache active games
        if args.game:
            cache_games = [g for g in cache_games if g in scanner.active_games]
        if cache_games:
            scanner.preload_cache(cache_games)
    
    if args.image:
        image_path = args.image
        print(f"\n[*] Scanning image: {image_path}")
        
        # Set filter if specified. Support either set codes (strings) or set IDs (integers)
        set_filter = None
        if args.set:
            set_codes = []
            set_ids = []
            for s in args.set:
                try:
                    set_ids.append(int(s))
                except ValueError:
                    # treat as set code string
                    set_codes.append(s)

            # Resolve set IDs to set codes via the sets table
            if set_ids:
                placeholders = ','.join(['?' for _ in set_ids])
                try:
                    rows = scanner.cursor.execute(
                        f"SELECT id, code, name, game FROM sets WHERE id IN ({placeholders})",
                        set_ids
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []

                if not rows:
                    print(f"[!] No sets found for IDs: {set_ids}")
                else:
                    for rid, code, name, game in rows:
                        set_codes.append(code)

            if set_codes:
                # Normalize to uppercase codes for comparisons
                set_filter = [sc.upper() for sc in set_codes]
                print(f"[+] Filtering to set(s): {', '.join(set_codes)}")
            else:
                set_filter = []
        
        # Foil type filter
        foil_type_filter = None
        if args.foil_type:
            foil_type_filter = args.foil_type
            print(f"[+] Filtering to foil type(s): {', '.join(foil_type_filter)}")
        
        # Rarity filter
        rarity_filter = None
        if args.rarity:
            rarity_filter = [r.upper() for r in args.rarity]
            print(f"[+] Filtering to rarity: {', '.join(rarity_filter)}")
        
        # Option 1: Fixed threshold scan
        print(f"\n[Method 1] Fixed threshold ({args.threshold}):")
        matches, elapsed = scanner.scan_from_file(image_path, threshold=args.threshold, top_n=args.top, set_filter=set_filter, foil_type_filter=foil_type_filter, rarity_filter=rarity_filter)
        
        print(f"\n[+] Found {len(matches)} matches in {elapsed:.2f}s")
        for i, match in enumerate(matches[:args.top], 1):
            print(f"\n{i}. {match['name']} ({match['game']})")
            print(f"   Set: {match['set']} #{match['number']}")
            print(f"   Rarity: {match.get('rarity', 'N/A')}")
            print(f"   Foil Type: {match.get('foil_type', 'N/A')}")
            print(f"   Unique ID: {match.get('unique_id', 'N/A')}")
            price = match.get('market_price')
            if price:
                print(f"   Market Price: ${price:.2f}")
            else:
                print(f"   Market Price: N/A")
            print(f"   Confidence: {match['confidence']:.1f}%")
            print(f"   Distance: {match['distance']:.2f}")
        
        # Option 2: Adaptive threshold scan — only run when Method 1 didn't find a "reasonable" match
        # "Reasonable" is defined by --min-confidence (default 85%). Also keep the previous behavior of
        # skipping adaptive scanning when a set filter is provided (to avoid expensive broad relax).
        top_confidence = matches[0]['confidence'] if matches else 0.0
        print(f"\n[+] Top match confidence (Method 1): {top_confidence:.1f}%")

        if top_confidence >= args.min_confidence:
            print(f"\n[+] Top match meets min-confidence ({args.min_confidence:.1f}%), skipping adaptive scan.")
        else:
            if not args.set:
                print("\n\n[Method 2] Adaptive threshold (running because Method 1 was not confident enough):")
                image = cv2.imread(image_path)
                adapt_matches, adapt_elapsed = scanner.adaptive_scan(image, max_threshold=20, target_matches=5, set_filter=set_filter, foil_type_filter=foil_type_filter, rarity_filter=rarity_filter)
                
                print(f"\n[+] Adaptive best matches:")
                for i, match in enumerate(adapt_matches[:5], 1):
                    price = match.get('market_price')
                    price_str = f"${price:.2f}" if price else "N/A"
                    print(f"{i}. {match['name']} - {match['confidence']:.1f}% ({match['distance']:.2f}) - Rarity: {match.get('rarity', 'N/A')} - Price: {price_str}")
            else:
                print("\n[+] Set filter present — adaptive scan skipped to keep results focused on the selected set(s).")
        
        scanner.print_stats()
    else:
        print("\nUsage:")
        print("  python optimized_scanner.py <image_path> [options]")
        print("\nModes:")
        print("  --interactive        Interactive mode with sorting menu (like original)")
        print("  --realtime           Realtime webcam scanning mode")
        print("\nOptions:")
        print("  --cache              Preload hash cache for Magic, Pokemon, YuGiOh (faster but uses RAM)")
        print("  -g, --game GAME      Limit to specific game(s) by ID or name. Can be used multiple times.")
        print("  -s, --set SET        Limit to specific set code(s) or ID(s). Can be used multiple times.")
        print("  -f, --foil-type TYPE Filter by foil type (subTypeName). Can be used multiple times.")
        print("  -r, --rarity RARITY  Filter by rarity code. Can be used multiple times.")
        print("  -t, --threshold N    Match threshold (default: 10, lower = stricter)")
        print("  -n, --top N          Number of top matches to show (default: 10)")
        print("  --serial-port PORT   Serial port for Arduino (e.g., COM3)")
        print("  --baud-rate BAUD     Serial baud rate (default: 9600)")
        print("  --track-inventory    Enable inventory tracking (reject duplicates)")
        print("  --list-games         List all available games with their IDs")
        print("  --list-sets GAME     List all sets for a specific game (by ID or name)")
        print("  --list-foil-types GAME   List all foil types for a specific game")
        print("  --list-rarities GAME     List all rarities for a specific game")
        print("\nExamples:")
        print("  python optimized_scanner.py --interactive --serial-port COM3")
        print("  python optimized_scanner.py --realtime --cache --track-inventory")
        print("  python optimized_scanner.py card.jpg --cache")
        print("  python optimized_scanner.py card.jpg -g 167              # Magic by ID")
        print("  python optimized_scanner.py card.jpg -g Magic -g Pokemon # By name")
        print("  python optimized_scanner.py card.jpg -g 311 -s 1234      # YuGiOh set by ID")
        print("  python optimized_scanner.py card.jpg -f Foil -r M        # Foil mythic rares only")
        print("  python optimized_scanner.py card.jpg -f Normal -r R -r M # Normal rares/mythics")
        print("  python optimized_scanner.py --list-games                 # Show all games with IDs")
        print("  python optimized_scanner.py --list-sets 167              # Magic sets by game ID")
        print("  python optimized_scanner.py --list-rarities Magic        # Show Magic rarities")
        print("  python optimized_scanner.py --list-foil-types 167        # Show Magic foil types")
    
    scanner.close()

if __name__ == '__main__':
    main()

