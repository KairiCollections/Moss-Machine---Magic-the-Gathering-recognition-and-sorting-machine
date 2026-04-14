#!/usr/bin/env python3
"""
Magic the Gathering Symbol Recognition Tool

This tool helps identify set symbols on MTG cards by:
1. Loading card images from the database (product_id.jpg)
2. Allowing users to click two points to define a region of interest
3. Matching the region against set symbol templates
4. Determining the set code based on the symbol match

Usage:
  python mtg_symbol_recognizer.py                    # Interactive mode (single card)
  python mtg_symbol_recognizer.py --batch <SET>     # Batch test mode for a set
  python mtg_symbol_recognizer.py <image_path>     # Test specific image
"""

import cv2
import numpy as np
import sqlite3
from pathlib import Path
import os
import sys
import argparse
import json
try:
    import torch
    import torch.nn as nn
    from torchvision import models
    _TORCH_AVAILABLE = True
except BaseException:
    _TORCH_AVAILABLE = False

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "unified_card_database.db"  # Root DB, not instance
VECTORS_PATH = PROJECT_ROOT / "Magic the Gathering Vectors" / "Set symbols" / "Physical"
TEMPLATES_BASE_PATH = PROJECT_ROOT / "mtg_symbol_templates"
TEMPLATES_PATH = TEMPLATES_BASE_PATH / "max150"
IMAGE_BASE_PATH = PROJECT_ROOT / "static"
MAGIC_TABLE = "cards_1"  # Magic the Gathering is game_id 1
ROI_CONFIG_PATH = PROJECT_ROOT / "mtg_set_roi_locations.json"
MODEL_PATH = PROJECT_ROOT / "models" / "rarity_cnn.pt"
SET_MODEL_PATH = PROJECT_ROOT / "models" / "set_cnn.pt"

# Global variables for ROI selection
points = []
roi_coords = None  # Stored ROI coordinates for batch mode
image_display = None
window_name = "MTG Symbol Recognition - Click 2 points to define ROI"

# ML model cache
_RARITY_MODEL = None
_RARITY_CLASS_MAP = None
_RARITY_IMAGE_SIZE = 64
_SET_MODEL = None
_SET_CLASS_MAP = None
_SET_IMAGE_SIZE = 64
_SET_SYMBOLS = None
_TEMPLATE_FEATURE_CACHE = {}



def get_random_magic_card():
    """Get a random Magic the Gathering card from the database"""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    
    # Get a random card with an image and set_code
    cur.execute(f"""
        SELECT product_id, name, set_code, rarity 
        FROM {MAGIC_TABLE} 
        WHERE has_image = 1 
        AND set_code IS NOT NULL
        AND set_code != ''
        ORDER BY RANDOM() 
        LIMIT 1
    """)
    
    result = cur.fetchone()
    conn.close()
    
    if result:
        return {
            'product_id': result[0],
            'name': result[1],
            'set_code': result[2],
            'rarity': result[3]
        }
    return None


def get_cards_by_set_and_rarity(set_code, rarity=None, limit=10):
    """Get cards from a specific set, optionally filtered by rarity"""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    
    if rarity:
        cur.execute(f"""
            SELECT product_id, name, set_code, rarity 
            FROM {MAGIC_TABLE} 
            WHERE set_code = ? 
            AND rarity = ?
            AND has_image = 1 
            ORDER BY RANDOM() 
            LIMIT ?
        """, (set_code, rarity, limit))
    else:
        cur.execute(f"""
            SELECT product_id, name, set_code, rarity 
            FROM {MAGIC_TABLE} 
            WHERE set_code = ? 
            AND has_image = 1 
            ORDER BY rarity, RANDOM()
            LIMIT ?
        """, (set_code, limit))
    
    results = cur.fetchall()
    conn.close()
    
    cards = []
    for result in results:
        cards.append({
            'product_id': result[0],
            'name': result[1],
            'set_code': result[2],
            'rarity': result[3]
        })
    return cards


def find_card_image(product_id):
    """Find the card image file by product_id"""
    # Search in common locations for Magic (game_id 1)
    search_paths = [
        IMAGE_BASE_PATH / "Originals" / f"{product_id}.jpg",
        PROJECT_ROOT / "Data" / "1" / f"{product_id}.jpg",
        PROJECT_ROOT / "Collection" / f"{product_id}.jpg",
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    # Search recursively in static folder as fallback
    for path in IMAGE_BASE_PATH.rglob(f"{product_id}.jpg"):
        return path
    
    return None


def load_set_symbols(force_reload=False):
    """Load all available set symbol templates (cached per process)."""
    global _SET_SYMBOLS
    if _SET_SYMBOLS is not None and not force_reload:
        return _SET_SYMBOLS

    symbols = {}

    # Prefer pre-rendered PNG templates if available
    if TEMPLATES_PATH.exists():
        print(f"Loading templates from: {TEMPLATES_PATH}")
        for set_folder in TEMPLATES_PATH.iterdir():
            if set_folder.is_dir():
                set_code = set_folder.name
                symbols[set_code] = {}

                for symbol_file in set_folder.glob("*.png"):
                    rarity = symbol_file.stem  # C, R, M, etc.
                    symbols[set_code][rarity] = symbol_file

        print(f"Loaded {len(symbols)} set symbol templates (PNG)")
        _SET_SYMBOLS = symbols
        return _SET_SYMBOLS

    # Fallback to SVG vectors
    if not VECTORS_PATH.exists():
        print(f"Vectors path not found: {VECTORS_PATH}")
        _SET_SYMBOLS = symbols
        return _SET_SYMBOLS

    print(f"Loading templates from: {VECTORS_PATH}")
    for set_folder in VECTORS_PATH.iterdir():
        if set_folder.is_dir():
            set_code = set_folder.name
            symbols[set_code] = {}

            for symbol_file in set_folder.glob("*.svg"):
                rarity = symbol_file.stem  # C, R, M, etc.
                symbols[set_code][rarity] = symbol_file

            # Also load PNG files if they exist
            for symbol_file in set_folder.glob("*.png"):
                rarity = symbol_file.stem
                if rarity not in symbols[set_code]:
                    symbols[set_code][rarity] = symbol_file

    print(f"Loaded {len(symbols)} set symbol templates")
    _SET_SYMBOLS = symbols
    return _SET_SYMBOLS


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for ROI selection"""
    global points, image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
        # Draw the point
        cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
        
        # If we have 2 points, draw the rectangle
        if len(points) == 2:
            cv2.rectangle(image_display, points[0], points[1], (0, 255, 0), 2)
        
        cv2.imshow(window_name, image_display)


def load_roi_config():
    """Load ROI configuration from JSON file"""
    if ROI_CONFIG_PATH.exists():
        try:
            with open(ROI_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Return default config
    return {
        "comment": "ROI coordinates for MTG set symbols",
        "default": [0.83, 0.57, 0.92, 0.61],
        "sets": {}
    }


def save_roi_config(config):
    """Save ROI configuration to JSON file"""
    try:
        with open(ROI_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving ROI config: {e}")
        return False


def get_roi_for_set(set_code):
    """Get ROI coordinates for a specific set, or default if not found"""
    config = load_roi_config()
    
    # Check if set has no symbol
    if set_code in config.get('no_symbol_sets', []):
        print(f"  Set {set_code} marked as NO SYMBOL - skipping")
        return None
    
    if set_code in config.get('sets', {}):
        roi = config['sets'][set_code]
        print(f"  Using saved ROI for {set_code}: {roi}")
        return tuple(roi)
    else:
        roi = config.get('default', [0.83, 0.57, 0.92, 0.61])
        print(f"  Using default ROI (no saved location for {set_code}): {roi}")
        return tuple(roi)


def save_roi_for_set(set_code, roi_coords):
    """Save ROI coordinates for a specific set"""
    config = load_roi_config()
    
    if 'sets' not in config:
        config['sets'] = {}
    
    config['sets'][set_code] = list(roi_coords)
    
    if save_roi_config(config):
        print(f"  [OK] Saved ROI for {set_code}: {roi_coords}")
        return True
    return False


def extract_roi(image, pt1, pt2):
    """Extract the region of interest from the image"""
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    
    return image[y1:y2, x1:x2]


def _expand_prior(prior, margin=0.05):
    """Expand a relative prior box by a margin (clamped to 0..1)."""
    if not prior:
        return None
    x1, y1, x2, y2 = prior
    x1 = max(0.0, x1 - margin)
    y1 = max(0.0, y1 - margin)
    x2 = min(1.0, x2 + margin)
    y2 = min(1.0, y2 + margin)
    return (x1, y1, x2, y2)


def _crop_to_largest_contour(gray, pad_ratio=0.08):
    """Crop grayscale image to the largest contour based on Otsu threshold."""
    if gray is None or gray.size == 0:
        return gray

    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            return gray

        pad_x = int(w * pad_ratio)
        pad_y = int(h * pad_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_y)

        cropped = gray[y1:y2, x1:x2]
        return cropped if cropped.size else gray
    except Exception:
        return gray


def mser_symbol_bbox(gray, min_area_ratio=0.00005, max_area_ratio=0.02, prior=None):
    """Return a tight bbox around MSER regions filtered by size/prior.

    Args:
        gray: Grayscale image
        min_area_ratio: Min region area / image area
        max_area_ratio: Max region area / image area
        prior: Optional (x1,y1,x2,y2) in relative coords (0..1) to constrain region centers
    Returns:
        (x1,y1,x2,y2) or None
    """
    if gray is None or gray.size == 0:
        return None

    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return None

    img_area = float(h * w)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    if not regions:
        return None

    prior_expanded = _expand_prior(prior, margin=0.05) if prior else None
    if prior_expanded:
        px1, py1, px2, py2 = prior_expanded
        px1, py1 = px1 * w, py1 * h
        px2, py2 = px2 * w, py2 * h

    xs = []
    ys = []
    for p in regions:
        if p is None or len(p) < 5:
            continue
        cnt = p.reshape(-1, 1, 2).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue

        area_ratio = area / img_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        if prior_expanded:
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            if not (px1 <= cx <= px2 and py1 <= cy <= py2):
                continue

        xs.extend([x, x + bw])
        ys.extend([y, y + bh])

    if not xs or not ys:
        return None

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def mser_extract_symbol_roi(image, prior=None, min_area_ratio=0.00005, max_area_ratio=0.02):
    """Extract symbol ROI using MSER region aggregation."""
    if image is None or image.size == 0:
        return None, None

    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If prior provided, crop to that region first to reduce background noise
    if prior:
        h, w = gray_full.shape[:2]
        x1 = max(0, int(prior[0] * w))
        y1 = max(0, int(prior[1] * h))
        x2 = min(w, int(prior[2] * w))
        y2 = min(h, int(prior[3] * h))
        if x2 > x1 and y2 > y1:
            gray = gray_full[y1:y2, x1:x2]
            bbox_local = mser_symbol_bbox(
                gray,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                prior=None
            )
            if bbox_local:
                lx1, ly1, lx2, ly2 = bbox_local
                bbox = (lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1)
            else:
                bbox = None
        else:
            bbox = None
    else:
        bbox = mser_symbol_bbox(
            gray_full,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            prior=None
        )

    if not bbox:
        return None, None

    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    return roi, bbox


def test_mser_on_templates(templates_root, max_sets=None, max_per_set=0, output_dir=None):
    """Run MSER detection on mtg_symbol_templates and report results."""
    templates_root = Path(templates_root)
    if not templates_root.exists():
        print(f"Templates path not found: {templates_root}")
        return 1

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    set_folders = [p for p in templates_root.iterdir() if p.is_dir()]
    set_folders.sort(key=lambda p: p.name)
    if max_sets:
        set_folders = set_folders[:max_sets]

    total = 0
    found = 0
    for set_folder in set_folders:
        symbol_files = sorted(set_folder.glob("*.png"))
        if max_per_set and max_per_set > 0:
            symbol_files = symbol_files[:max_per_set]

        for symbol_file in symbol_files:
            img = load_symbol_as_image(symbol_file, target_size=None)
            total += 1
            if img is None:
                continue

            # Templates are mostly symbol-only; allow much larger region ratios
            roi, bbox = mser_extract_symbol_roi(
                img,
                prior=None,
                min_area_ratio=0.001,
                max_area_ratio=0.95
            )
            if bbox:
                found += 1

            if output_dir:
                vis = img.copy()
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
                out_name = f"{set_folder.name}_{symbol_file.stem}.png"
                cv2.imwrite(str(output_dir / out_name), vis)

    hit_rate = (found / total * 100.0) if total > 0 else 0.0
    print(f"MSER templates test: {found}/{total} found ({hit_rate:.1f}%)")
    if output_dir:
        print(f"Debug output: {output_dir}")
    return 0


def detect_rarity_by_color(roi, quiet=False):
    """
    Detect MTG card rarity by analyzing the symbol color
    Returns: list of likely rarity codes ['C', 'U', 'R', 'M'] in order of confidence
    
    MTG Rarity Colors:
    - Common: Black/dark gray
    - Uncommon: Silver/light gray
    - Rare: Gold/yellow
    - Mythic: Red-orange
    """
    if roi is None or roi.size == 0:
        return None
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate average color values
    avg_bgr = np.mean(roi, axis=(0, 1))
    avg_hsv = np.mean(hsv, axis=(0, 1))
    
    # Also check the center region (most likely to be pure symbol color)
    h, w = roi.shape[:2]
    center_roi = roi[h//4:3*h//4, w//4:3*w//4]
    if center_roi.size > 0:
        center_bgr = np.mean(center_roi, axis=(0, 1))
        center_hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        center_hsv_avg = np.mean(center_hsv, axis=(0, 1))
    else:
        center_bgr = avg_bgr
        center_hsv_avg = avg_hsv
    
    # Extract values
    b, g, r = avg_bgr
    hue, sat, val = avg_hsv
    center_hue, center_sat, center_val = center_hsv_avg
    
    brightness = (r + g + b) / 3
    
    # Scoring system for each rarity
    scores = {
        'C': 0,  # Common (black/dark)
        'U': 0,  # Uncommon (silver/light gray)
        'R': 0,  # Rare (gold/yellow)
        'M': 0   # Mythic (red-orange)
    }
    
    # Common: Dark colors, low brightness, low saturation
    if brightness < 100:
        scores['C'] += 4
    elif brightness < 130:
        scores['C'] += 3
    elif brightness < 150:
        scores['C'] += 2
    elif brightness < 165:
        scores['C'] += 1
    
    if sat < 50 and brightness < 150:  # Low saturation AND dark
        scores['C'] += 2
    
    # Uncommon: Light gray/silver - high brightness, low saturation
    if brightness > 180:
        scores['U'] += 4
    elif brightness > 160:
        scores['U'] += 3
    elif brightness > 145:
        scores['U'] += 2
    
    if sat < 70 and brightness > 145:  # Gray but bright
        scores['U'] += 2
    
    # Rare: Gold/yellow - high saturation, yellow hue (20-40 in OpenCV)
    if 15 < hue < 45 or 15 < center_hue < 45:  # Yellow range
        scores['R'] += 3
    
    if sat > 80:  # High saturation
        scores['R'] += 2
    elif sat > 50:
        scores['R'] += 1
    
    if 140 < brightness < 220:  # Mid-high brightness
        scores['R'] += 1
    
    # Mythic: Red-orange - red hue (0-10 or 170-180 in OpenCV)
    if hue < 15 or hue > 165 or center_hue < 15 or center_hue > 165:  # Red range
        scores['M'] += 3
    
    if sat > 100:  # Very high saturation for red
        scores['M'] += 2
    elif sat > 70:
        scores['M'] += 1
    
    if r > (g + b) / 2:  # More red than green+blue
        scores['M'] += 2
    
    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if not quiet:
        print(f"\n  Color Analysis:")
        print(f"    Brightness: {brightness:.1f}, Hue: {hue:.1f}, Saturation: {sat:.1f}")
        print(f"    Rarity scores: C={scores['C']}, U={scores['U']}, R={scores['R']}, M={scores['M']}")
        print(f"    Predicted rarities: {[r[0] for r in ranked if r[1] > 0]}")
    
    # Return rarities with non-zero scores, in order
    likely_rarities = [r[0] for r in ranked if r[1] > 0]

    # Determine confidence margin between top scores
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    confidence_margin = top_score - second_score

    # If no clear winner or scores are tied, return top 2-3
    if not likely_rarities or ranked[0][1] == ranked[1][1]:
        return [r[0] for r in ranked[:3] if r[1] >= ranked[2][1]], confidence_margin

    return likely_rarities, confidence_margin


def apply_clahe(gray):
    """Apply CLAHE to improve local contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def auto_canny(gray, sigma=0.33):
    """Compute Canny thresholds based on median intensity."""
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


if _TORCH_AVAILABLE:
    class SmallCNN(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
else:
    class SmallCNN:  # type: ignore[no-redef]
        """Stub when torch is not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for SmallCNN but is not installed.")


def load_rarity_model():
    """Load the rarity CNN model if available."""
    global _RARITY_MODEL, _RARITY_CLASS_MAP, _RARITY_IMAGE_SIZE
    if _RARITY_MODEL is not None:
        return _RARITY_MODEL

    if not _TORCH_AVAILABLE or not MODEL_PATH.exists():
        return None

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    _RARITY_CLASS_MAP = idx_to_class
    _RARITY_IMAGE_SIZE = checkpoint.get("image_size", 64)

    model = SmallCNN(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _RARITY_MODEL = model
    return _RARITY_MODEL


def predict_rarity_ml(roi):
    """Predict rarity with the CNN model. Returns (label, confidence) or (None, 0)."""
    model = load_rarity_model()
    if model is None:
        return None, 0.0

    if roi is None or roi.size == 0:
        return None, 0.0

    # Prepare input
    img = cv2.resize(roi, (_RARITY_IMAGE_SIZE, _RARITY_IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # normalize like training
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy().flatten()

    idx = int(np.argmax(probs))
    label = _RARITY_CLASS_MAP.get(idx, None)
    confidence = float(probs[idx])
    return label, confidence


def load_set_model():
    """Load the set CNN model if available."""
    global _SET_MODEL, _SET_CLASS_MAP, _SET_IMAGE_SIZE
    if _SET_MODEL is not None:
        return _SET_MODEL

    if not _TORCH_AVAILABLE or not SET_MODEL_PATH.exists():
        return None

    checkpoint = torch.load(SET_MODEL_PATH, map_location="cpu")
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    _SET_CLASS_MAP = idx_to_class
    _SET_IMAGE_SIZE = checkpoint.get("image_size", 64)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _SET_MODEL = model
    return _SET_MODEL


def predict_set_ml(roi):
    """Predict set with the CNN model. Returns (set_code, confidence) or (None, 0)."""
    model = load_set_model()
    if model is None:
        return None, 0.0

    if roi is None or roi.size == 0:
        return None, 0.0

    img = cv2.resize(roi, (_SET_IMAGE_SIZE, _SET_IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy().flatten()

    idx = int(np.argmax(probs))
    label = _SET_CLASS_MAP.get(idx, None)
    confidence = float(probs[idx])
    return label, confidence


def load_symbol_as_image(symbol_path, target_size=None):
    """Load a symbol (SVG or PNG) and convert to image at target size (optional)"""
    try:
        if symbol_path.suffix.lower() == '.svg':
            # Check if we have a cached PNG version
            cache_dir = PROJECT_ROOT / "mtg_symbol_cache"
            cache_dir.mkdir(exist_ok=True)
            
            # Create cache filename: setcode_rarity.png
            cache_name = f"{symbol_path.parent.name}_{symbol_path.stem}.png"
            cache_path = cache_dir / cache_name
            
            if cache_path.exists():
                # Load from cache
                img = cv2.imread(str(cache_path))
                if img is not None:
                    if target_size:
                        img = cv2.resize(img, target_size)
                    return img
            
            # Try ImageMagick first (if available)
            try:
                import subprocess
                result = subprocess.run(
                    ['magick', 'convert', 
                     '-background', 'white',
                     '-density', '300',
                     str(symbol_path), 
                     str(cache_path)],
                    capture_output=True,
                    timeout=5
                )
                
                if result.returncode == 0 and cache_path.exists():
                    img = cv2.imread(str(cache_path))
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        return img
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass  # ImageMagick not available or timeout
            
            # Fallback: Try svglib + reportlab (won't work without Cairo on Windows)
            try:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPM
                from PIL import Image
                from io import BytesIO
                
                drawing = svg2rlg(str(symbol_path))
                if drawing:
                    png_data = renderPM.drawToString(drawing, fmt='PNG')
                    pil_img = Image.open(BytesIO(png_data))
                    
                    if pil_img.mode == 'RGBA':
                        bg = Image.new('RGB', pil_img.size, (255, 255, 255))
                        bg.paste(pil_img, mask=pil_img.split()[3])
                        pil_img = bg
                    elif pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    if target_size:
                        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(cache_path), img)
                    return img
            except:
                pass
            
            return None
        else:
            # Load PNG/JPG directly
            img = cv2.imread(str(symbol_path))
            if img is not None:
                if target_size:
                    img = cv2.resize(img, target_size)
                return img
    except Exception as e:
        return None


def _get_template_features(symbol_path):
    """Load and cache template preprocessing for faster matching."""
    key = str(symbol_path)
    if key in _TEMPLATE_FEATURE_CACHE:
        return _TEMPLATE_FEATURE_CACHE[key]

    template = load_symbol_as_image(symbol_path, target_size=None)
    if template is None:
        _TEMPLATE_FEATURE_CACHE[key] = None
        return None

    try:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template_gray = apply_clahe(template_gray)
        template_gray = _crop_to_largest_contour(template_gray)

        template_edges = auto_canny(template_gray)
        _, template_thresh = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        template_thresh_inv = cv2.bitwise_not(template_thresh)

        contours_template, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_template_inv, _ = cv2.findContours(template_thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = (template_gray, template_edges, template_thresh, template_thresh_inv, contours_template, contours_template_inv)
        _TEMPLATE_FEATURE_CACHE[key] = features
        return features
    except Exception:
        _TEMPLATE_FEATURE_CACHE[key] = None
        return None


def match_symbol_template(roi, symbols, true_set_code=None, set_only=False, search_all_sets=False, quiet=False, allow_ml_set=True, allow_ml=True, score_mode="max"):
    """
    Match the ROI against available symbol templates using template matching
    """
    if not quiet:
        print(f"\nROI extracted: {roi.shape}")
        print(f"True set code: {true_set_code}")
    
    # Save the ROI for inspection (only in verbose mode)
    if not quiet:
        roi_path = PROJECT_ROOT / "roi_extracted.jpg"
        cv2.imwrite(str(roi_path), roi)
        print(f"Saved ROI to: {roi_path}")
    
    # Determine which sets to search
    search_sets = {}
    if search_all_sets:
        search_sets = symbols
        if not quiet:
            print(f"Searching across ALL {len(symbols)} sets to find best match...")
            print(f"(This may take a moment...)")
    elif true_set_code and true_set_code in symbols:
        # Found the exact set - only search this one
        search_sets[true_set_code] = symbols[true_set_code]
        if not quiet:
            print(f"\nTrying to match against {true_set_code} symbols: {list(search_sets[true_set_code].keys())}")
    else:
        # Set not found in templates - try ML-only fallback if allowed
        if allow_ml:
            ml_rarity, ml_rarity_conf = predict_rarity_ml(roi)
            ml_set, ml_set_conf = predict_set_ml(roi)
            result_set = ml_set if ml_set else true_set_code
            result_rarity = ml_rarity
            result_conf = float(ml_set_conf) if ml_set else 0.0
            if not quiet:
                print(f"\n  ML-only (no templates): set={result_set} ({ml_set_conf:.2f}), rarity={result_rarity} ({ml_rarity_conf:.2f})")
            if result_conf > 0 or result_rarity:
                return {'set_code': result_set, 'rarity': result_rarity, 'confidence': result_conf, 'best_score': result_conf, 'margin': None}
        if not quiet:
            if true_set_code:
                print(f"\n[NO] Set code '{true_set_code}' not found in available templates")
                print(f"   Cannot perform symbol matching without set templates.")
                print(f"   Available sets: {len(symbols)} (but not including '{true_set_code}')")
            else:
                print(f"\n[NO] No set code provided")
        return None
    
    # Rarity mapping
    rarity_names = {
        'C': 'Common',
        'U': 'Uncommon',
        'R': 'Rare',
        'M': 'Mythic',
        'S': 'Special',
        'T': 'Timeshifted',
        '80': 'Special_80',
        'WM': 'Special_WM'
    }
    
    # Prepare ROI base grayscale
    roi_gray_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray_base = apply_clahe(roi_gray_base)
    roi_gray_base = _crop_to_largest_contour(roi_gray_base)

    best_match = None
    best_score = 0
    best_set = None
    results = []

    # ML set prediction (preferred for set-only mode when true set is unknown)
    if allow_ml and allow_ml_set and set_only and not true_set_code:
        ml_set, ml_set_conf = predict_set_ml(roi)
        if ml_set and ml_set_conf >= 0.70:
            if not quiet:
                print(f"\n  ML Set: {ml_set} (confidence {ml_set_conf:.2f})")
            return {'set_code': ml_set, 'rarity': None, 'confidence': ml_set_conf, 'best_score': ml_set_conf}

    # ML rarity prediction (preferred when confident)
    ml_rarity, ml_confidence = (None, 0.0)
    if allow_ml:
        ml_rarity, ml_confidence = predict_rarity_ml(roi)

    # Detect rarity by color to narrow down search
    likely_rarities, rarity_confidence = detect_rarity_by_color(roi, quiet=quiet)

    if set_only:
        # For set-only detection, only filter by ML rarity if confidence is high
        if ml_rarity and ml_confidence >= 0.70:
            if not quiet:
                print(f"\n  ML Rarity: {ml_rarity} (confidence {ml_confidence:.2f})")
            likely_rarities = [ml_rarity[0]]
        else:
            likely_rarities = None
    else:
        if ml_rarity and ml_confidence >= 0.70:
            if not quiet:
                print(f"\n  ML Rarity: {ml_rarity} (confidence {ml_confidence:.2f})")
            likely_rarities = [ml_rarity[0]]  # use first letter
        elif likely_rarities and rarity_confidence >= 2:
            # If color detection is confident, only keep top rarity
            likely_rarities = [likely_rarities[0]]
    
    # Try each set and rarity symbol
    total_checked = 0
    for set_code, set_symbols in search_sets.items():
        for rarity_code, symbol_path in set_symbols.items():
            # Filter by detected rarity if available
            if likely_rarities:
                # Check if this rarity code matches any of the likely rarities
                rarity_letter = rarity_code[0] if rarity_code else ''
                if rarity_letter not in likely_rarities:
                    # Skip templates that don't match the color-detected rarity
                    continue
            
            features = _get_template_features(symbol_path)
            if features is None:
                continue

            total_checked += 1
            template_gray, template_edges, template_thresh, template_thresh_inv, contours_template, contours_template_inv = features

            # Resize ROI to match template size (preserve template aspect ratio)
            th, tw = template_gray.shape[:2]
            if th <= 0 or tw <= 0:
                continue

            roi_gray = cv2.resize(roi_gray_base, (tw, th))

            # Apply edge detection to focus on symbol shape, not background
            roi_edges = auto_canny(roi_gray)
            # Apply binary thresholding to separate symbol from background
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Inverted versions (symbols can be light on dark or vice versa)
            roi_thresh_inv = cv2.bitwise_not(roi_thresh)

            # Method 1: Edge-based template matching (compares shapes)
            result_edge = cv2.matchTemplate(roi_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            score_edge = result_edge[0][0]

            # Method 2: Threshold-based matching (compares foreground/background)
            score_thresh = max(
                cv2.matchTemplate(roi_thresh, template_thresh, cv2.TM_CCOEFF_NORMED)[0][0],
                cv2.matchTemplate(roi_thresh_inv, template_thresh, cv2.TM_CCOEFF_NORMED)[0][0],
                cv2.matchTemplate(roi_thresh, template_thresh_inv, cv2.TM_CCOEFF_NORMED)[0][0],
                cv2.matchTemplate(roi_thresh_inv, template_thresh_inv, cv2.TM_CCOEFF_NORMED)[0][0]
            )

            # Method 2b: SQDIFF-based matching (lower is better; convert to score)
            score_sqdiff = max(
                1.0 - cv2.matchTemplate(roi_thresh, template_thresh, cv2.TM_SQDIFF_NORMED)[0][0],
                1.0 - cv2.matchTemplate(roi_thresh_inv, template_thresh, cv2.TM_SQDIFF_NORMED)[0][0],
                1.0 - cv2.matchTemplate(roi_thresh, template_thresh_inv, cv2.TM_SQDIFF_NORMED)[0][0],
                1.0 - cv2.matchTemplate(roi_thresh_inv, template_thresh_inv, cv2.TM_SQDIFF_NORMED)[0][0]
            )

            # Method 3: SSIM on edges (guard tiny ROIs)
            try:
                from skimage.metrics import structural_similarity
                min_dim = min(roi_edges.shape[0], roi_edges.shape[1])
                if min_dim < 7:
                    score_ssim = 0.0
                else:
                    win_size = min(7, min_dim)
                    if win_size % 2 == 0:
                        win_size -= 1
                    if win_size < 3:
                        score_ssim = 0.0
                    else:
                        score_ssim = structural_similarity(roi_edges, template_edges, win_size=win_size)
            except ImportError:
                score_ssim = 0.0
            except ValueError:
                score_ssim = 0.0

            # Method 4: Contour matching (shape similarity)
            contours_roi, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_roi_inv, _ = cv2.findContours(roi_thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            score_contour = 0
            contour_pairs = [
                (contours_roi, contours_template),
                (contours_roi_inv, contours_template),
                (contours_roi, contours_template_inv),
                (contours_roi_inv, contours_template_inv)
            ]
            for c_roi, c_tpl in contour_pairs:
                if not c_roi or not c_tpl:
                    continue
                cnt_roi = max(c_roi, key=cv2.contourArea)
                cnt_template = max(c_tpl, key=cv2.contourArea)
                try:
                    # Use both Hu-moments diff and matchShapes to score similarity
                    moments_roi = cv2.HuMoments(cv2.moments(cnt_roi)).flatten()
                    moments_template = cv2.HuMoments(cv2.moments(cnt_template)).flatten()
                    diff = np.sum(np.abs(np.log10(np.abs(moments_roi) + 1e-10) -
                                         np.log10(np.abs(moments_template) + 1e-10)))
                    score_hu = max(0, 1.0 - (diff / 10.0))

                    shape_score = cv2.matchShapes(cnt_roi, cnt_template, cv2.CONTOURS_MATCH_I1, 0.0)
                    score_shape = 1.0 / (1.0 + shape_score * 5.0)

                    score = max(score_hu, score_shape)
                    if score > score_contour:
                        score_contour = score
                except Exception:
                    continue
            
            # Combined score
            scores = [score_edge, score_thresh, score_sqdiff, score_ssim, score_contour]
            if score_mode == "top3":
                top3 = sorted(scores, reverse=True)[:3]
                combined_score = sum(top3) / 3.0
            elif score_mode == "avg":
                combined_score = sum(scores) / float(len(scores))
            else:
                # default: strongest signal
                combined_score = max(scores)
            
            rarity_name = rarity_names.get(rarity_code, rarity_code)
            results.append((set_code, rarity_name, rarity_code, combined_score, score_edge, score_thresh, score_ssim, score_contour))
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = rarity_name
                best_set = set_code
    
    if not quiet:
        print(f"\nChecked {total_checked} symbol templates")

    # Sort and display top results
    results.sort(key=lambda x: x[3], reverse=True)

    if set_only:
        # Aggregate best score per set
        best_by_set = {}
        for set_code, rarity_name, rarity_code, combined, edge, thresh, ssim, contour in results:
            if set_code not in best_by_set or combined > best_by_set[set_code][0]:
                best_by_set[set_code] = (combined, rarity_name)

        set_results = [(set_code, score, rarity) for set_code, (score, rarity) in best_by_set.items()]
        set_results.sort(key=lambda x: x[1], reverse=True)

        second_best_score = None
        if set_results:
            best_set, best_score, best_match = set_results[0][0], set_results[0][1], set_results[0][2]
            if len(set_results) > 1:
                second_best_score = set_results[1][1]

        if not quiet:
            print("\n" + "=" * 80)
            print(f"Top Matching Results (Set-Only):")
            print("=" * 80)
            print(f"{'Set':<8} {'BestRarity':<15} {'Combined':<10}")
            print("-" * 80)
            for i, (set_code, combined, rarity_name) in enumerate(set_results[:15]):
                marker = "*** " if set_code == best_set else "    "
                print(f"{marker}{set_code:<8} {rarity_name:<15} {combined:>6.3f}")
            if len(set_results) > 15:
                print(f"    ... and {len(set_results) - 15} more")
            print("=" * 80)
    else:
        if not quiet:
            print("\n" + "=" * 80)
            print(f"Top Matching Results (Shape-Based):")
            print("=" * 80)
            print(f"{'Set':<8} {'Rarity':<15} {'Combined':<10} {'Edge':<10} {'Thresh':<10} {'SSIM':<10} {'Contour':<10}")
            print("-" * 80)

            # Show top 15 results
            for set_code, rarity_name, rarity_code, combined, edge, thresh, ssim, contour in results[:15]:
                marker = "*** " if set_code == best_set and rarity_name == best_match else "    "
                print(f"{marker}{set_code:<8} {rarity_name:<15} {combined:>6.3f}    {edge:>6.3f}     {thresh:>6.3f}     {ssim:>6.3f}     {contour:>6.3f}")

            if len(results) > 15:
                print(f"    ... and {len(results) - 15} more")

            print("=" * 80)

    if best_score > 0.5:  # Threshold for confidence
        if not quiet:
            print(f"\n[OK] Best Match: {best_set} - {best_match} (confidence: {best_score:.3f})")
        margin = None
        if second_best_score is not None:
            margin = best_score - second_best_score
        return {'set_code': best_set, 'rarity': best_match, 'confidence': best_score, 'best_score': best_score, 'margin': margin}
    else:
        if not quiet:
            print(f"\n[NO] No confident match (best score: {best_score:.3f} < 0.5 threshold)")
            if best_set:
                print(f"   Best guess: {best_set} - {best_match}")
        return {'set_code': best_set, 'rarity': best_match, 'confidence': best_score, 'best_score': best_score}


def batch_test_set(set_code, roi_relative=None, save_roi=False, set_only=True, per_rarity=1, quiet=False, candidate_sets=None):
    """
    Batch test multiple cards from the same set
    roi_relative: tuple of (x1_pct, y1_pct, x2_pct, y2_pct) as percentages of image size
    save_roi: if True and roi_relative is provided, save it to the config file
    """
    if not quiet:
        print("=" * 80)
        print(f"BATCH TESTING MODE - Set: {set_code}")
        print("=" * 80)
    
    # If ROI not provided, try to load from config
    if roi_relative is None:
        roi_relative = get_roi_for_set(set_code)
        if roi_relative is None:
            if not quiet:
                print(f"\n[NO] Cannot test {set_code} - set has no symbol")
            return {}
    else:
        if not quiet:
            print(f"  Using provided ROI: {roi_relative}")
        # Save this ROI for the set if requested
        if save_roi:
            save_roi_for_set(set_code, roi_relative)
    
    # Load symbols
    symbols = load_set_symbols()
    if candidate_sets:
        symbols = {k: v for k, v in symbols.items() if k in candidate_sets}
    
    # Get cards for each rarity
    target_rarities = ['Common', 'Uncommon', 'Rare', 'Mythic']
    results_by_rarity = {}
    
    for rarity in target_rarities:
        if not quiet:
            print(f"\nFetching {rarity} cards from {set_code}...")
        cards = get_cards_by_set_and_rarity(set_code, rarity, limit=per_rarity)
        
        if not cards:
            if not quiet:
                print(f"  No {rarity} cards found")
            continue

        if not quiet:
            print(f"  Found {len(cards)} cards")
        results_by_rarity[rarity] = []
        
        for card in cards:
            if not quiet:
                print(f"\n{'─' * 80}")
                print(f"Testing: {card['name']} ({card['rarity']})")
                print(f"Product ID: {card['product_id']}")
            
            # Find image
            image_path = find_card_image(card['product_id'])
            if not image_path:
                if not quiet:
                    print(f"  [NO] Image not found")
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                if not quiet:
                    print(f"  [NO] Could not load image")
                continue
            
            # Extract ROI
            h, w = image.shape[:2]
            
            if roi_relative:
                # Use provided relative coordinates
                x1 = int(w * roi_relative[0])
                y1 = int(h * roi_relative[1])
                x2 = int(w * roi_relative[2])
                y2 = int(h * roi_relative[3])
            else:
                # Use default location (middle-right area where set symbols usually are)
                x1, y1 = int(w * 0.85), int(h * 0.85)
                x2, y2 = int(w * 0.95), int(h * 0.95)
            
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                if not quiet:
                    print(f"  [NO] Invalid ROI")
                continue
            
            if not quiet:
                print(f"  ROI extracted: {roi.shape}")
            
            # Match
            match = match_symbol_template(
                roi,
                symbols,
                set_code if not set_only else None,
                set_only=set_only,
                search_all_sets=set_only,
                quiet=quiet
            )
            
            # Record result
            if set_only:
                correct = match and match['set_code'] == set_code
            else:
                correct = match and match['rarity'] == rarity
            results_by_rarity[rarity].append({
                'card': card,
                'match': match,
                'correct': correct,
                'score': match['best_score'] if match else 0.0
            })
            
            if not quiet:
                if correct:
                    print(f"  [OK] CORRECT ({match['best_score']:.3f})")
                else:
                    if set_only:
                        predicted = match['set_code'] if match else 'None'
                        print(f"  [NO] WRONG - Predicted set: {predicted} (True: {set_code}, Score: {match['best_score'] if match else 0.0:.3f})")
                    else:
                        predicted = match['rarity'] if match else 'None'
                        print(f"  [NO] WRONG - Predicted: {predicted} (True: {rarity}, Score: {match['best_score'] if match else 0.0:.3f})")
    
    # Print summary
    if not quiet:
        print("\n" + "=" * 80)
        print("BATCH TEST SUMMARY")
        print("=" * 80)
    
    total_tested = 0
    total_correct = 0
    
    for rarity in target_rarities:
        if rarity not in results_by_rarity or not results_by_rarity[rarity]:
            if not quiet:
                print(f"\n{rarity}: No tests")
            continue
        
        results = results_by_rarity[rarity]
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_score = sum(r['score'] for r in results) / total if total > 0 else 0
        
        total_tested += total
        total_correct += correct
        
        if not quiet:
            label = f"{rarity}" if not set_only else f"{rarity} (set-only)"
            print(f"\n{label}:")
            print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
            print(f"  Avg Score: {avg_score:.3f}")

            for r in results:
                status = "[OK]" if r['correct'] else "[NO]"
                pred = r['match']['rarity'] if r['match'] else 'None'
                print(f"    {status} {r['card']['name'][:40]:<40} → {pred} ({r['score']:.3f})")
    
    overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0
    if not quiet:
        print(f"\n{'=' * 80}")
        print(f"OVERALL: {total_correct}/{total_tested} correct ({overall_accuracy:.1f}%)")
        print(f"{'=' * 80}")
    
    return results_by_rarity


def main():
    """Main program loop"""
    global points, image_display, window_name, roi_coords
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MTG Symbol Recognition Tool')
    parser.add_argument('image', nargs='?', help='Path to card image (optional)')
    parser.add_argument('--batch', metavar='SET', help='Batch test mode for a specific set (e.g., M12)')
    parser.add_argument('--roi', metavar='X1,Y1,X2,Y2', help='ROI coordinates as percentages (e.g., 0.85,0.85,0.95,0.95)')
    parser.add_argument('--save-roi', action='store_true', help='Save the provided ROI coordinates for this set')
    parser.add_argument('--mser-auto', action='store_true', help='Auto-detect symbol ROI with MSER (single image only)')
    parser.add_argument('--mser-template-test', choices=['max100', 'max150'], help='Run MSER test on mtg_symbol_templates')
    parser.add_argument('--mser-output', help='Output directory for MSER template debug images')
    parser.add_argument('--mser-max-sets', type=int, default=0, help='Limit number of sets for MSER template test')
    parser.add_argument('--mser-max-per-set', type=int, default=0, help='Limit number of symbols per set for MSER template test')
    args = parser.parse_args()

    # MSER template test mode
    if args.mser_template_test:
        templates_root = TEMPLATES_BASE_PATH / args.mser_template_test
        max_sets = args.mser_max_sets if args.mser_max_sets and args.mser_max_sets > 0 else None
        max_per_set = args.mser_max_per_set if args.mser_max_per_set and args.mser_max_per_set > 0 else 0
        return test_mser_on_templates(
            templates_root=templates_root,
            max_sets=max_sets,
            max_per_set=max_per_set,
            output_dir=args.mser_output
        )
    
    # Batch mode
    if args.batch:
        roi_relative = None
        if args.roi:
            try:
                coords = [float(x) for x in args.roi.split(',')]
                if len(coords) == 4:
                    roi_relative = tuple(coords)
            except:
                print("Invalid ROI format. Use: --roi x1,y1,x2,y2 (as decimals 0.0-1.0)")
                return 1
        
        batch_test_set(args.batch.upper(), roi_relative, save_roi=args.save_roi)
        return 0
    
    print("=" * 60)
    print("Magic the Gathering Symbol Recognition Tool")
    print("=" * 60)
    
    # Check for command line argument for image path
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: Image file not found: {image_path}")
            return 1
        
        # Try to extract product_id from filename
        product_id = image_path.stem
        
        # Try to get card info from database
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(f"SELECT product_id, name, set_code, rarity FROM {MAGIC_TABLE} WHERE product_id = ?", (product_id,))
        result = cur.fetchone()
        conn.close()
        
        if result:
            card = {
                'product_id': result[0],
                'name': result[1],
                'set_code': result[2],
                'rarity': result[3]
            }
            print(f"\nCard Selected (from database):")
            print(f"  Product ID: {card['product_id']}")
            print(f"  Name: {card['name']}")
            print(f"  Set Code: {card['set_code']}")
            print(f"  Rarity: {card['rarity']}")
        else:
            card = {'product_id': product_id, 'name': 'Unknown', 'set_code': 'Unknown', 'rarity': 'Unknown'}
            print(f"\nUsing image: {image_path.name}")
            print(f"  (Card not found in database)")
    else:
        # Load set symbols
        symbols = load_set_symbols()
        
        if not symbols:
            print("ERROR: No set symbols loaded!")
            print(f"Make sure vectors are in: {VECTORS_PATH}")
            return 1
        
        # Get a random card
        print("\nFetching random Magic card from database...")
        card = get_random_magic_card()
        
        if not card:
            print("ERROR: Could not fetch a Magic card from database!")
            return 1
        
        print(f"\nCard Selected:")
        print(f"  Product ID: {card['product_id']}")
        print(f"  Name: {card['name']}")
        print(f"  Set Code: {card['set_code']}")
        print(f"  Rarity: {card['rarity']}")
        
        # Find the image
        image_path = find_card_image(card['product_id'])
        
        if not image_path:
            print(f"\nERROR: Could not find image for product_id: {card['product_id']}")
            print("\n" + "=" * 60)
            print("USAGE: python mtg_symbol_recognizer.py <path_to_card_image.jpg>")
            print("=" * 60)
            print("Please provide a card image path as the first argument")
            return 1
    
    # Load set symbols
    symbols = load_set_symbols()
    
    print(f"\nImage found: {image_path}")
    
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return 1
    
    # Resize if too large
    max_height = 800
    h, w = image.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, max_height))
    
    # Create display copy
    image_display = image.copy()
    
    # Auto MSER ROI mode (skip interactive selection)
    if args.mser_auto:
        prior = None
        if card and card.get('set_code') and isinstance(card.get('set_code'), str):
            prior = get_roi_for_set(card.get('set_code').upper())
        roi, bbox = mser_extract_symbol_roi(
            image,
            prior=prior,
            min_area_ratio=0.00005,
            max_area_ratio=0.02
        )
        if roi is None or bbox is None:
            print("[!] MSER could not find a symbol ROI.")
            return 1

        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(PROJECT_ROOT / "roi_mser_bbox.jpg"), image_display)
        print(f"[+] MSER ROI saved to: {PROJECT_ROOT / 'roi_mser_bbox.jpg'}")

        # Match against symbols
        match = match_symbol_template(roi, symbols, card.get('set_code'))
        if match:
            print(f"\nMatched set: {match}")
            return 0
        print("\nNo definitive match found")
        return 1

    # Create window and set mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("  1. Click on the TOP-LEFT corner of the set symbol")
    print("  2. Click on the BOTTOM-RIGHT corner of the set symbol")
    print("  3. Press ENTER to process")
    print("  4. Press 'r' to reset points")
    print("  5. Press ESC to exit")
    print("=" * 60)
    
    cv2.imshow(window_name, image_display)
    
    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            points = []
            image_display = image.copy()
            cv2.imshow(window_name, image_display)
            print("Points reset!")
        elif key == 13 and len(points) == 2:  # ENTER
            # Extract ROI
            roi = extract_roi(image, points[0], points[1])
            
            # Calculate and display relative coordinates (for use in batch mode)
            h, w = image.shape[:2]
            x1_rel = points[0][0] / w
            y1_rel = points[0][1] / h
            x2_rel = points[1][0] / w
            y2_rel = points[1][1] / h
            
            print("\n" + "=" * 60)
            print("ROI COORDINATES:")
            print(f"  Absolute pixels: ({points[0][0]}, {points[0][1]}) to ({points[1][0]}, {points[1][1]})")
            print(f"  Relative (0-1):  ({x1_rel:.4f}, {y1_rel:.4f}) to ({x2_rel:.4f}, {y2_rel:.4f})")
            print(f"\n  For batch mode, use:")
            print(f"    --roi {x1_rel:.4f},{y1_rel:.4f},{x2_rel:.4f},{y2_rel:.4f}")
            print("=" * 60)
            
            # Match against symbols
            match = match_symbol_template(roi, symbols, card.get('set_code'))
            
            if match:
                print(f"\nMatched set: {match}")
            else:
                print("\nNo definitive match found")
            
            # Reset for next try
            points = []
            image_display = image.copy()
            cv2.imshow(window_name, image_display)
    
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
