"""
TCG Card Scanner — Error Code Reference
========================================
Format: E{category}{number}

  E1xx  Database & Initialisation
  E2xx  Camera & Vision
  E3xx  Recognition / Matching
  E4xx  Serial / Arduino Communication  (Python side)
  E5xx  Arduino Hardware                (reported by Arduino)
  E6xx  Collection & Export
  E7xx  Network / Download
"""

# ---------------------------------------------------------------------------
# Master table: code -> {title, description, fix}
# ---------------------------------------------------------------------------
ERROR_CODES: dict[str, dict] = {

    # ── E1xx — Database & Initialisation ───────────────────────────────────
    "E101": {
        "title": "Database Not Found",
        "description": "unified_card_database.db was not found on disk.",
        "fix": "Go to the Downloads tab and download unified_card_database.db from the server.",
        "severity": "error",
    },
    "E102": {
        "title": "Database Download Failed",
        "description": "All server endpoints were tried but the database could not be downloaded.",
        "fix": "Check your internet connection and verify the server URL in the Downloads tab.",
        "severity": "error",
    },
    "E103": {
        "title": "No Games Loaded",
        "description": "The database opened but no game tables were found inside it.",
        "fix": "Re-download unified_card_database.db — the file may be corrupt or outdated.",
        "severity": "error",
    },
    "E104": {
        "title": "Collection Manager Init Failed",
        "description": "The card collection manager could not be started.",
        "fix": "Check that the Collection/ folder is writable. Restart the application.",
        "severity": "warning",
    },
    "E105": {
        "title": "Database Connection Failed",
        "description": "A thread-local database connection could not be opened.",
        "fix": "Ensure the database file is not locked by another process.",
        "severity": "error",
    },
    "E106": {
        "title": "Scanner Not Initialised",
        "description": "An operation was attempted before the scanner was set up.",
        "fix": "Wait for the scanner to finish initialising, or restart the application.",
        "severity": "error",
    },
    "E107": {
        "title": "Image Load Failed",
        "description": "An image file could not be read from disk (missing or corrupt).",
        "fix": "Verify the file path exists and is a valid image format.",
        "severity": "error",
    },
    "E108": {
        "title": "Plugin Load Failed",
        "description": "A camera, recognition, or Arduino plugin could not be loaded.",
        "fix": "Check that the plugin file exists in the plugins/ folder and has no syntax errors.",
        "severity": "warning",
    },

    # ── E2xx — Camera & Vision ──────────────────────────────────────────────
    "E201": {
        "title": "Camera Failed to Open",
        "description": "OpenCV could not open the selected camera index.",
        "fix": "Check Camera Index in the Scanner tab. Ensure no other app is using the camera.",
        "severity": "error",
    },
    "E202": {
        "title": "Camera Read Failed",
        "description": "A frame could not be read from the camera (disconnected or driver error).",
        "fix": "Replug the camera USB cable and restart the camera in the Scanner tab.",
        "severity": "error",
    },
    "E203": {
        "title": "Card Contour Not Found",
        "description": "The vision pipeline could not detect a card-shaped contour in the frame.",
        "fix": "Ensure the card is flat, well-lit, fully visible against a contrasting background.",
        "severity": "warning",
    },
    "E204": {
        "title": "Perspective Warp Failed",
        "description": "Card contour found but perspective transform could not be computed.",
        "fix": "Move the card slightly and ensure all four corners are visible.",
        "severity": "warning",
    },
    "E205": {
        "title": "Frame Grab Failed",
        "description": "Camera returned no frame (disconnected mid-stream or driver crash).",
        "fix": "Replug camera USB and restart the camera in the Scanner tab.",
        "severity": "error",
    },

    # ── E3xx — Recognition / Matching ──────────────────────────────────────
    "E301": {
        "title": "Card Not Recognised",
        "description": "No matching card was found in the pHash database.",
        "fix": "Ensure the correct game is selected and its pHash DB is downloaded. Check lighting.",
        "severity": "warning",
    },
    "E302": {
        "title": "Match Confidence Too Low",
        "description": "The best match found was below the minimum confidence threshold.",
        "fix": "Lower the Match Threshold in the Scanner tab, or improve lighting/angle.",
        "severity": "warning",
    },
    "E303": {
        "title": "No pHash Data for Game",
        "description": "The selected game has no perceptual hash data in the local database.",
        "fix": "Go to Downloads and download the pHash DB for this game.",
        "severity": "error",
    },
    "E304": {
        "title": "Card Processing Error",
        "description": "An unexpected exception occurred during card recognition or perspective correction.",
        "fix": "Check the status log for detail. Ensure camera image is clear and card is not obstructed.",
        "severity": "error",
    },
    "E305": {
        "title": "Recognition Plugin Error",
        "description": "The optional recognition plugin raised an exception during card identification.",
        "fix": "Check the plugin code for bugs or disable it via the plugins/ folder.",
        "severity": "warning",
    },

    # ── E4xx — Serial / Arduino Communication (Python side) ────────────────
    "E401": {
        "title": "pyserial Not Installed",
        "description": "The pyserial library is not available in the Python environment.",
        "fix": "Run:  pip install pyserial  then restart the application.",
        "severity": "error",
    },
    "E402": {
        "title": "Serial Port Not Configured",
        "description": "No serial port was set before attempting to connect.",
        "fix": "Enter the correct COM port in the Arduino tab (e.g. COM3 or /dev/ttyUSB0).",
        "severity": "error",
    },
    "E403": {
        "title": "Serial Port Open Failed",
        "description": "The serial port could not be opened (busy, wrong port, or permissions).",
        "fix": "Check COM port and baud rate. Ensure no other program has the port open.",
        "severity": "error",
    },
    "E404": {
        "title": "Arduino Ready Timeout",
        "description": "The Arduino did not send its ready signal within the timeout period.",
        "fix": "Check USB cable and verify correct firmware (Main.ino) is flashed.",
        "severity": "error",
    },
    "E405": {
        "title": "Arduino Send Error",
        "description": "An exception occurred while writing data to the serial port.",
        "fix": "Check USB connection. If port disconnected, reconnect via the Arduino tab.",
        "severity": "error",
    },
    "E406": {
        "title": "Arduino No Response",
        "description": "A command was sent to the Arduino but no valid response was received.",
        "fix": "Check USB connection. If port disconnected, reconnect via the Arduino tab.",
        "severity": "warning",
    },
    "E407": {
        "title": "Arduino Param Upload Partial",
        "description": "Not all parameters were accepted by the Arduino during an upload.",
        "fix": "Check status log for which param failed. Ensure machine is stopped and at home.",
        "severity": "warning",
    },

    # ── E5xx — Arduino Hardware (reported by Arduino over serial) ───────────
    "E501": {
        "title": "ToF Sensor Not Found",
        "description": "VL6180X distance sensor not detected on I2C bus.",
        "fix": "Check SDA/SCL wiring to pins 20/21 on Mega. Verify 3.3 V power. Check for swapped SDA/SCL.",
        "severity": "error",
    },
    "E502": {
        "title": "ToF System Error",
        "description": "VL6180X reported an internal system error.",
        "fix": "Power-cycle the Arduino and sensor. If persistent, sensor may need replacement.",
        "severity": "error",
    },
    "E503": {
        "title": "ToF ECE Failure",
        "description": "Early Convergence Estimate failure on ToF sensor.",
        "fix": "Clean sensor lens. Ensure no strong ambient IR light is hitting the sensor.",
        "severity": "warning",
    },
    "E504": {
        "title": "ToF No Convergence",
        "description": "ToF sensor could not converge on a stable distance reading.",
        "fix": "Ensure target surface is not too reflective or too dark. Check sensor alignment.",
        "severity": "warning",
    },
    "E505": {
        "title": "Pickup Retry Limit Reached",
        "description": "Pickup arm retried 10 times without picking a card. Machine halted.",
        "fix": "Check vacuum tubing for leaks. Verify Z-axis calibration (Zcal) and pickup threshold.",
        "severity": "error",
    },
    "E506": {
        "title": "Overflow Tray Full",
        "description": "Overflow tray (tray 34) reached 375 cards. Machine halted.",
        "fix": "Empty the overflow tray, then restart the machine.",
        "severity": "error",
    },
    "E507": {
        "title": "Tray Command Rejected: Machine Stopped",
        "description": "A tray/sort command arrived but the machine is not in Started state.",
        "fix": "Click 'Start Machine' in the Arduino tab before scanning cards.",
        "severity": "warning",
    },
    "E508": {
        "title": "Manual Command Rejected: Machine Running",
        "description": "A manual move/calibration command was rejected — machine is running.",
        "fix": "Click 'Stop Machine' in the Arduino tab before using manual controls.",
        "severity": "warning",
    },
    "E509": {
        "title": "Parameter Change Rejected: Not at Home",
        "description": "Parameter upload rejected — machine is not at home position.",
        "fix": "Click 'Home Machine' first, then upload parameters.",
        "severity": "warning",
    },
    "E510": {
        "title": "Homing Failed: X Endstop",
        "description": "X-axis endstop was not triggered within timeout during homing.",
        "fix": "Check X endstop wiring and position. Verify endstop LED lights when triggered.",
        "severity": "error",
    },
    "E511": {
        "title": "Homing Failed: Y Endstop",
        "description": "Y-axis endstop was not triggered within timeout during homing.",
        "fix": "Check Y endstop wiring and position.",
        "severity": "error",
    },
    "E512": {
        "title": "Homing Failed: Z Endstop",
        "description": "Z-axis endstop was not triggered within timeout during homing.",
        "fix": "Check Z endstop wiring and position.",
        "severity": "error",
    },
    "E513": {
        "title": "Serial Buffer Overflow",
        "description": "Incoming serial command exceeded the Arduino buffer size and was truncated.",
        "fix": "Shorten command strings. Max command length is 39 chars (buffSize-1).",
        "severity": "warning",
    },
    "E514": {
        "title": "Unknown Command Received",
        "description": "Arduino received a command it did not recognise.",
        "fix": "Check for typos in command names. Ensure firmware version matches the PC software.",
        "severity": "warning",
    },

    # ── E6xx — Collection & Export ──────────────────────────────────────────
    "E601": {
        "title": "Collection Manager Not Initialised",
        "description": "Export or save attempted but collection manager was not started.",
        "fix": "Restart the scanner. If persistent, check that Collection/ folder is writable.",
        "severity": "error",
    },
    "E602": {
        "title": "Export Failed",
        "description": "An error occurred while writing the export CSV file.",
        "fix": "Ensure destination folder exists and is writable. Check available disk space.",
        "severity": "error",
    },
    "E603": {
        "title": "Save to Collection Failed",
        "description": "A scanned card could not be saved to the collection JSON file.",
        "fix": "Check that Collection/master_collection.json is not open in another program.",
        "severity": "warning",
    },
    "E604": {
        "title": "Inventory File Not Accessible",
        "description": "The inventory tracking file could not be read or written.",
        "fix": "Check that Collection/ folder exists and is not read-only.",
        "severity": "warning",
    },
    "E605": {
        "title": "Inventory Check Error",
        "description": "Unexpected error while checking the card inventory.",
        "fix": "Check Collection/Collection.txt for corruption. Delete and restart if needed.",
        "severity": "warning",
    },

    # ── E7xx — Network / Download ───────────────────────────────────────────
    "E701": {
        "title": "Server Unreachable",
        "description": "A network or SSL error prevented connecting to the download server.",
        "fix": "Check internet connection. Verify server URL. Try 'Bypass SSL' on local networks.",
        "severity": "error",
    },
    "E702": {
        "title": "File Not Found on Server",
        "description": "Requested file was not found at any server endpoint (404).",
        "fix": "Check server URL. The file may have been moved — contact the server administrator.",
        "severity": "error",
    },
    "E703": {
        "title": "SSL Certificate Error",
        "description": "Server SSL certificate could not be verified.",
        "fix": "Use a CA bundle or enable 'Bypass SSL' in the Downloads tab (trusted networks only).",
        "severity": "warning",
    },
    "E704": {
        "title": "Download Write Error",
        "description": "File downloaded but could not be saved to disk.",
        "fix": "Check disk space and that recognition_data/ folder is writable.",
        "severity": "error",
    },
}


def lookup(code: str) -> dict:
    """Return the error entry for *code*, or a generic unknown entry."""
    return ERROR_CODES.get(code.upper(), {
        "title": "Unknown Error",
        "description": f"No entry found for code {code}.",
        "fix": "Check logs for more detail.",
        "severity": "error",
    })


def format_log(code: str, detail: str = "") -> str:
    """Return a compact one-line log string: '[E4xx] Title — detail'."""
    entry = lookup(code)
    base = f"[{code}] {entry['title']}"
    return f"{base} — {detail}" if detail else base
