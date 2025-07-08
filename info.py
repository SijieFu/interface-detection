import os
import tempfile


class _Info:
    """
    instance for dealing with configuration settings
    """

    # `Flask.py` configuration
    WORKING_DIR = os.path.dirname(__file__)
    WORKING_DIRNAME = os.path.basename(WORKING_DIR)
    if not WORKING_DIRNAME:
        WORKING_DIRNAME = "vial_detection"

    IMG_DIR = os.path.join(WORKING_DIR, "images")
    README = f"readme.json"  # IMG_DIR / README
    IMG_RESULT_DIRNAME = "analysis"
    IMG_RESULT_DIR = os.path.join(IMG_DIR, IMG_RESULT_DIRNAME)
    os.makedirs(IMG_RESULT_DIR, exist_ok=True)
    HTML_DIR = os.path.join(WORKING_DIR, "templates")
    TMP_DIR = os.path.join(tempfile.gettempdir(), WORKING_DIRNAME)
    os.makedirs(TMP_DIR, exist_ok=True)

    ALLOWED_EXTENSIONS = {
        "image/png",
        "image/jpg",
        "image/jpeg",
        "image/gif",
        "image/svg",
        "image/bmp",
        "image/tiff",
        "image/webp",
        "application/json",
        "application/xml",
        "application/zip",
        "application/pdf",
        "text/plain",
    }

    # external API server for vial detection
    VIAL_DETECTION_SERVER = "http://localhost:5002"


info = _Info()
