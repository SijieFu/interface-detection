"""
Flask server to host image analysis requests
"""

import os
import json
import base64
from io import BytesIO
import logging
from collections import defaultdict

import cv2
from flask import Flask, request
from flask import jsonify, send_file, render_template

from info import info
from Flask_utils import get_image_md5_hash, _parse_bool
from Flask_utils import analyze_image, visualize_analysis


app = Flask(__name__, template_folder=info.HTML_DIR, static_folder=info.IMG_DIR)


@app.errorhandler(404)
def page_not_found(e):
    # Log the error
    app.logger.warning("Route not found: %s", request.url)
    return (
        jsonify({"message": f"404 Not Found: `{request.url}` is not a valid route."}),
        404,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    """
    HTTP request to analyze an image
    """

    arguments = defaultdict(lambda: False)
    arguments.update(request.form.to_dict())
    # URL parameters overwrite form parameters
    arguments.update(request.args.to_dict())

    # default parameters designed for autoHSP images
    kwargs_for_analysis = {
        # Flask_utils.analyze_image --> interface_detection.vial_detection.detect_interfaces
        "weights": "yolov8n.pt",
        "name": "vial",
        "conf": 0.7,
        "iou": 0.5,
        "device": "cpu",
        "max_det": 100,
        "width_expansion": 0.025,
        "alignment_cutoff": 0.5,
        "config": "",
        # Flask_utils.analyze_image --> interface_detection.vial_detection.detect_interfaces --> interface_detection.vial_contour.get_interfaces
        "over_exposure_threshold": 200,
        "bg_correction": "True",
        "bg_tolerance": "40",
        "bg_resolution": 0.02,
        "bg_sobel_ratio": 0.55,
        "cap_ratio": "0.1,0.25",
        "cap_target": "-1",
        "cap_tolerance": "30",
        "vessel_sobel_threshold": 31,
        "sobel_threshold": 24,
        "sobel_xy_ratio": 1.0,
        "dilation": "True",
        "phase_resolution": 0.08,
        "boundary_resolution": "0.4,0.1",
        "label_correction": "True",
        "label_low": "150",
        "label_high": "220",
        "label_check_gray": "False",
        "interface_signal": 0.55,
    }
    # default parameters for the general image analysis
    if use_default := _parse_bool(arguments.pop("usedefault", False)):
        kwargs_for_analysis.update(
            {
                "width_expansion": 0.0,
                "bg_tolerance": "30",
                "cap_ratio": "0.1,0.3",
                "cap_tolerance": "40",
                "vessel_sobel_threshold": 63,
                "sobel_xy_ratio": 0.75,
                "phase_resolution": 0.06,
                "boundary_resolution": "0.1,0.15",
                "interface_signal": 0.6,
            }
        )
    for key, value in kwargs_for_analysis.items():
        if key in arguments:
            try:
                kwargs_for_analysis[key] = type(value)(arguments[key])
            except:
                pass

    if request.method == "GET":
        return render_template(
            "img_upload_get.html",
            parameters=kwargs_for_analysis,
            use_default=use_default,
        )

    if "file" not in request.files:
        return f"No file part: {request.files}", 405

    file = request.files.get("file", False)
    filename = request.form.get("filename", "").strip() or file.filename
    filetype = request.form.get("mimetype", "").strip() or file.mimetype

    if not file or not filename:
        return "No selected file", 405

    if not file or filetype.lower() not in info.ALLOWED_EXTENSIONS:
        return "Invalid file type", 405

    # how to handle the image in the server
    saveflag = _parse_bool(
        arguments["saveflag"] or arguments["nosaveflag"] or "no"
    )  # default to "no" - save to tmp dir
    force_rerun = _parse_bool(arguments["force"] or "no")  # default to no
    # visualization parameters
    return_image = _parse_bool(arguments["returnimage"])
    zoomin = return_image and _parse_bool(arguments["zoomin"])
    return_json = _parse_bool(
        arguments["returnjson"] or arguments["donot_returnjson"] or "yes"
    )  # default to yes
    kwargs_for_analysis["bg_correction"] = _parse_bool(
        arguments["bg_correction"] or kwargs_for_analysis["bg_correction"]
    ) and not _parse_bool(arguments["no_bg_correction"] or "no")
    kwargs_for_analysis["dilation"] = _parse_bool(
        arguments["dilation"] or kwargs_for_analysis["dilation"]
    ) and not _parse_bool(arguments["no_dilation"] or "no")
    kwargs_for_analysis["label_correction"] = _parse_bool(
        arguments["label_correction"] or kwargs_for_analysis["label_correction"]
    ) and not _parse_bool(arguments["no_label_correction"] or "no")
    kwargs_for_analysis["label_check_gray"] = _parse_bool(
        arguments["label_check_gray"] or kwargs_for_analysis["label_check_gray"]
    )

    # filename = secure_filename(filename) # no need to secure the filename
    file_md5 = get_image_md5_hash(file)
    file_ext = (
        os.path.splitext(filename)[-1]
        or filetype.split("/")[-1].strip().split()[0].split(";")[0]
    )
    file_ext = file_ext.lstrip(".").lower() or "jpg"
    save_dir = info.IMG_DIR if saveflag else info.TMP_DIR
    os.makedirs(save_dir, exist_ok=True)
    file_save_name = f"{file_md5}.{file_ext}"
    file_save_path = os.path.join(save_dir, file_save_name)

    file.seek(0)
    if force_rerun or not os.path.isfile(file_save_path):
        file.save(file_save_path)
    file.close()

    # Analyze the image, result_path = IMG_RESULT_DIR / {file_md5}.json
    result_dir = os.path.join(save_dir, info.IMG_RESULT_DIRNAME)
    os.makedirs(result_dir, exist_ok=True)
    result_path, force_rerun = analyze_image(
        file_md5, file_save_path, force_rerun=force_rerun, **kwargs_for_analysis
    )

    readme_path = os.path.join(save_dir, info.README)
    try:
        with open(readme_path, "r") as f:
            readme_data = json.load(f)
    except FileNotFoundError:
        readme_data = {}

    # clean up the previous files if the file has been uploaded before
    _prev_file = readme_data.get(filename, "")
    _prev_md5 = os.path.splitext(_prev_file)[0]
    if _prev_file and _prev_md5 != file_md5:
        readme_data.pop(filename, None)
        readme_data.pop(_prev_file, None)
        if os.path.isfile(_tmp_file := os.path.join(save_dir, _prev_file)):
            os.remove(_tmp_file)
        if os.path.isfile(_tmp_file := os.path.join(result_dir, f"{_prev_md5}.json")):
            os.remove(_tmp_file)

    readme_data[file_save_name] = filename
    readme_data[filename] = file_save_name
    with open(readme_path, "w") as f:
        json.dump(readme_data, f, indent=4)

    # return annotated_filename and the image data
    with open(result_path) as f:
        result = json.load(f)
    result = {filename: result}

    if not return_image:  # return json analysis
        return jsonify(result)

    return_image = visualize_analysis(
        file_save_path,
        result[filename],
        zoomin=zoomin,
        force_rerun=force_rerun,
        debug=False,
    )

    default_img_ext = "jpg"
    _success, return_image = cv2.imencode(f".{default_img_ext}", return_image)
    if not _success:
        result["error"] = "Failed to encode image"
        return jsonify(result)

    return_image_io = BytesIO(return_image.tobytes())
    return_image_io.seek(0)

    return_image_name = f"annotated{'_zoomin' if zoomin else ''}_{filename}"
    if not return_json:
        return send_file(
            return_image_io,
            mimetype=f"image/{default_img_ext}",
            as_attachment=False,
            download_name=return_image_name,
        )

    # return both json and image
    result["annotated_image"] = {
        "file": base64.b64encode(return_image_io.read()).decode("utf-8"),
        "filename": return_image_name,
        "filetype": f"image/{default_img_ext}",
        "mimetype": f"image/{default_img_ext}",
    }
    return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(filename="flask.log", level=logging.INFO)
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False, threaded=False)
