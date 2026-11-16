"""Web routes of the demo e-challan feature (a Flask blueprint)."""

from flask import Blueprint, Response, jsonify, request, send_from_directory

import config
from backend.services.challan_demo import SIMULATED_TYPES


def create_challan_blueprint(database, pipeline):
    bp = Blueprint("challans", __name__)

    @bp.route("/api/challans")
    def list_challans():
        limit = request.args.get("limit", 10, type=int)
        return jsonify(database.recent_challans(limit=limit))

    @bp.route("/api/simulate_violation", methods=["POST"])
    def simulate_violation():
        kind = (request.get_json(silent=True) or {}).get("type", "")
        if kind not in SIMULATED_TYPES or not pipeline.request_simulated_violation(kind):
            return jsonify({"ok": False, "error": "unknown violation type"}), 400
        return jsonify({"ok": True})

    @bp.route("/challan_files/<path:name>")
    def challan_file(name):
        return send_from_directory(str(config.CHALLAN_FOLDER), name)

    @bp.route("/api/export/challans.csv")
    def export_challans():
        return Response(database.export_csv("challans"), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=challans.csv"})

    return bp