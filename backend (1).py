"""
backend.py
----------
Flask REST API for the Bird Size Identifier.

Endpoints:
  GET  /api/health              -> service status
  GET  /api/species              -> list all species in the database
  POST /api/identify             -> identify a bird from length + wingspan

Run:
  pip install -r requirements.txt
  python backend.py
  -> serves on http://localhost:5000
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from bird_size_finder import (
    BIRDS,
    Bird,
    find_best_matches,
    fetch_wikipedia_image_url,
)

app = Flask(__name__)
CORS(app)  # allow requests from a browser-based frontend on a different origin


def _bird_to_dict(bird: Bird, include_image: bool = False) -> Dict[str, Any]:
    data = asdict(bird)
    if include_image:
        data["image_url"] = fetch_wikipedia_image_url(bird.wiki_title)
    data["wikipedia_url"] = f"https://en.wikipedia.org/wiki/{bird.wiki_title}"
    return data


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok", "species_count": len(BIRDS)})


@app.route("/api/species", methods=["GET"])
def list_species() -> Any:
    """Return the full species database (no live image fetch, for speed)."""
    return jsonify({"species": [_bird_to_dict(b) for b in BIRDS]})


@app.route("/api/identify", methods=["POST"])
def identify() -> Any:
    """
    Body (JSON): { "length_cm": number, "wingspan_cm": number, "top_n": int (optional) }
    Response: best match + runners-up, each with size range and a live photo URL.
    """
    payload = request.get_json(silent=True) or {}

    try:
        length_cm = float(payload["length_cm"])
        wingspan_cm = float(payload["wingspan_cm"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide numeric 'length_cm' and 'wingspan_cm'."}), 400

    if length_cm <= 0 or wingspan_cm <= 0:
        return jsonify({"error": "'length_cm' and 'wingspan_cm' must be greater than 0."}), 400

    top_n = int(payload.get("top_n", 3))
    top_n = max(1, min(top_n, len(BIRDS)))

    matches = find_best_matches(length_cm, wingspan_cm, top_n=top_n)

    results = []
    for i, match in enumerate(matches):
        # Only fetch the live photo for the top match to keep the response fast;
        # runners-up can be fetched lazily by the frontend if needed.
        include_image = i == 0
        results.append({
            "score": round(match.score, 3),
            "exact_match": match.score == 0.0,
            "bird": _bird_to_dict(match.bird, include_image=include_image),
        })

    return jsonify({
        "query": {"length_cm": length_cm, "wingspan_cm": wingspan_cm},
        "results": results,
    })


@app.errorhandler(404)
def not_found(_e: Any) -> Any:
    return jsonify({"error": "Not found. Available routes: /api/health, /api/species, /api/identify"}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
