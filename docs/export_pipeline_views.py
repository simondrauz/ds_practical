#!/usr/bin/env python3
"""Generate the pipeline SVG views and export regular PNG/PDF files.

The script uses a local Chrome/Chromium executable. It has no Python package
dependencies and keeps the PDF page size equal to the SVG canvas, so the larger
standalone view remains a single page.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent

VIEWS = {
    "assisting": {
        "svg": "current_pipeline_overview.svg",
        "width": 1800,
        "height": 1120,
    },
    "by_itself": {
        "svg": "current_pipeline_overview_by_itself.svg",
        "width": 2200,
        "height": 1580,
    },
}

SVG_SOURCES_GZIP_BASE64 = {
    "assisting": "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",
    "by_itself": "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",
}


def render_svg(view_name: str) -> str:
    encoded = SVG_SOURCES_GZIP_BASE64[view_name]
    return gzip.decompress(base64.b64decode(encoded)).decode("utf-8")


def write_svg(view_name: str, output_path: Path) -> None:
    output_path.write_text(render_svg(view_name), encoding="utf-8")


def find_chrome(explicit_path: str | None = None) -> str:
    candidates = [
        explicit_path,
        os.environ.get("CHROME_BIN"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise SystemExit(
        "Could not find Chrome/Chromium. Install Google Chrome or pass "
        "`--chrome-bin /path/to/chrome`."
    )


def run_chrome(chrome: str, args: list[str]) -> None:
    command = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--run-all-compositor-stages-before-draw",
        *args,
    ]
    subprocess.run(command, check=True)


def export_png(chrome: str, svg_path: Path, output_path: Path, width: int, height: int) -> None:
    run_chrome(
        chrome,
        [
            f"--screenshot={output_path}",
            f"--window-size={width},{height}",
            svg_path.as_uri(),
        ],
    )


def export_pdf(chrome: str, svg_path: Path, output_path: Path, width: int, height: int) -> None:
    svg = svg_path.read_text(encoding="utf-8")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @page {{ size: {width}px {height}px; margin: 0; }}
    html, body {{
      margin: 0;
      width: {width}px;
      height: {height}px;
      overflow: hidden;
      background: #ffffff;
    }}
    svg {{
      display: block;
      width: {width}px;
      height: {height}px;
    }}
  </style>
</head>
<body>
{svg}
</body>
</html>
"""

    with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as handle:
        handle.write(html)
        temp_html = Path(handle.name)

    try:
        run_chrome(
            chrome,
            [
                f"--print-to-pdf={output_path}",
                "--no-pdf-header-footer",
                temp_html.as_uri(),
            ],
        )
    finally:
        temp_html.unlink(missing_ok=True)


def selected_views(view: str) -> list[str]:
    if view == "all":
        return list(VIEWS)
    return [view]


def selected_formats(file_format: str) -> list[str]:
    if file_format == "all":
        return ["svg", "png", "pdf"]
    return [file_format]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        choices=["all", *VIEWS.keys()],
        default="all",
        help="Which view to export. Default: all.",
    )
    parser.add_argument(
        "--format",
        choices=["all", "svg", "png", "pdf"],
        default="all",
        help="Which output format to create. Default: all.",
    )
    parser.add_argument(
        "--chrome-bin",
        help="Optional path to a Chrome/Chromium executable.",
    )
    args = parser.parse_args()

    formats = selected_formats(args.format)
    chrome = None
    if any(file_format in formats for file_format in ["png", "pdf"]):
        chrome = find_chrome(args.chrome_bin)

    for view_name in selected_views(args.view):
        view = VIEWS[view_name]
        svg_path = DOCS_DIR / view["svg"]
        stem = svg_path.stem
        width = int(view["width"])
        height = int(view["height"])

        write_svg(view_name, svg_path)
        if "svg" in formats:
            print(f"wrote {svg_path.relative_to(DOCS_DIR.parent)}")

        for file_format in formats:
            if file_format == "svg":
                continue
            output_path = DOCS_DIR / f"{stem}.{file_format}"
            if file_format == "png":
                assert chrome is not None
                export_png(chrome, svg_path, output_path, width, height)
            elif file_format == "pdf":
                assert chrome is not None
                export_pdf(chrome, svg_path, output_path, width, height)
            print(f"wrote {output_path.relative_to(DOCS_DIR.parent)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
