"""
Generate a printable PDF for ArUco marker IDs 0-3.

Each page contains one marker square that is exactly 20 cm x 20 cm,
including the white quiet-zone margin.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "aruco_markers"
OUTPUT_DIR = SOURCE_DIR / "print_20cm"
OUTPUT_PDF = BASE_DIR / "aruco_markers_20cm_ids_0_3.pdf"

MARKER_IDS = [0, 1, 2, 3]
PRINTED_SIZE_CM = 20.0
CUT_GUIDE_LENGTH_CM = 0.8
CUT_GUIDE_GAP_CM = 0.12

# A 5x5 ArUco marker has a one-module black border, so the marker bitmap is
# 7 modules wide. Add a one-module white quiet zone on each side: 9 modules.
MARKER_MODULES_WITH_BLACK_BORDER = 7
TOTAL_MODULES_WITH_WHITE_MARGIN = 9
OUTPUT_PIXELS = 1800


def make_marker_with_margin(marker_id: int) -> Path:
    source_path = SOURCE_DIR / f"aruco_marker_{marker_id}.png"
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"aruco_marker_{marker_id}_20cm_with_margin.png"

    if not source_path.exists():
        if output_path.exists():
            return output_path
        raise FileNotFoundError(f"Missing source marker: {source_path}")

    source = Image.open(source_path).convert("L")
    marker_pixels = round(OUTPUT_PIXELS * MARKER_MODULES_WITH_BLACK_BORDER / TOTAL_MODULES_WITH_WHITE_MARGIN)
    margin_pixels = (OUTPUT_PIXELS - marker_pixels) // 2
    resized = source.resize((marker_pixels, marker_pixels), Image.Resampling.NEAREST)

    page = Image.new("L", (OUTPUT_PIXELS, OUTPUT_PIXELS), 255)
    page.paste(resized, (margin_pixels, margin_pixels))
    page.save(output_path)
    return output_path


def build_pdf(marker_paths: list[tuple[int, Path]]) -> None:
    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=A4)
    page_width, page_height = A4
    marker_size = PRINTED_SIZE_CM * cm
    guide_length = CUT_GUIDE_LENGTH_CM * cm
    guide_gap = CUT_GUIDE_GAP_CM * cm
    x = (page_width - marker_size) / 2
    y = (page_height - marker_size) / 2

    pdf.setTitle("ArUco markers 0-3, 20 cm with white margins")
    pdf.setAuthor("kiwi")

    for marker_id, marker_path in marker_paths:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(page_width / 2, page_height - 1.7 * cm, f"ArUco marker ID {marker_id}")
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(
            page_width / 2,
            page_height - 2.15 * cm,
            "Print at 100% scale. The square below is 20 cm x 20 cm including white margins.",
        )
        pdf.drawImage(
            ImageReader(str(marker_path)),
            x,
            y,
            width=marker_size,
            height=marker_size,
            preserveAspectRatio=True,
            mask="auto",
        )

        # Cut guides mark the outside edge of the 20 cm square without drawing
        # over the white quiet zone needed for ArUco detection.
        left = x
        right = x + marker_size
        bottom = y
        top = y + marker_size
        pdf.setStrokeColorRGB(0.85, 0.0, 0.0)
        pdf.setLineWidth(0.5)
        for corner_x, x_sign in ((left, -1), (right, 1)):
            for corner_y, y_sign in ((bottom, -1), (top, 1)):
                pdf.line(
                    corner_x + x_sign * guide_gap,
                    corner_y,
                    corner_x + x_sign * (guide_gap + guide_length),
                    corner_y,
                )
                pdf.line(
                    corner_x,
                    corner_y + y_sign * guide_gap,
                    corner_x,
                    corner_y + y_sign * (guide_gap + guide_length),
                )

        pdf.setFont("Helvetica", 8)
        pdf.setFillColorRGB(0.55, 0.0, 0.0)
        pdf.drawCentredString(page_width / 2, y - 0.55 * cm, "Red corner guides mark the 20 cm cut boundary.")
        pdf.setFillColorRGB(0.0, 0.0, 0.0)
        pdf.showPage()

    pdf.save()


def main() -> None:
    marker_paths = [(marker_id, make_marker_with_margin(marker_id)) for marker_id in MARKER_IDS]
    build_pdf(marker_paths)
    print(f"Wrote {OUTPUT_PDF}")
    for _, path in marker_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
