#!/usr/bin/env python3
"""
Groups images by their 'name' prefix (from 'name_number.ext' filenames),
lets you select which group(s) to process, then packs them into A4-width
grid images suitable for a LaTeX appendix.

Changes vs v1:
  - Images are larger: fixed number of columns (default 3), rows spill to
    extra pages instead of shrinking everything to fit one page.
  - Vertical whitespace is tight: each row height = tallest actual image in
    that row (not a fixed square cell).
  - Filename is overlaid in the top-right corner of each image in a
    readable contrasting colour.

Usage:
    python make_appendix_grids.py --input_dir /path/to/images --output_dir /path/to/output [--cols N]
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# A4 @ 300 DPI, single-column with ~20 mm margins
# ---------------------------------------------------------------------------
A4_WIDTH_PX  = 2480
A4_HEIGHT_PX = 3508
MARGIN_PX    = 236          # 20 mm
CONTENT_W    = A4_WIDTH_PX  - 2 * MARGIN_PX   # 2008 px
CONTENT_H    = A4_HEIGHT_PX - 2 * MARGIN_PX   # 3036 px
H_GAP        = 16           # horizontal gap between cells
V_GAP        = 16           # vertical gap between rows (tight)

FONT_PATH    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_PADDING = 12          # px inside the label background box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_groups(input_dir: Path) -> dict[str, list[Path]]:
    pattern = re.compile(r"^(.+?)_(\d+)\.[^.]+$", re.IGNORECASE)
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            groups[m.group(1)].append(p)
    for name in groups:
        groups[name].sort(key=lambda p: int(re.search(r"_(\d+)\.", p.name).group(1)))
    return dict(sorted(groups.items()))


def prompt_group_selection(groups: dict[str, list[Path]]) -> list[str]:
    print("\nDiscovered image groups:")
    print(f"  {'#':<4}  {'Group name':<40}  Images")
    print(f"  {'-'*4}  {'-'*40}  ------")
    names = list(groups.keys())
    for i, name in enumerate(names, 1):
        print(f"  {i:<4}  {name:<40}  {len(groups[name])}")

    print("\nSelect groups to process:")
    print("  • Numbers separated by commas/spaces  (e.g.  1, 3, 5)")
    print("  • A range                             (e.g.  2-4)")
    print("  • 'all' to process every group")

    while True:
        raw = input("\nYour selection: ").strip()
        if not raw:
            print("  Nothing entered — please try again.")
            continue
        if raw.lower() == "all":
            return names

        selected_indices: set[int] = set()
        tokens = re.split(r"[,\s]+", raw)
        valid = True
        for token in tokens:
            if not token:
                continue
            range_match = re.fullmatch(r"(\d+)-(\d+)", token)
            if range_match:
                a, b = int(range_match.group(1)), int(range_match.group(2))
                if a < 1 or b > len(names) or a > b:
                    print(f"  Invalid range '{token}' — must be between 1 and {len(names)}.")
                    valid = False
                    break
                selected_indices.update(range(a, b + 1))
            elif token.isdigit():
                idx = int(token)
                if idx < 1 or idx > len(names):
                    print(f"  Number {idx} is out of range (1–{len(names)}).")
                    valid = False
                    break
                selected_indices.add(idx)
            else:
                print(f"  Unrecognised token '{token}'.")
                valid = False
                break

        if valid and selected_indices:
            chosen = [names[i - 1] for i in sorted(selected_indices)]
            print(f"\n  Selected: {', '.join(chosen)}")
            confirm = input("  Confirm? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                return chosen
        elif valid:
            print("  No groups selected — please try again.")


# ---------------------------------------------------------------------------
# Label drawing
# ---------------------------------------------------------------------------

def _brightness(gray_img: Image.Image) -> float:
    """Return mean pixel value of a grayscale image."""
    try:
        data = gray_img.get_flattened_data()
    except AttributeError:
        data = list(gray_img.getdata())
    return sum(data) / max(1, gray_img.width * gray_img.height)


def pick_label_color(img: Image.Image, x: int, y: int, w: int, h: int) -> tuple[int,int,int]:
    """Sample average brightness of the top-right region and return black or white."""
    region = img.crop((x, y, x + w, y + h)).convert("L")
    avg = _brightness(region)
    return (0, 0, 0) if avg > 160 else (255, 255, 255)


def draw_label(img: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Overlay filename text in top-right corner with a semi-transparent background."""
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = LABEL_PADDING
    box_w = tw + pad * 2
    box_h = th + pad * 2

    bx = img.width - box_w - pad      # right-aligned with a little extra margin
    by = pad

    # Semi-transparent dark or light background
    region_x1 = max(0, bx)
    region_y1 = max(0, by)
    region_x2 = min(img.width,  bx + box_w)
    region_y2 = min(img.height, by + box_h)

    region_crop = img.crop((region_x1, region_y1, region_x2, region_y2)).convert("L")
    avg_brightness = _brightness(region_crop)
    bg_color    = (0,   0,   0,  160) if avg_brightness > 140 else (255, 255, 255, 160)
    text_color  = (255, 255, 255)     if avg_brightness > 140 else (0,   0,   0)

    # Draw background rectangle on a composited RGBA layer
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([bx, by, bx + box_w, by + box_h], fill=bg_color)
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")

    # Draw text
    draw = ImageDraw.Draw(img)
    draw.text((bx + pad, by + pad), text, font=font, fill=text_color)
    return img


# ---------------------------------------------------------------------------
# Layout: scale each image to fill column width, track per-row heights
# ---------------------------------------------------------------------------

def autocrop(img: Image.Image, bg_color: tuple = (255, 255, 255),
             tolerance: int = 15) -> Image.Image:
    """
    Crop away solid-colour borders around an image.
    Pixels within `tolerance` of `bg_color` on all channels are treated as background.
    Falls back to the original image if the result would be empty.
    """
    import numpy as np
    arr = np.array(img)
    bg = np.array(bg_color, dtype=np.int32)
    diff = np.abs(arr.astype(np.int32) - bg)
    mask = diff.max(axis=2) > tolerance   # True where pixel is NOT background

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img   # fully blank — return as-is

    r0, r1 = int(rows.argmax()), int(len(rows) - rows[::-1].argmax() - 1)
    c0, c1 = int(cols.argmax()), int(len(cols) - cols[::-1].argmax() - 1)
    return img.crop((c0, r0, c1 + 1, r1 + 1))


def scale_to_width(img: Image.Image, target_w: int) -> Image.Image:
    """Scale image so its width == target_w, preserving aspect ratio."""
    if img.width == 0:
        return img
    new_h = max(1, int(target_w * img.height / img.width))
    return img.resize((target_w, new_h), Image.LANCZOS)


def trim_page_bottom(page: Image.Image, content_bottom: int) -> Image.Image:
    """Crop blank whitespace below the last row, keeping one MARGIN_PX of padding."""
    new_h = min(page.height, content_bottom + MARGIN_PX)
    return page.crop((0, 0, page.width, new_h))


def layout_pages(
    imgs: list[Image.Image],
    filenames: list[str],
    cols: int,
    font: ImageFont.FreeTypeFont,
) -> list[Image.Image]:
    """
    Arrange images into A4 pages.
    - Each source image has its whitespace border cropped first.
    - Each image is then scaled to fill its column width.
    - Row height = tallest image in that row (no fixed square cell).
    - If a new row would overflow the page, start a new page.
    - The last page is trimmed so there is no blank bottom beyond MARGIN_PX.
    """
    cell_w = (CONTENT_W - H_GAP * (cols - 1)) // cols

    # Pre-crop whitespace, scale, and label all images
    scaled: list[Image.Image] = []
    for img, fname in zip(imgs, filenames):
        s = autocrop(img)
        s = scale_to_width(s, cell_w)
        s = draw_label(s, fname, font)
        scaled.append(s)

    pages: list[Image.Image] = []
    page = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255))
    cursor_y = MARGIN_PX
    last_content_y = MARGIN_PX
    row_start = 0

    while row_start < len(scaled):
        row_imgs = scaled[row_start : row_start + cols]
        row_h = max(im.height for im in row_imgs)

        # Would this row overflow the page?
        if cursor_y + row_h > MARGIN_PX + CONTENT_H and cursor_y > MARGIN_PX:
            pages.append(trim_page_bottom(page, last_content_y))
            page = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255))
            cursor_y = MARGIN_PX
            last_content_y = MARGIN_PX

        for col_idx, im in enumerate(row_imgs):
            x = MARGIN_PX + col_idx * (cell_w + H_GAP)
            page.paste(im, (x, cursor_y))

        last_content_y = cursor_y + row_h
        cursor_y = last_content_y + V_GAP
        row_start += cols

    pages.append(trim_page_bottom(page, last_content_y))
    return pages


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def process_group(
    name: str,
    paths: list[Path],
    output_dir: Path,
    cols: int,
) -> None:
    print(f"\n  Processing group '{name}' ({len(paths)} images, {cols} columns) ...")

    loaded: list[Image.Image] = []
    filenames: list[str] = []
    for p in paths:
        try:
            loaded.append(Image.open(p).convert("RGB"))
            filenames.append(p.name)
        except Exception as e:
            print(f"    Warning: could not open {p.name}: {e}")

    if not loaded:
        print("    No images loaded — skipping.")
        return

    cell_w = (CONTENT_W - H_GAP * (cols - 1)) // cols
    # Font size ~2% of cell width, minimum 24 px
    font_size = max(24, cell_w // 50)
    font = load_font(font_size)

    pages = layout_pages(loaded, filenames, cols, font)
    print(f"    → {len(pages)} page(s)")

    for i, page_img in enumerate(pages, 1):
        suffix = f"_page{i}" if len(pages) > 1 else ""
        out_path = output_dir / f"{name}{suffix}.png"
        page_img.save(out_path, dpi=(300, 300))
        print(f"    Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pack name_number.ext images into A4 grid PNGs for a LaTeX appendix."
    )
    parser.add_argument("--input_dir",  required=True,            help="Folder containing source images.")
    parser.add_argument("--output_dir", default="appendix_grids", help="Output folder (default: appendix_grids).")
    parser.add_argument("--cols",       type=int, default=3,      help="Images per row (default: 3).")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        print(f"Error: '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    groups = scan_groups(input_dir)
    if not groups:
        print("No files matching 'name_number.ext' found in the input directory.")
        sys.exit(1)

    selected_names = prompt_group_selection(groups)

    print(f"\nOutput directory: {output_dir}")
    for name in selected_names:
        process_group(name, groups[name], output_dir, cols=args.cols)

    print("\nDone.")


if __name__ == "__main__":
    main()