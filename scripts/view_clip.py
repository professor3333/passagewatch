"""Replay a CFC clip with its annotations: an MP4 video, or a PNG contact sheet.

Examples:
    uv run python scripts/view_clip.py --location kenai-train --list
    uv run python scripts/view_clip.py --location kenai-train --clip <name>
    uv run python scripts/view_clip.py --location kenai-train --clip <name> --sheet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from passagewatch.ingestion.cfc import LOCATIONS, CfcLayout, load_clip
from passagewatch.visualization.overlay import contact_sheet, write_video

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--location", choices=LOCATIONS, required=True)
    parser.add_argument("--clip", help="clip name (see --list); default: the first clip")
    parser.add_argument("--list", action="store_true", help="list the clips and exit")
    parser.add_argument("--sheet", action="store_true", help="write a PNG contact sheet")
    parser.add_argument("--out", type=Path, default=None, help="default: data/cache/viewer/")
    args = parser.parse_args(argv)

    layout = CfcLayout.tiny(args.extract_dir)
    clips = layout.metadata(args.location).clips
    if args.list:
        for name in sorted(clips):
            print(name)
        return 0
    name = args.clip or min(clips)
    if name not in clips:
        parser.error(f"unknown clip {name!r} in {args.location}; use --list")

    clip = load_clip(layout, args.location, clips[name])
    suffix = ".png" if args.sheet else ".mp4"
    out = args.out or REPO_ROOT / "data/cache/viewer" / f"{args.location}__{name}{suffix}"
    if args.sheet:
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), contact_sheet(clip)):
            print(f"failed to write {out}", file=sys.stderr)
            return 1
        print(f"contact sheet of frames {clip.frame_start}-{clip.frame_stop - 1} -> {out}")
    else:
        written = write_video(clip, out)
        print(f"{written} frames ({clip.frame_start}-{clip.frame_stop - 1}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
