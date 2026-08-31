"""
    Because FLIR cameras could drop frames, we use OCR to get a mapping of the frames to the actual times.
    Inputs:
        1. Two camera .avi files
    Outputs:
        1. .csv files that contains the mapping of frames to correct frames/times
"""
# FlirSync_EasyOCR_folder.py
# Reads one video or a folder of videos and writes per-video CSVs that include:
# AVI_framenum, OCR_framenum, CORRECTED_framenum, delta_t, n_missing,
# total_missing, n_chunks_missing, all_text

import sys
from pathlib import Path
import csv
import cv2, numpy as np
import easyocr

VIDEO_EXTS = {'.avi'}


def ZEROS(s: str) -> str:
    """Replace letter 'o'/'O' with zero to stabilize OCR numerics."""
    return s.lower().replace('o', '0')


def process_video(path: Path, reader: easyocr.Reader, out_csv: Path):
    fname = str(path)
    V = cv2.VideoCapture(fname)
    if not V.isOpened():
        print(f"[error] Could not open video: {fname}", file=sys.stderr)
        return

    nframes = int(round(V.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"[info] File {fname} has {nframes} frames.", file=sys.stderr)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        # Per-video header filename is implied by the CSV path
        w.writerow([
            'AVI_framenum', 'OCR_framenum', 'CORRECTED_framenum',
            'delta_t', 'n_missing', 'total_missing', 'n_chunks_missing', 'all_text'
        ])

        avi_fnum = -1
        total_missing = 0
        n_chunks_missing = 0

        while True:
            avi_fnum += 1
            ret, frame = V.read()
            if not ret:
                break

            # Crop the text band at the top
            ftext = frame[0:23, 4:]
            ftext = cv2.cvtColor(ftext, cv2.COLOR_BGR2GRAY)
            _, ftext = cv2.threshold(ftext, 50, 255, cv2.THRESH_BINARY)
            ftext = cv2.dilate(ftext, np.ones((2, 2), np.uint8), iterations=1)

            results = reader.readtext(ftext)

            if len(results) == 3:
                # Parse fields robustly
                ocr_fnum = ZEROS(results[0][1]).strip()

                delta_t = ''
                delta_t_str = results[1][1].strip()
                if delta_t_str[:3].lower() == 'dt:' and delta_t_str[-4:].lower() == 'msec':
                    delta_t = ZEROS(delta_t_str[3:-4]).strip()

                try:
                    n_missing = round(float(delta_t) / 10.0 - 1.0) if delta_t else 0
                except Exception:
                    n_missing = 0

                if n_missing > 0:
                    n_chunks_missing += 1
                total_missing += n_missing

                corrected_fnum = avi_fnum + total_missing
                text = ZEROS(' '.join(r[1] for r in results))

                w.writerow([
                    avi_fnum, ocr_fnum, corrected_fnum,
                    delta_t, n_missing, total_missing, n_chunks_missing, text
                ])
            else:
                # Bad OCR for this frame — keep row minimal
                w.writerow([avi_fnum, '', '', '', '', total_missing, n_chunks_missing, ''])

            if not avi_fnum % 100:
                print(
                    f'[{path.name}] Frame: {avi_fnum}/{nframes}        \r',
                    file=sys.stderr,
                    end='',
                    flush=True
                )

    V.release()
    print(f"[done] {fname} -> {out_csv}", file=sys.stderr)


def get_pipeline_video_root() -> Path:
    """
    Get current session's Video folder from your pipeline config.

    This uses config_loading.py, where you already have:

        VIDEO_ROOT = SESSION_LOC / "Video"

    This function is only used when no command-line path is provided.
    """
    try:
        from RCP_analysis.python.functions.config_loading import VIDEO_ROOT
    except Exception as e:
        raise RuntimeError(
            "Could not import VIDEO_ROOT from config_loading. "
            "If running manually, pass a video file or folder path."
        ) from e

    if VIDEO_ROOT is None:
        raise RuntimeError(
            "VIDEO_ROOT is None. Check config_loading.py and make sure one session "
            "is marked Yes in data_status_reaching.csv."
        )

    video_root = Path(VIDEO_ROOT)

    if not video_root.exists():
        raise FileNotFoundError(f"Video folder not found: {video_root}")

    return video_root


def main():
    # OLD behavior:
    #   python FlirSync_EasyOCR_folder.py /path/to/video-or-folder
    #
    # NEW pipeline behavior:
    #   python FlirSync_EasyOCR_folder.py
    #
    # If no argument is passed, use VIDEO_ROOT from config_loading.py.

    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
        print(f"[info] Using command-line target: {target}", file=sys.stderr)
    else:
        target = get_pipeline_video_root()
        print(f"[info] Using pipeline VIDEO_ROOT: {target}", file=sys.stderr)

    print('Initializing OCR...\r', file=sys.stderr, end='', flush=True)
    try:
        READER = easyocr.Reader(['en'], gpu=True)
    except Exception:
        print('[warn] GPU init failed; falling back to CPU', file=sys.stderr)
        READER = easyocr.Reader(['en'], gpu=False)
    print('OCR ready.                                      ', file=sys.stderr)

    if target.is_file():
        ocr_root = target.parent / "OCR"
        ocr_root.mkdir(parents=True, exist_ok=True)
        out_csv = ocr_root / f"{target.stem}_ocr.csv"

        # Skip if output already exists
        if out_csv.exists():
            print(f"[skip] Output already exists: {out_csv}", file=sys.stderr)
        else:
            process_video(target, READER, out_csv)

    elif target.is_dir():
        files = sorted(
            p for p in target.rglob('*')
            if p.suffix.lower() in VIDEO_EXTS
        )

        # Do not recursively process anything inside the OCR output folder
        files = [
            p for p in files
            if "OCR" not in p.relative_to(target).parts
        ]

        if not files:
            print(
                f"[warn] No video files with extensions {sorted(VIDEO_EXTS)} under {target}",
                file=sys.stderr
            )
            return

        ocr_root = target / "OCR"
        for p in files:
            # Mirror subdirectory structure under OCR/
            rel = p.relative_to(target)
            out_dir = ocr_root / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{p.stem}_ocr.csv"

            if out_csv.exists():
                print(f"[skip] Output already exists: {out_csv}", file=sys.stderr)
                continue

            process_video(p, READER, out_csv)

    else:
        print(f"[error] Path not found: {target}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()