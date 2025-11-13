#!/usr/bin/env python3
# FlirSync_EasyOCR_newdisplay.py
# Parses new FLIR overlay format:
#   "<frame>   dt:<Δt>mSec   <FPS>FPS   #drop:<N>"

import sys
from pathlib import Path
import csv
import cv2, numpy as np
import easyocr
import re
import statistics
from collections import deque

VIDEO_EXTS = {'.avi'}
def _bbox_left_x(bbox):
    # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    try:
        xs = [pt[0] for pt in bbox]
        return min(xs)
    except Exception:
        return 1e12  # very right

_INT_RE = re.compile(r'^\d{1,8}$')

def _best_frame_from_tokens(results, last_expected=None, min_conf=0.45):
    """
    Choose the left-most high-confidence integer token as the frame number.
    If multiple candidates: prefer higher confidence; tie-break by more-left.
    If last_expected is provided and a candidate is close (±3), prefer it.
    Returns (frame_str or '', used_fallback: bool).
    """
    candidates = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        bbox, txt, conf = item[0], item[1], item[2]
        if conf is None:
            continue
        t = ZEROS(txt)
        t = t.strip()
        # keep pure integers only (frame counter has no decimal)
        if not _INT_RE.match(t):
            continue
        if conf < min_conf:
            continue
        x = _bbox_left_x(bbox)
        candidates.append((t, conf, x))

    if not candidates:
        return '', True

    # If we know what to expect (last OCR + 1), bias toward that
    if last_expected is not None:
        near = [(t, conf, x) for (t, conf, x) in candidates
                if abs(int(t) - last_expected) <= 3]
        if near:
            # pick highest conf, then left-most
            near.sort(key=lambda z: (-z[1], z[2]))
            return near[0][0], False

    # Otherwise: highest conf, then left-most
    candidates.sort(key=lambda z: (-z[1], z[2]))
    return candidates[0][0], False

def ZEROS(s: str) -> str:
    """
    Normalize typical OCR confusions.
    - o/O -> 0, i/I/l/|/! -> 1, s/S -> 5, q/g -> 9, B -> 8
    - comma -> dot, sfps -> 5fps, extra spaces removed
    """
    if s is None:
        return ""
    t = s
    t = t.replace(',', '.')
    t = re.sub(r'sfps', '5fps', t, flags=re.I)
    trans = str.maketrans({
        'o':'0','O':'0',
        'i':'1','I':'1','l':'1','|':'1','!':'1',
        's':'5','S':'5',
        'q':'9','g':'9',
        'B':'8',
    })
    t = t.translate(trans)
    t = re.sub(r'\s+fps', 'fps', t, flags=re.I)
    return t.strip()

def _interpolate_ocr(ocr_list: list[int | None]) -> tuple[list[int | None], list[bool]]:
    """
    Returns (ocr_interp, invalid_flags). invalid_flags[i] == True if the original
    ocr_list[i] was missing/garbled or failed the 'between neighbors' rule.
    Interpolation is linear between valid anchors; edges are extrapolated with
    the last known step (falling back to +1).
    """
    n = len(ocr_list)
    ocr = list(ocr_list)  # copy
    invalid = [False] * n

    # Helper to find previous/next valid indices
    def prev_valid(i):
        j = i - 1
        while j >= 0 and ocr[j] is None:
            j -= 1
        return j
    def next_valid(i):
        j = i + 1
        while j < n and ocr[j] is None:
            j += 1
        return j if j < n else -1

    # Mark missing as invalid
    for i in range(n):
        if ocr[i] is None:
            invalid[i] = True

    # Mark "not-between" cases invalid when both neighbors exist
    for i in range(1, n - 1):
        if ocr[i] is None:
            continue
        pv = prev_valid(i)
        nv = next_valid(i)
        if pv >= 0 and nv >= 0:
            a, b, x = ocr[pv], ocr[nv], ocr[i]
            # If x is not within [min(a,b), max(a,b)] (inclusive), mark invalid
            lo, hi = (a, b) if a <= b else (b, a)
            if not (lo <= x <= hi):
                invalid[i] = True

    # Interpolate runs of invalids between valid anchors
    i = 0
    while i < n:
        if not invalid[i]:
            i += 1
            continue
        # start of invalid run
        run_start = i
        while i < n and invalid[i]:
            i += 1
        run_end = i - 1  # inclusive

        pv = prev_valid(run_start)
        nv = next_valid(run_end)

        if pv >= 0 and nv >= 0:
            # Linear interpolation between ocr[pv] and ocr[nv]
            a, b = ocr[pv], ocr[nv]
            span = nv - pv
            step = (b - a) / span if span != 0 else 0.0
            for k in range(run_start, run_end + 1):
                t = (k - pv) * step
                ocr[k] = int(round(a + t))
        elif pv >= 0 and nv < 0:
            # Extrapolate forward using last step if possible (prefer +1)
            # Try to infer step from last two valids
            pre2 = prev_valid(pv)
            step = (ocr[pv] - ocr[pre2]) if (pre2 >= 0) else 1
            if step == 0:
                step = 1
            val = ocr[pv]
            for k in range(pv + 1, run_end + 1):
                val += step
                if k >= run_start:
                    ocr[k] = val
        elif pv < 0 and nv >= 0:
            # Extrapolate backward from next valid using inferred step (prefer -1)
            nxt2 = next_valid(nv)
            step = (ocr[nv] - ocr[nxt2]) if (nxt2 >= 0) else -1
            if step == 0:
                step = -1
            val = ocr[nv]
            for k in range(nv - 1, run_start - 1, -1):
                val -= step
                if k <= run_end:
                    ocr[k] = val
        else:
            # No anchors at all: just create a simple increasing sequence starting at 0
            val = 0
            for k in range(run_start, run_end + 1):
                ocr[k] = val
                val += 1

    return ocr, invalid


def process_video(path: Path, reader: easyocr.Reader, out_csv: Path):
    V = cv2.VideoCapture(str(path))
    if not V.isOpened():
        print(f"[error] Could not open video: {path}", file=sys.stderr)
        return

    nframes = int(round(V.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"[info] Processing {path.name} ({nframes} frames)", file=sys.stderr)

    rows = []  # we’ll write after interpolation

    avi_fnum = -1
    total_missing = 0
    n_chunks_missing = 0

    fps_buf = deque(maxlen=25)
    dt_buf  = deque(maxlen=25)
    prev_corrected = None

    while True:
        avi_fnum += 1
        ret, frame = V.read()
        if not ret:
            break

        ftext = frame[0:30, 0:700]
        ftext = cv2.cvtColor(ftext, cv2.COLOR_BGR2GRAY)
        _, ftext = cv2.threshold(ftext, 60, 255, cv2.THRESH_BINARY)
        ftext = cv2.dilate(ftext, np.ones((2, 2), np.uint8), iterations=1)

        results = reader.readtext(ftext)
        text = " ".join(r[1] for r in results)
        text = ZEROS(text)

        # --- tolerant parsing ---
        ocr_fnum, delta_t, fps_val = '', '', ''

        last_expected = None
        if prev_corrected is not None:
            last_expected = prev_corrected + 1
        token_frame, token_fallback = _best_frame_from_tokens(results, last_expected=last_expected)

        if token_frame:
            ocr_fnum = token_frame
        else:
            m = re.search(
                r'(?P<frame>\d+)\s+dt[: ]\s*(?P<dt>[0-9.]+)\s*m?s?e?c\s+(?P<fps>[0-9.]+)\s*fp?s?',
                text, re.IGNORECASE
            )
            if not m:
                m = re.search(
                    r'(?P<frame>\d+)\s+dt[: ]\s*(?P<dt>[0-9.]+)\s*m?s?e?c',
                    text, re.IGNORECASE
                )
            if not m:
                m = re.search(r'\b(\d{1,7})\b', text)
            if m:
                if 'frame' in getattr(m, "groupdict", lambda: {})():
                    ocr_fnum = m.group('frame')
                elif m.groups():
                    ocr_fnum = m.group(1)
                if 'dt' in getattr(m, "groupdict", lambda: {})():
                    delta_t = m.group('dt')
                if 'fps' in getattr(m, "groupdict", lambda: {})():
                    fps_val = m.group('fps')

        if token_frame and not delta_t:
            m_dt = re.search(r'dt[: ]\s*([0-9.]+)\s*m?s?e?c', text, re.I)
            if m_dt: delta_t = m_dt.group(1)
        if token_frame and not fps_val:
            m_fps = re.search(r'([0-9.]+)\s*fp?s?', text, re.I)
            if m_fps: fps_val = m_fps.group(1)

        # ---- fps/dt smoothing + fps-aware n_missing ----
        dt_float = None
        fps_float = None
        try:
            if delta_t:
                v = float(delta_t)
                if 5.0 <= v <= 40.0:
                    dt_float = v
                    dt_buf.append(v)
        except Exception:
            pass
        try:
            if fps_val:
                v = float(fps_val)
                if 20.0 <= v <= 200.0:
                    fps_float = v
                    fps_buf.append(v)
        except Exception:
            pass

        fps_med = statistics.median(fps_buf) if fps_buf else None
        dt_med  = statistics.median(dt_buf)  if dt_buf  else None

        base_ms = None
        if fps_med is not None:
            if fps_med >= 80.0:
                base_ms = 10.0
            elif 35.0 <= fps_med <= 65.0:
                base_ms = 20.0
        if base_ms is None and dt_med is not None:
            base_ms = 10.0 if dt_med < 15.0 else 20.0
        if base_ms is None:
            base_ms = 10.0

        if dt_float is not None:
            n_missing = max(0, round(dt_float / base_ms - 1.0))
        else:
            n_missing = 0

        if fps_med is not None:
            fps_val = f"{fps_med:.1f}"

        if n_missing > 0:
            n_chunks_missing += 1
        total_missing += n_missing
        corrected_fnum = avi_fnum + total_missing

        # Optional one-row override from raw OCR (we'll later re-compare with interpolated)
        try:
            ocr_int = int(ocr_fnum) if ocr_fnum and ocr_fnum.isdigit() else None
        except Exception:
            ocr_int = None

        if ocr_int is not None:
            if prev_corrected is None or ocr_int > prev_corrected:
                if ocr_int > avi_fnum:
                    corrected_fnum = ocr_int

        if prev_corrected is not None:
            min_allowed = prev_corrected + 1
            max_allowed = prev_corrected + 1 + max(0, int(n_missing))
            if corrected_fnum < min_allowed:
                corrected_fnum = min_allowed
            elif corrected_fnum > max_allowed:
                corrected_fnum = max_allowed

        rows.append({
            "AVI_framenum": avi_fnum,
            "OCR_framenum_raw": ocr_int,
            "CORRECTED_framenum": int(corrected_fnum),
            "delta_t": delta_t,
            "fps": fps_val,
            "n_missing": n_missing,
            "total_missing": total_missing,
            "n_chunks_missing": n_chunks_missing,
            "all_text": text,
        })

        try:
            prev_corrected = int(corrected_fnum)
        except Exception:
            prev_corrected = None

        if not avi_fnum % 100:
            print(f"[{path.name}] Frame {avi_fnum}/{nframes}\r",
                  end='', file=sys.stderr, flush=True)

    V.release()

    # ---------- Pass 2: interpolate/clean OCR_framenum ----------
    raw_ocr = [r["OCR_framenum_raw"] for r in rows]
    ocr_interp, invalid_flags = _interpolate_ocr(raw_ocr)

    # Optional: re-apply the "OCR jump ahead" override using the **interpolated** OCR
    # ---------- Pass 2: interpolate/clean OCR_framenum ----------
    raw_ocr = [r["OCR_framenum_raw"] for r in rows]
    ocr_interp, invalid_flags = _interpolate_ocr(raw_ocr)

    prev_corr = None
    for i, r in enumerate(rows):
        avi   = r["AVI_framenum"]
        ocr_i = ocr_interp[i]

        # 1) If OCR is valid and EXACTLY equals AVI, force equality.
        if (ocr_i is not None) and (not invalid_flags[i]) and (ocr_i == avi):
            corr = avi  # = ocr_i
        # 2) Otherwise, if OCR is valid and ahead of AVI (and not retrograde vs prev), use it.
        elif (ocr_i is not None) and (not invalid_flags[i]) and \
            (ocr_i >= avi) and (prev_corr is None or ocr_i >= prev_corr + 1):
            corr = int(ocr_i)
        # 3) Fallback: monotone progression (at least prev+1, never below AVI).
        else:
            base = (prev_corr + 1) if prev_corr is not None else avi
            corr = max(base, avi)

        r["CORRECTED_framenum"] = corr
        prev_corr = corr

    # ---------- Pass 3: backward strictly-increasing fix ----------
    corr = [r["CORRECTED_framenum"] for r in rows]
    for i in range(len(corr) - 2, -1, -1):   # from n-2 down to 0
        if corr[i] >= corr[i + 1]:
            corr[i] = corr[i + 1] - 1        # enforce corr[i] < corr[i+1]

    # (optional) keep non-negative if you want:
    # corr = [max(0, x) for x in corr]

    for i, c in enumerate(corr):
        rows[i]["CORRECTED_framenum"] = int(c)

    # ---------- Write CSV (now with interpolated OCR + validity flag) ----------
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            'AVI_framenum','OCR_framenum','OCR_interp','OCR_invalid',
            'CORRECTED_framenum','delta_t','fps','n_missing','total_missing',
            'n_chunks_missing','all_text'
        ])
        for r, oi, bad in zip(rows, ocr_interp, invalid_flags):
            w.writerow([
                r["AVI_framenum"],
                "" if r["OCR_framenum_raw"] is None else r["OCR_framenum_raw"],
                "" if oi is None else oi,
                int(bad),
                r["CORRECTED_framenum"],
                r["delta_t"], r["fps"], r["n_missing"], r["total_missing"],
                r["n_chunks_missing"], r["all_text"]
            ])

    print(f"[done] {path.name} -> {out_csv}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("Usage: python FlirSync_EasyOCR_newdisplay.py <video-file-or-folder>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    print("Initializing EasyOCR...", file=sys.stderr)
    try:
        READER = easyocr.Reader(['en'], gpu=True)
    except Exception:
        print("[warn] GPU init failed; using CPU", file=sys.stderr)
        READER = easyocr.Reader(['en'], gpu=False)

    if target.is_file():
        out_csv = target.with_name(f"{target.stem}_ocr.csv")
        process_video(target, READER, out_csv)
    elif target.is_dir():
        files = sorted(p for p in target.rglob('*') if p.suffix.lower() in VIDEO_EXTS)
        if not files:
            print(f"[warn] No videos found under {target}", file=sys.stderr)
            return
        for p in files:
            out_csv = p.with_name(f"{p.stem}_ocr.csv")
            process_video(p, READER, out_csv)
    else:
        print(f"[error] Path not found: {target}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
