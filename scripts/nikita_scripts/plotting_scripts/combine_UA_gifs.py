import os
import pathlib
import imageio
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import re
import RCP_analysis.python.functions.config_loading as cfg

# --- Configuration ---
# Derived from project config to avoid hardcoding
SEARCH_DIRS = [
    cfg.OUT_BASE / "figures" / "UA_LFP" / "Spatial_GIF", # LFP Bands
    cfg.OUT_BASE / "figures" / "peristim_new_npz"             # Firing Rates
]

OUTPUT_ROOT = cfg.OUT_BASE / "figures" / "UA_LFP" / "Quad_GIFs"

# Canvas setup
CANVAS_SIZE = (1200, 1200) # (width, height)
BG_COLOR = (255, 255, 255) # White background

# Layout mapping based on anatomical schematic
# Centers are normalized (0-1) where (0,0) is top-left
LAYOUT = {
    "SMA": {"center": (0.50, 0.25), "angle": 20},  # Top-Middle
    "M1s": {"center": (0.80, 0.20), "angle": 0},   # Top-Right
    "PMd": {"center": (0.25, 0.70), "angle": 30},  # Bottom-Left
    "M1i": {"center": (0.65, 0.80), "angle": 5}   # Bottom-Middle/Right
}

# Scaling factor for the individual array GIFs relative to the canvas width
# (approx 30% of canvas width)
THUMB_SCALE = 0.30

# --- Refinement Specs ---
# Region-specific cropping to reduce overlap footprint and trim labels.
# (left, top, right, bottom) as fraction of size.
REGION_CROP_MARGINS = {
    "M1i": (0.09, 0.05, 0.02, 0),
    "M1s": (0.09, 0.05, 0.02, 0),
    "PMd": (0, 0.05, 0.02, 0),
    "SMA": (0, 0.05, 0.18, 0)
}

def combine_gifs(session_id, datatype, mode, region_paths):
    """
    Combines 4 array GIFs into an anatomical quad view with consistent scaling.
    """
    
    # Load all 4 GIFs and determine max frame count
    region_frames = {}
    region_orig_size = {} # Store original (w, h) to maintain physical scale
    fps = 10
    max_frames = 0
    
    for reg, path in region_paths.items():
        try:
            reader = imageio.get_reader(path)
            meta = reader.get_meta_data()
            if 'fps' in meta:
                fps = meta['fps']
            
            # Get margins for this specific region
            crop_margins = REGION_CROP_MARGINS.get(reg, (0, 0, 0, 0))
            
            frames = []
            orig_w, orig_h = 0, 0
            for f in reader:
                img = Image.fromarray(f).convert("RGBA")
                if orig_w == 0:
                    orig_w, orig_h = img.size
                    region_orig_size[reg] = (orig_w, orig_h)
                
                # 1. Smart Crop (Remove white margins / user-specified labels)
                w, h = img.size
                left = int(w * crop_margins[0])
                top = int(h * crop_margins[1])
                right = int(w * (1.0 - crop_margins[2]))
                bottom = int(h * (1.0 - crop_margins[3]))
                img = img.crop((left, top, right, bottom))
                
                frames.append(img)
                
            region_frames[reg] = frames
            max_frames = max(max_frames, len(frames))
            reader.close()
        except Exception as e:
            print(f"    [Error] Failed to load {reg} GIF from {path.name}: {e}")
            return None

    # Compose frames
    combined_frames = []
    
    for i in range(max_frames):
        canvas = Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 255))
        
        for reg, config in LAYOUT.items():
            if reg not in region_frames:
                continue
                
            # Get frame (hold if shorter than others)
            frames = region_frames[reg]
            frame_idx = min(i, len(frames) - 1)
            img = frames[frame_idx].copy()
            
            # 2. Resize based on ORIGINAL size to keep physical scale consistent
            # We want the ORIGINAL image to have a scaling multiplier of (CanvasWidth * THUMB_SCALE / OriginalWidth)
            orig_w, _ = region_orig_size[reg]
            scale_factor = (CANVAS_SIZE[0] * THUMB_SCALE) / orig_w
            
            w, h = img.size
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 3. Rotate (expand=True ensures nothing is cropped, CCW rotation)
            # Use transparent fill color for the expanded corners
            img = img.rotate(config['angle'], expand=True, fillcolor=(255, 255, 255, 0))
            
            # 4. Paste at center of specified location using alpha mask
            cx_norm, cy_norm = config['center']
            cx = int(cx_norm * CANVAS_SIZE[0])
            cy = int(cy_norm * CANVAS_SIZE[1])
            
            paste_w, paste_h = img.size
            x = cx - (paste_w // 2)
            y = cy - (paste_h // 2)
            
            canvas.paste(img, (x, y), img)
            
        # Convert back to RGB for GIF saving
        combined_frames.append(np.array(canvas.convert("RGB")))

    # Save output
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_name = f"quad_heatmap_{datatype}_{session_id}_{mode}.gif"
    out_path = OUTPUT_ROOT / out_name
    
    # User requested to slow down by 1.5
    final_fps = fps / 1.5
    
    imageio.mimsave(out_path, combined_frames, fps=final_fps, loop=0)
    return out_path

def main():
    all_files = []
    for sdir in SEARCH_DIRS:
        if sdir.exists():
            all_files.extend(list(sdir.rglob("spatial_*.gif")))
        else:
            print(f"Directory not found, skipping: {sdir}")
            
    print(f"Found {len(all_files)} individual region GIFs across search directories.")

    # Index files: (datatype, session, mode) -> {region: path}
    file_map = {}
    
    for f in all_files:
        name = f.name
        # Match: spatial_heatmap_{datatype}_{session}_{region}_{mode}.gif
        # datatype could be 'beta', 'alpha', 'FR', etc.
        m = re.match(r"spatial_(?P<datatype>.+?)_(?P<session>.+)_(?P<region>SMA|PMd|M1i|M1s)_(?P<mode>blocky|smoothed)\.gif$", name)
        if m:
            datatype, session, region, mode = m.groups()
            key = (datatype, session, mode)
            if key not in file_map:
                file_map[key] = {}
            file_map[key][region] = f

    # Filter for complete sets (4 regions)
    complete_sets = {k: v for k, v in file_map.items() if len(v) == 4}
    
    print(f"Identified {len(complete_sets)} complete quad sets out of {len(file_map)} potential session/datatype groups.")
    
    if not complete_sets:
        # Diagnostic: what regions were found for the potential sets?
        for k, v in sorted(file_map.items())[:5]:
            print(f"  Partial match: {k} -> {list(v.keys())}")
        return

    items = sorted(complete_sets.items())

    with tqdm(total=len(items), desc="Combining quad GIFs", dynamic_ncols=True) as pbar:
        for (datatype, session, mode), region_paths in items:
            pbar.set_postfix({
                "datatype": datatype,
                "mode": mode,
                "session": session[:20]
            })

            combine_gifs(session, datatype, mode, region_paths)

            pbar.update(1)

if __name__ == "__main__":
    main()
