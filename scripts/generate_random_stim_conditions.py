import random
import json
import argparse
from pathlib import Path

"""
Script: generate_random_stim_conditions.py

Description:
    Generates a JSON file containing random stimulation conditions for the Cerebellum project.
    Each condition consists of a random selection of "rows" (pairs of adjacent channels).
    
    The output format for each condition is:
    "Condition #: CATHODIC channels; ANODIC channels"
    
    Example Output Line:
    "34": "1-2 33-34 65-66 23-24;" 
    (Condition 34: Channels 1, 2, 33, 34, 65, 66, 23, 24 are Cathodic. No Anodic channels.)

Usage:
    python generate_random_stim_conditions.py [options]

Options:
    --N INT             Number of conditions to generate (default: 60)
    --output STR        Output filename (default: random_stim_conditions.json)
    --seed INT          Random seed for reproducibility (default: 42)
    --rows-per-cond INT Number of channel pairs (rows) per condition (default: 12)

channel Mapping Logic:
    - Total Channels: 128
    - Rows: 64 (Each row has 2 channels: [2k-1, 2k])
    - Selection: Randomly samples N rows and combines their channels.
    - Formatting: Compact string with ranges (e.g. "1-2" instead of "1 2").

Example:
    python generate_random_stim_conditions.py --N 100 --seed 123 --output my_experiment.json
"""

def format_channels_compact(channels):
    """
    Format a list of sorted integers into a space-separated string with ranges.
    E.g. [1, 2, 5, 6, 7] -> "1-2 5-7"
    """
    if not channels:
        return ""
        
    channels = sorted(list(set(channels)))
    ranges = []
    
    start = channels[0]
    prev = channels[0]
    
    for x in channels[1:]:
        if x == prev + 1:
            prev = x
        else:
            # End of previous range
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = x
            prev = x
            
    # Add final range
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")
        
    return " ".join(ranges)

def main():
    parser = argparse.ArgumentParser(description="Generate random stimulation condition files.")
    parser.add_argument("--N", type=int, default=60, help="Number of conditions to generate")
    parser.add_argument("--output", type=str, default="random_stim_conditions.json", help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--rows-per-cond", type=int, default=12, help="Number of rows (pairs) to select per condition")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 1. Define Rows (Pairs of channels)
    # Total 128 channels. 64 rows.
    # Logic: Row k has channels [2k-1, 2k]
    # Rows are 1-indexed for convenience in logic, but internal mapping is just lists.
    # Row 1: [1, 2]
    # ...
    # Row 64: [127, 128]
    all_rows = []
    for r in range(1, 65):
        ch1 = 2*r - 1
        ch2 = 2*r
        all_rows.append([ch1, ch2])
        
    conditions = {}
    
    print(f"Generating {args.N} conditions (Seed: {args.seed})...")
    
    for i in range(1, args.N + 1):
        # Sample random rows
        selected_rows = random.sample(all_rows, args.rows_per_cond)
        
        # Flatten to channel list
        channels = []
        for row in selected_rows:
            channels.extend(row)
            
        # Format string
        # "CATHODIC channels; ANODIC channels"
        # User requested all cathodic first. Anodic empty.
        cathodic_str = format_channels_compact(channels)
        anodic_str = "" # Empty
        
        val_str = f"{cathodic_str}; {anodic_str}".strip()
        
        # Key: Condition Number
        conditions[str(i)] = val_str
        
    # Write output
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(conditions, f, indent=4)
        
    print(f"Saved to {out_path.absolute()}")
    
    # Preview
    first_key = list(conditions.keys())[0]
    print(f"\nPreview Condition {first_key}:\n{first_key}: {conditions[first_key]}")

if __name__ == "__main__":
    main()
