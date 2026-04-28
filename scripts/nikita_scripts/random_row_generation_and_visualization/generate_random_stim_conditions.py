import random
import json
import argparse
from pathlib import Path

"""
Script: generate_random_stim_conditions.py

Description:
    Generates a JSON file containing random stimulation conditions for the Cerebellum project.
    Each condition consists of a random selection of "groups" of rows.
    
    The output format for each condition is:
    "Condition #: CATHODIC channels; ANODIC channels"
    
    Example Output Line:
    "3": "1-4 33-36 65-68;" 
    (Condition 3: 3 groups of 4 channels each, all Cathodic. No Anodic channels.)

Usage:
    python generate_random_stim_conditions.py [options]

Options:
    --N INT                  Number of conditions to generate (default: 60)
    --output STR             Output filename (default: random_stim_conditions.json)
    --seed INT               Random seed for reproducibility (default: 42)
    --groups-per-cond INT    Number of groups to select per condition (default: 12)
    --rows-per-group INT     Number of contiguous rows to combine into one group (default: 1)
                             (e.g., set to 2 to group 2 rows = 4 channels together)

channel Mapping Logic:
    - Total Channels: 128
    - Rows: 64 (Each row has 2 channels: [2k-1, 2k])
    - Groups: Rows are grouped adjacent to each other based on --rows-per-group.
    - Selection: Randomly samples groups-per-cond groups.
    - Formatting: Compact string with ranges (e.g. "1-4" instead of "1 2 3 4").

Example:
    python generate_random_stim_conditions.py --N 100 --groups-per-cond 6 --rows-per-group 2

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
    parser.add_argument("--groups-per-cond", type=int, default=12, help="Number of groups to select per condition")
    parser.add_argument("--rows-per-group", type=int, default=1, help="Number of rows per group (1 row = 2 channels)")
    parser.add_argument("--rows-per-cond", type=int, help=argparse.SUPPRESS) # Backwards compat
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    groups_per_cond = args.rows_per_cond if args.rows_per_cond is not None else args.groups_per_cond
    
    # 1. Define Rows (Pairs of channels)
    # Total 128 channels. 64 rows.
    # Logic: Row k has channels [2k-1, 2k]
    all_rows = []
    for r in range(1, 65):
        ch1 = 2*r - 1
        ch2 = 2*r
        all_rows.append([ch1, ch2])
        
    conditions = {}
    
    num_channels_per_cond = groups_per_cond * args.rows_per_group * 2
    print(f"Generating {args.N} conditions (Seed: {args.seed})...")
    print(f"Using {args.rows_per_group} row(s) per group ({args.rows_per_group * 2} channels per group).")
    print(f"Selecting {groups_per_cond} group(s) per condition. Total {num_channels_per_cond} channels per condition.")
    
    # Validation check for max possible non-overlapping groups
    max_possible_groups = len(all_rows) // args.rows_per_group
    if groups_per_cond > max_possible_groups:
        raise ValueError(f"Cannot mutually exclusively select {groups_per_cond} groups of size {args.rows_per_group} from {len(all_rows)} rows.")
    
    for i in range(1, args.N + 1):
        selected_groups = []
        available_starts = list(range(len(all_rows) - args.rows_per_group + 1))
        
        while len(selected_groups) < groups_per_cond:
            start_idx = random.choice(available_starts)
            group_rows = all_rows[start_idx:start_idx+args.rows_per_group]
            
            group_channels = []
            for r in group_rows:
                group_channels.extend(r)
            selected_groups.append(group_channels)
            
            # Remove any overlapping start indices to ensure the chosen groups are completely distinct
            to_remove = [s for s in available_starts if abs(s - start_idx) < args.rows_per_group]
            for s in to_remove:
                available_starts.remove(s)
        
        # Flatten to channel list
        channels = []
        for group in selected_groups:
            channels.extend(group)
            
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
