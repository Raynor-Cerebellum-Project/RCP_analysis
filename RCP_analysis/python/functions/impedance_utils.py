from pathlib import Path
import csv

def _get_bad_channels_utah(file_path, high_threshold=800, exclude_low_imp=True):
    """
    Parses a Utah Array impedance file and returns a set of excluded channel IDs.
    
    Args:
        file_path (Path): Path to the impedance file.
        high_threshold (float): Impedance value (kOhm) above which to exclude. Default 800.
        exclude_low_imp (bool): If True, exclude channels <= 15 kOhm. Default True.
    
    Criteria:
    - Impedance > high_threshold kOhm
    - Impedance <= 15 kOhm (if exclude_low_imp is True)
    
    Format example:
      elec2-124 15 kOhm
      elec4-245 <= 15kOhm
    """
    bad_channels = set()
    if not file_path.exists():
        print(f"[WARN] Utah impedance file not found: {file_path}")
        return bad_channels
        
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('Chan'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # Format examples: 
            # elec2-124 15 kOhm
            # elec4-245 <= 15kOhm
            
            chan_str = parts[0]
            val_str = parts[1]
            try:
                if '-' in chan_str:
                    ch_id = int(chan_str.split('-')[1])
                else:
                    continue

                is_less_equal = False
                if val_str == '<=':
                    is_less_equal = True
                    # If format is "<= 15kOhm", parts[2] is likely "15kOhm"
                    if len(parts) > 2:
                        val_num_str = parts[2].replace('kOhm', '')
                    else:
                        continue 
                else:
                    # Format "15 kOhm" -> parts[1]="15", parts[2]="kOhm" usually
                    # But code above handles "15" from parts[1] if parts[1] is just number
                    val_num_str = val_str.replace('kOhm', '') # Should be safe
                
                try:
                    imp_val = float(val_num_str)
                except:
                    continue 
                
                # Criteria
                if is_less_equal:
                    # Treat "<= X" as low impedance.
                    if exclude_low_imp and imp_val <= 15: 
                        bad_channels.add(ch_id)
                else:
                    if imp_val > high_threshold:
                        bad_channels.add(ch_id)
                    elif exclude_low_imp and imp_val <= 15:
                        bad_channels.add(ch_id)

            except Exception:
                continue
    return bad_channels

def _get_bad_channels_intan(file_path, high_threshold=4.5e6):
    """
    Parses an Intan impedance CSV and returns a set of excluded channel IDs.
    
    Args:
        file_path (Path): Path to the impedance CSV.
        high_threshold (float): Impedance value (Ohms) above which to exclude. Default 4.5e6.
        
    Criteria:
    - Impedance > high_threshold Ohm
    - ID derived from last 3 digits of 'Channel Name' (col 1, index 1)
    """
    bad_channels = set()
    if not file_path.exists():
        print(f"[WARN] Intan impedance file not found: {file_path}")
        return bad_channels
        
    with open(file_path, 'r', newline='') as f:
        rdr = csv.reader(f)
        try:
            header = next(rdr)
        except StopIteration:
            return bad_channels
            
        for row in rdr:
            if len(row) < 5: continue
            try:
                name = row[1] # "NPXL-015"
                imp_str = row[4] # "2.83e+05"
                
                digits = "".join([c for c in name if c.isdigit()])
                if not digits: continue
                ch_id = int(digits[-3:]) 
                
                imp_val = float(imp_str)
                
                # Criteria: > high_threshold Ohm
                if imp_val > high_threshold:
                    bad_channels.add(ch_id)
            except Exception:
                continue
    return bad_channels

def get_session_impedances(session_loc, 
                           utah_high_thresh=800, 
                           utah_exclude_low=True, 
                           intan_high_thresh=4.5e6):
    """
    Looks for 'Impedances' folder in session_loc and parses standard files.
    
    Args:
        session_loc (Path): Root directory of the session.
        utah_high_thresh (float): Max impedance for Utah (kOhm).
        utah_exclude_low (bool): Whether to exclude Utah channels <= 15 kOhm.
        intan_high_thresh (float): Max impedance for Intan (Ohm).
        
    Returns:
        dict: {'utah': set(bad_ids), 'nprw': set(bad_ids)}
    """
    bad_map = {'utah': set(), 'nprw': set()}
    
    if session_loc is None:
        return bad_map
        
    session_loc = Path(session_loc)
    imp_dir = session_loc / "Impedances"
    
    if not imp_dir.exists():
        # Try finding it? Or just return empty
        # Sometimes folder might be named differently? Assuming standard "Impedances"
        return bad_map
        
    utah_imp_file = imp_dir / "Utah_imp_start_PortA"
    intan_imp_file = imp_dir / "Intan_imp_start.csv"
    
    if utah_imp_file.exists():
        bad_map['utah'] = _get_bad_channels_utah(utah_imp_file, 
                                                high_threshold=utah_high_thresh, 
                                                exclude_low_imp=utah_exclude_low)
        
    if intan_imp_file.exists():
        bad_map['nprw'] = _get_bad_channels_intan(intan_imp_file, 
                                                 high_threshold=intan_high_thresh)
        
    return bad_map
