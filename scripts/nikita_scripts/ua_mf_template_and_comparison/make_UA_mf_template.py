import argparse
import sys
import os
from pathlib import Path

import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import spikeinterface.sorters as ss

# Add parent dir to path so we can import RCP_analysis if run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

def main():
    parser = argparse.ArgumentParser(description="Create Matched Filter templates for Utah Arrays via SpikeInterface GUI.")
    parser.add_argument("--session", type=str, required=True, help="BR session name (e.g. NRR_RW022_001) - Highly recommended to pick a continuous or baseline session.")
    parser.add_argument("--sorter", type=str, default="spykingcircus2", help="Spike sorter to use (default: spykingcircus2)")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to slice the recording to for fast sorting (default 60s)")
    args = parser.parse_args()

    sess_name = args.session
    clean_sess_name = sess_name.replace('.ns6', '').replace('.nsx6', '')
    sess_path = BR_ROOT / clean_sess_name
    
    # We check if .ns6 exists instead of the prefix directly
    if not sess_path.with_suffix('.ns6').exists() and not sess_path.exists():
        print(f"Error: Session file not found at {sess_path}.ns6")
        return

    print(f"Loading Blackrock session {sess_name}...")
    rec_ns6 = se.read_blackrock(sess_path, stream_name='nsx6', all_annotations=True)
    
    # Mapping
    try:
        clean_sess_name = sess_name.replace('.ns6', '').replace('.nsx6', '')
        br_idx = int(clean_sess_name.split('_')[-1])
    except ValueError:
        print("Could not parse numeric BR index from session name. Defaulting to 001")
        br_idx = 1
        
    XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
    UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
    
    if UA_MAP is None:
        raise RuntimeError("UA mapping required.")
        
    rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
    
    # Highpass
    print("Applying highpass filter...")
    rec_hp = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
    
    valid_mask = ua_elec > 0
    ch_ids = rec_hp.get_channel_ids()
    valid_ch_ids = [ch_ids[i] for i, valid in enumerate(valid_mask) if valid]
    
    print(f"Keeping {len(valid_ch_ids)} valid mapped channels...")
    rec_valid = rec_hp.select_channels(valid_ch_ids)
    
    port_a_ch = valid_ch_ids if ua_port == 'A' else []
    port_b_ch = valid_ch_ids if ua_port == 'B' else []

    # Slicing to speed up GUI template creation
    if args.duration > 0 and rec_valid.get_total_duration() > args.duration:
        print(f"Slicing down to first {args.duration} seconds to speed up sorting...")
        rec_valid = rec_valid.frame_slice(0, int(args.duration * rec_valid.get_sampling_frequency()))

    results_dir = UA_CKPT_ROOT / f"mf_templates_{sess_name}"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    conf_dir = REPO_ROOT / 'config'
    conf_dir.mkdir(exist_ok=True)

    for port_name, port_ch in [("PortA", port_a_ch), ("PortB", port_b_ch)]:
        if not port_ch:
            continue
            
        print(f"\n=======================================================")
        print(f"--- Processing {port_name} ({len(port_ch)} channels) ---")
        rec_port = rec_valid.select_channels(port_ch)
        
        # CMR
        rec_port = spre.common_reference(rec_port, reference="global", operator="median")

        # Set generic 2D channel locations for Kilosort (400um pitch typical for UA)
        n_ch = rec_port.get_num_channels()
        locs = np.zeros((n_ch, 2))
        side = int(np.ceil(np.sqrt(n_ch)))
        for i in range(n_ch):
            locs[i, 0] = (i % side) * 400.0
            locs[i, 1] = (i // side) * 400.0
        rec_port.set_channel_locations(locs)
        rec_port.set_property("group", np.zeros(n_ch, dtype=int))

        # Sorting
        sorting_out = results_dir / f"sorting_{port_name}"
        sorting = None
        if sorting_out.exists():
            print(f"Attempting to load existing sorting for {port_name}...")
            try:
                sorting = si.load_extractor(sorting_out / 'sorter_output')
            except Exception:
                try:
                    sorting = si.load_extractor(sorting_out)
                except Exception:
                    print("Could not load existing sorting. Re-running...")
                    import shutil
                    shutil.rmtree(sorting_out, ignore_errors=True)
                    sorting = None

        if sorting is None:
            print(f"Running {args.sorter} on {port_name}...")
            try:
                sorting = ss.run_sorter(args.sorter, rec_port, folder=str(sorting_out), remove_existing_folder=True, verbose=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed to run spike sorting: {e}")
                print(f"Hint: Make sure {args.sorter} is installed correctly or try another sorter.")
                return
        
        # Building Analyzer for SpikeInterface >= 0.100
        analyzer_out = results_dir / f"analyzer_{port_name}"
        if not analyzer_out.exists():
            print(f"Creating SortingAnalyzer for {port_name}...")
            analyzer = si.create_sorting_analyzer(sorting, rec_port, format="binary_folder", folder=str(analyzer_out), overwrite=True)
            print("Computing random spikes, waveforms, templates, and metrics...")
            analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
            analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0)
            analyzer.compute("templates")
            try:
                analyzer.compute("quality_metrics")
            except Exception as e:
                print(f"Skipping quality metrics due to error: {e}")
        else:
            print(f"Loading existing SortingAnalyzer for {port_name}...")
            analyzer = si.load_sorting_analyzer(str(analyzer_out))
            
        unit_ids = analyzer.unit_ids
        print(f"Found {len(unit_ids)} units.")
        if len(unit_ids) == 0:
            print("No units found! Skipping.")
            continue
        
        try:
            import spikeinterface_gui as sig
            print("\n*** Launching SpikeInterface GUI... ***")
            print("INSTRUCTIONS:")
            print(" - Explore the units in the GUI.")
            print(" - Look for a clean single unit with high SNR and a good canonical spike shape.")
            print(" - Note the Unit ID of the best unit.")
            print(" - Close the GUI window and return here to enter the ID.\n")
            app = sig.run_mainwindow(analyzer)
        except ImportError:
            print("\n[WARN] spikeinterface-gui not installed!")
            print("Please run 'pip install spikeinterface-gui' to use the interactive picker natively.")
            
        print("\n---------------------------------------------------------")
        user_input = input(f"Enter the best Unit ID for {port_name} (leave empty to skip): ")
        if user_input.strip() != "":
            try:
                if unit_ids.dtype.kind in 'iu':
                    best_id = int(user_input.strip())
                else:
                    best_id = str(user_input.strip())
                
                if best_id not in unit_ids:
                    print(f"Unit {best_id} not found: {unit_ids}")
                    continue
                
                # Fetch templates and isolate 1D waveform
                ext_templates = analyzer.get_extension("templates").get_data() # (n_units, n_samples, n_channels)
                unit_idx = list(unit_ids).index(best_id)
                template_2d = ext_templates[unit_idx]
                
                # Extract extremum channel dictionary
                ext_dict = si.get_template_extremum_channel(analyzer, peak_sign='neg')
                ext_ch = ext_dict[best_id]
                ext_ch_idx = rec_port.id_to_index(ext_ch)
                
                template_1d = template_2d[:, ext_ch_idx]
                
                # Normalize (neg peak equals -1.0)
                norm_factor = np.abs(np.min(template_1d))
                template_1d_norm = template_1d / norm_factor
                
                out_path = conf_dir / f"median_extremum_templates_norm_UA_{port_name}.npy"
                np.save(out_path, template_1d_norm)
                print(f"-> SUCCESS! Saved normalized template to {out_path}")
                print(f"   Original magnitude (abs extremum): {norm_factor:.2f}\n")
                
            except Exception as e:
                print(f"Error extracting and saving template: {e}")

if __name__ == '__main__':
    main()
