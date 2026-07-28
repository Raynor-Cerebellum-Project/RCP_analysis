from pathlib import Path

# Import internal modules relatively to avoid circular dependency with the package export
from .params_loading import load_experiment_params

REPO_ROOT = Path(__file__).resolve().parents[3]

# Load params
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"
if not PARAMS_PATH.exists():
    # Fallback or error
    print(f"[config_loading] Warning: {PARAMS_PATH} does not exist. Please check your repo structure.")
    PARAMS = None
else:
    try:
        PARAMS = load_experiment_params(PARAMS_PATH, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[config_loading] Error loading params: {e}")
        PARAMS = None

# Paths
if PARAMS is not None:
    try:
        MONKEY         = PARAMS.monkey

        # Directories
        SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
        OUT_BASE       = SESSION_LOC / "results"
        BR_ROOT        = SESSION_LOC / "Blackrock"
        VIDEO_ROOT     = SESSION_LOC / "Video"
        VOG_ROOT       = SESSION_LOC / "VOG"
        INTAN_ROOT     = SESSION_LOC / "Intan"
        METADATA_ROOT  = SESSION_LOC / "Metadata" 
        
        # Sub-directories (Outputs)
        BEHV_AUX_DATA  = OUT_BASE / "aux_data" / "Behavior"
        NPRW_AUX_DATA  = OUT_BASE / "aux_data" / "NPRW"
        UA_AUX_DATA    = OUT_BASE / "aux_data" / "UA"
        
        BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "Behavior"
        NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"
        UA_CKPT_ROOT   = OUT_BASE / "checkpoints" / "UA"
        ALIGNED_CKPT_ROOT = OUT_BASE / "checkpoints" / "Aligned"
        VOG_CKPT_ROOT  = OUT_BASE / "checkpoints" / "VOG"
        PERI_ROOT      = OUT_BASE / "checkpoints" / "PeriStim"
        
        # Ensure common output directories exist
        OUT_BASE.mkdir(parents=True, exist_ok=True)
        BR_ROOT.mkdir(parents=True, exist_ok=True)
        VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
        VOG_ROOT.mkdir(parents=True, exist_ok=True)
        INTAN_ROOT.mkdir(parents=True, exist_ok=True)
        METADATA_ROOT.mkdir(parents=True, exist_ok=True)
        
        BEHV_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        NPRW_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        UA_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        ALIGNED_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        VOG_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        
        PERI_ROOT.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        METADATA_CSV   = METADATA_ROOT / f"{PARAMS.session}_metadata.csv"

        # Probe config
        UA_CFG = PARAMS.probes.get("UA", {})
        CAMERA_SYNC_CH = int(UA_CFG.get("camera_sync_ch", 134))

        # Kinematics
        if PARAMS.kinematics:
            KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints", []))
        else:
            KEYPOINTS_ORDER = ()

    except Exception as e:
        print(f"[config_loading] Error deriving paths from PARAMS: {e}")
        # Initialize variables to None
        SESSION_LOC = None
        OUT_BASE = None
        MONKEY = None
else:
    SESSION_LOC = None
    OUT_BASE = None
    MONKEY = None

