from pathlib import Path

from RCP_analysis.python.functions.config_loading import PARAMS, OUT_BASE, PERI_ROOT
from RCP_analysis.python.functions.rsa_utils import run_rsa

# Config
def main():
    rsa_cfg = PARAMS.rsa_params

    poststim_win_ms = tuple(rsa_cfg.get("poststim_win_ms"))
    channel_criterion = rsa_cfg.get("channel_criterion")
    cond_label_extras = {"at_rest": "At rest", "Baseline": "Baseline"}
    move_alpha = rsa_cfg.get("move_alpha")
    stim_alpha = rsa_cfg.get("stim_alpha")
    targets = rsa_cfg.get("targets")
    probes = rsa_cfg.get("probes")
    
    # Loop over NPRW and UA, Target A and Target B
    for probe in probes:
        for target in targets:
            run_rsa(
                source=probe,
                target=target,
                poststim_win_ms=poststim_win_ms,
                channel_criterion=channel_criterion,
                cond_label_extras=cond_label_extras,
                vmin=-1.0,
                vmax=1.0,
                skip_conds=None,
                cond_order=None,
                move_alpha=move_alpha,
                stim_alpha=stim_alpha,
                debug_masks=True,
                peri_root=Path(PERI_ROOT),
                out_base=Path(OUT_BASE),
                fig_dir=Path(OUT_BASE) / "figures" / "rsa_from_peristim",
            )   
            
if __name__ == "__main__":
    main()