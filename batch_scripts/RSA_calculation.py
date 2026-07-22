import RCP_analysis as rcp

# Config
def main():
    # Load in params (poststim window, move_alpha, stim_alpha)
    poststim_win_ms=(0.0, 50.0)
    channel_criterion=["movement", "stim", "union", "intersection"]
    cond_label_extras={"at_rest": "At rest", "Baseline": "Baseline"}
    move_alpha=0.05
    stim_alpha=0.05
    targets = ["target_A", "target_B"]
    probes = ['NPRW', 'UA']
    
    # Loop over NPRW and UA, Target A and Target B
    for probe in probes:
        for target in targets:
            rcp.run_rsa(
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
            )     
            
if __name__ == "__main__":
    main()