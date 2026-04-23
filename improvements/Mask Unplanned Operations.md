Here, I lay out plan to handle Unplanned Operations, as well as some addition to previous Unknown operation.

1. Certain operations are unplanned and are sidetrack from normal operations. We need to merge them under "Unplanned" name.
2. We also need to mask all such observations loss in training so that they do not have an effect on the training. We should mask for major_ops_code, operation, duration hours for all such Unplanned operations. Below is list of Unplanned operations. dump it in the pipeline.yaml so that it is configurable (in case we want to add or remove from the list)
3. We should also mask duration hours in Unknown operations as well


- Unplanned Operations:

        "RIG_RPR_NPT", "3RD_PTY", "WAIT", "WOW", "H2S_FALSE", "LEL_FALSE",
        "WELL_CTRL", "K&M_CLEANUP", "DRL_K&M", "LCM_SQUEEZE", "PACKOFF", "JAR",
        "FISH", "PULL_CSG", "PULL_LNR",
        "CMT_REM",
        "SFTY_NPT"