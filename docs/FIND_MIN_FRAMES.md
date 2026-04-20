# Default: 200 FPS frames, batch mode, default thresholds
SCENE=room0 N_FRAMES=200 bash shells/run_fps_experiment.sh
# → ablations/fps_pose_200/

# 50 FPS frames with relaxed thresholds
SCENE=room0 N_FRAMES=50 SIM_THRESH=1.0 PHYS_BIAS=0.3 bash shells/run_fps_experiment.sh
# → ablations/fps_pose_50_st1.0_pb0.3/

# 100 FPS frames (custom count) with aggressive merge
SCENE=room0 N_FRAMES=100 MERGE_OVERLAP_THRESH=0.5 MERGE_VISUAL_SIM_THRESH=0.5 bash shells/run_fps_experiment.sh
# → ablations/fps_pose_100_mot0.5_mvs0.5/

# Force incremental mode for comparison
SCENE=room0 N_FRAMES=200 MATCHING_MODE=incremental bash shells/run_fps_experiment.sh
# → ablations/fps_pose_200_incremental/

Finishing the Stages
`STAGES="s4_captions" bash generate_groundtruth/run_all.sh`

`STAGES="s5_vocab s6_vetting s7_vqa" bash generate_groundtruth/run_all.sh`