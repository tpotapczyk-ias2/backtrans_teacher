#!/bin/bash
cd data/checkpoints
step=400000
onmt_average_models -models  checkpoint_step_$(($step-18000)).pt checkpoint_step_$(($step-16000)).pt checkpoint_step_$(($step-14000)).pt checkpoint_step_$(($step-12000)).pt checkpoint_step_$(($step-10000)).pt checkpoint_step_$(($step-8000)).pt checkpoint_step_$(($step-6000)).pt checkpoint_step_$(($step-4000)).pt checkpoint_step_$(($step-2000)).pt checkpoint_step_$(($step)).pt -o average$step

cd ../..
echo $(date)
onmt_translate  -model data/checkpoints/average$step -src data/src_valid_tokenized.txt -output pred_$step'_avg.txt' -gpu 0
echo $(date)
