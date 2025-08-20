conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export PYTHONPATH=${workspaceFolder}:${env:PYTHONPATH}
export ABS_VIS_START_ID=151666
export ABS_VIS_END_ID=151667
python -m src.quick_infer \
                --model_path "/home/dids/shiyang/checkpoints/08_15-avt_stage1-6-30-40-wt1.0-ep2" \
                --data_path "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json" \
                --num_samples "50"