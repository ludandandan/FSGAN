exp_id=0
ep=84
python test.py --checkpoints_dir checkpoints --name ckpt_${exp_id} \
--model test --dataset_mode aligned --norm batch --use_local \
--which_epoch ${ep} --results_dir dataset/celeba_hq_256/results \
--dataroot dataset/celeba_hq_256/photo --data_json dataset/celeba_hq_256/celeba.json \
--bg_dir dataset/celeba_hq_256/mask --gpu_ids 1

