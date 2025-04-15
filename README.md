# About This Repository

This is a fork of [MaxText](https://github.com/google/maxtext) modified to support sparse pre-training.

For more information about our approach, please refer to our paper: [The Journey Matters: Average Parameter Count over Pre-training Unifies Sparse and Dense Scaling Laws](https://arxiv.org/abs/2501.12486)

# Example Usage

Below is an example script for running a Llama2-1B model training job:

```bash
PROJECT=your-project-name
ZONE=your-compute-zone
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE

BUCKET_NAME=gs://your-storage-bucket
NODE_COUNT=1
TPU_TYPE=v4-128
RUN_NAME=llama2-1b-v4-128-run
python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --TPU_TYPE=$TPU_TYPE --BUCKET_DIR=job-log-dir --CQR_EXTRA_ARGS="--best-effort" --COMMAND="bash setup.sh && python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME model_name=llama2-1b base_output_directory=$BUCKET_NAME/runs dataset_type=tfds dataset_path=$BUCKET_NAME/c4 enable_checkpointing=true"
```

Make sure to replace the placeholder values with your actual project information before running the script.


```bibtex
@inproceedings{
jin2025the,
title={The Journey Matters: Average Parameter Count over Pre-training Unifies Sparse and Dense Scaling Laws},
author={Tian Jin and Ahmed Imtiaz Humayun and Utku Evci and Suvinay Subramanian and Amir Yazdanbakhsh and Dan Alistarh and Gintare Karolina Dziugaite},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=ud8FtE1N4N}
}
```

For the full documentation and features of the original MaxText project, please visit the [original MaxText repository](https://github.com/google/maxtext).
