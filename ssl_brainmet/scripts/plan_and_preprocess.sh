export nnUNet_raw="/home/valentin/data/target/data/nnUnet_raw"
export nnUNet_preprocessed="/home/valentin/data/target/data/nnUNet_preprocessed"
export nnUNet_results="/home/valentin/data/target/data/nnUNet_results"

TASK=510

nnUNetv2_plan_and_preprocess -d $TASK --verify_dataset_integrity