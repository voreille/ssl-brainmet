from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)

target_data_path = Path(
    "/home/vincent/repos/brain-mets/data/nnUNet_raw_data_base/nnUNet_raw_data/Task510_BrainMetsSegT1to1nodnec"
)
brats_data_path = Path("/mnt/nas6/data/BRATS-MET/raw/Task510_BrainMetsSegT1to1nodnec")

nnunet_raw_dir = (
    "/home/valentin/data/target/data/nnUnet_raw/Dataset510_BrainMetsSegT1to1nodnec"
)

subdirs = ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]
dry_run = False


def get_number_id(x):
    if not isinstance(x, Path):
        x = Path(x)
    output = x.name.replace(".nii.gz", "")
    return int(output.split("_")[1])


def copy_file(file_to_copy, output_dir, offset=None, dry_run=False):
    filename = file_to_copy.name.replace(".nii.gz", "")
    filename_parts = filename.split("_")
    number_id = int(filename_parts[1])

    if offset:
        number_id += offset

    filename_parts[1] = f"{number_id:04d}"
    new_filename = "_".join(filename_parts) + ".nii.gz"
    new_path = output_dir / new_filename
    logging.info(f"Copying {file_to_copy} to {new_path} ")

    if dry_run:
        return

    shutil.copy(file_to_copy, new_path)


def main():
    output_dir = Path(nnunet_raw_dir)

    files = list((target_data_path).rglob("*.nii.gz"))
    files.sort(key=lambda x: get_number_id(x))
    biggest_target_id = get_number_id(files[-1])

    files = list((brats_data_path).rglob("*.nii.gz"))
    files.sort(key=lambda x: get_number_id(x))
    lowest_brats_id = get_number_id(files[0])

    offset_brats = biggest_target_id - lowest_brats_id + 1

    for subdir_name in subdirs:
        subdir = output_dir / subdir_name
        subdir.mkdir(exist_ok=True, parents=True)

        logging.info(f"Processing TARGET data: {target_data_path}")
        for file in (target_data_path / subdir_name).glob("*.nii.gz"):
            copy_file(file, subdir, dry_run=dry_run)

        logging.info(f"Processing BRATS data: {brats_data_path}")
        for file in (brats_data_path / subdir_name).glob("*.nii.gz"):
            copy_file(file, subdir, offset=offset_brats, dry_run=dry_run)
        logging.info("DONE.")


if __name__ == "__main__":
    main()
