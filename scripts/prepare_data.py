import os
import subprocess
from pathlib import Path
import argparse
import sys


# METADATA = {
#     "im2latex_formulas.norm.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_formulas.norm.lst",
#     "im2latex_validate_filter.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_validate_filter.lst",
#     "im2latex_train_filter.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_train_filter.lst",
#     "im2latex_test_filter.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_test_filter.lst",
#     # "formula_images.tar.gz": "http://lstm.seas.harvard.edu/latex/data/formula_images.tar.gz",
# }
PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
DATA_DIRNAME = PROJECT_DIRNAME / "data"
RAW_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images"
PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images_processed"
VOCAB_FILE = PROJECT_DIRNAME / "image_to_latex" / "data" / "vocab.json"


def main():
    DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    cur_dir = os.getcwd()
    os.chdir(DATA_DIRNAME)

    # Download images and grouth truth files
    # for filename, url in METADATA.items():
    #     if not Path(filename).is_file():
    #         utils.download_url(url, filename)

    # Unzip
    # if not RAW_IMAGES_DIRNAME.exists():
    #     RAW_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
    #     utils.extract_tar_file("formula_images.tar.gz")

    # Extract regions of interest
    if not PROCESSED_IMAGES_DIRNAME.exists():
        PROCESSED_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        print("Cropping images...")
        for image_filename in RAW_IMAGES_DIRNAME.glob("*.png"):
            cropped_image = utils.crop(image_filename, padding=8)
            if not cropped_image:
                continue
            cropped_image.save(PROCESSED_IMAGES_DIRNAME / image_filename.name)

    # Clean the ground truth file
    # cleaned_file = "im2latex_formulas.norm.new.lst"
    # if not Path(cleaned_file).is_file():
    #     print("Cleaning data...")
    #     script = Path(__file__).resolve().parent / "find_and_replace.sh"
    #     subprocess.call(["sh", f"{str(script)}", "im2latex_formulas.norm.lst", cleaned_file])

    # Build vocabulary
    if not VOCAB_FILE.is_file():
        print("Building vocabulary...")
        all_formulas = utils.get_all_formulas("im2latex_formulas.norm.new.lst")
        _, train_formulas = utils.get_split(all_formulas, "im2latex_train_filter.lst")
        tokenizer = utils.Tokenizer()
        tokenizer.train(train_formulas)
        tokenizer.save(VOCAB_FILE)
    os.chdir(cur_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_path', type=str,default='/kaggle/working/image-to-latex',help='path of the project')
    args = parser.parse_args()
    sys.path.append(args.project_path)
    import image_to_latex.data.utils as utils
    main()
