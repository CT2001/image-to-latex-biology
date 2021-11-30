import os
import shutil

label_dir = '../data/formula_labels/'
image_dir = '/kaggle/input/formula-image/math_formula_images_grey/math_formula_images_grey/'
output_dir = '../data/formula_images/'

label_name_list = os.listdir(label_dir)

for i in range(len(label_name_list)):
    label_name_list[i] = label_name_list[i][:-4]

# print(label_list)

image_name_list = os.listdir(image_dir)

for image_name in image_name_list:
    if image_name[:-4] in label_name_list:
        print(image_name)
        shutil.copy(image_dir + image_name, output_dir + image_name)
