import os
import shutil
path = r'E:\ME_dataset'
dest_path = r'D:\Bbox_crop_anno'
id_list = os.listdir(path)
for id in id_list:
    if not os.path.isfile(os.path.join(dest_path, f'{id}_nodule_count_crop.json')):
        try:
            shutil.copy(os.path.join(path, id, 'mask', f'{id}_nodule_count_crop.json'), dest_path)
        except:
            print(id, 'error')