import os
import random

hq_data_path = "/raid/ai22mtech02003/EESRGAN/DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/"
lq_data_path = "/raid/ai22mtech02003/EESRGAN/DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/"

hq_files_list = os.listdir(hq_data_path)
lq_files_list = os.listdir(lq_data_path)

list_jpg_hq = []
list_jpg_lq = []

for i in range(len(hq_files_list)):
    if "jpg" in str(hq_files_list[i]):
        list_jpg_hq.append(hq_files_list[i])

for i in range(len(lq_files_list)):
    if "jpg" in str(lq_files_list[i]):
        list_jpg_lq.append(lq_files_list[i])


random.shuffle(list_jpg_lq)
random.shuffle(list_jpg_hq)

tot = len(list_jpg_lq)

num = int(tot*0.2)

list_jpg_lq = list_jpg_lq[0:num]
list_jpg_hq = list_jpg_hq[0:num]


val_lr_path = "/raid/ai22mtech02003/EESRGAN/DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/test_dir/LR/"
val_hr_path = "/raid/ai22mtech02003/EESRGAN/DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/test_dir/HR/"
os.makedirs(val_lr_path,exist_ok=True)
os.makedirs(val_hr_path,exist_ok=True)
for i in range(len(list_jpg_hq)):
    src_jpg = os.path.join(hq_data_path,list_jpg_hq[i])
    dest_jpg = os.path.join(val_hr_path,list_jpg_hq[i])
    src_txt = str(list_jpg_hq[i]).split(".")
    src_txt = src_txt[0:3]
    src_txt_file = '.'.join(src_txt)+'.txt'
    src_txt = os.path.join(hq_data_path,src_txt_file)
    dst_txt = os.path.join(val_hr_path,src_txt_file)
    os.rename(src_jpg,dest_jpg)
    os.rename(src_txt,dst_txt)

for i in range(len(list_jpg_lq)):
    src_jpg = os.path.join(lq_data_path,list_jpg_lq[i])
    dest_jpg = os.path.join(val_lr_path,list_jpg_lq[i])
    src_txt = str(list_jpg_lq[i]).split(".")
    src_txt = src_txt[0:3]
    src_txt_file = '.'.join(src_txt)+'.txt'
    src_txt = os.path.join(lq_data_path,src_txt_file)
    dst_txt = os.path.join(val_lr_path,src_txt_file)
    os.rename(src_jpg,dest_jpg)
    os.rename(src_txt,dst_txt)