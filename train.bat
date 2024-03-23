@REM python my_train_CPMv2.py --train_set ./data/client0_labeled_train.txt --val_set ./data/client0_val.txt --batch-size 3 --num_workers 3 --se --num_samples 5 --mixed_precision --start_val_epoch 60 --pretrained_model_path ./save/pretrained_model.pth --exp_name client0_labeled_mp_bs3_ns5 --epochs 250
@REM python my_train_CPMv2.py --train_set ./data/client1_labeled_train.txt --val_set ./data/client1_val.txt --batch-size 3 --num_workers 3 --se --num_samples 5 --mixed_precision --start_val_epoch 60 --pretrained_model_path ./save/pretrained_model.pth --exp_name client1_labeled_mp_bs3_ns5 --epochs 250
@REM python my_train_CPMv2.py --train_set ./data/client2_labeled_train.txt --val_set ./data/client2_val.txt --batch-size 3 --num_workers 3 --se --num_samples 5 --mixed_precision --start_val_epoch 100 --pretrained_model_path ./save/pretrained_model.pth --exp_name client2_labeled_mp_bs3_ns5 --epochs 250
@REM python my_train_CPMv2.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --batch-size 3 --num_workers 3 --se --num_samples 5 --mixed_precision --exp_name pretrained_mp_bs4_ns4 --epochs 250 --resume_folder ./save/[2024-02-11-2214]_pretrained_mp_bs3_ns5
@REM python my_train_CPMv2.py --train_set ./data/all_client_train.txt --val_set ./data/all_client_val.txt --batch-size 3 --num_workers 3 --se --num_samples 5 --mixed_precision --pretrained_model_path ./save/pretrained_model.pth --exp_name from_pretrained_all_client_mp_bs3_ns5 --epochs 150 --start_val_epoch 60
@REM python my_train_CPMv2.py --train_set ./data/all_train.txt --val_set ./data/all_val.txt --batch-size 3 --num_workers 3 --se --num_samples 5 --mixed_precision --exp_name all_ME_bs3_ns5 --epochs 250 --save_model_interval 1
python train.py --train_set E:/Jack/Me_dataset_dicom_resize_npy/train/data_list1.txt --val_set E:/Jack/Me_dataset_dicom_resize_npy/val/data_list1.txt --test_set E:/Jack/Me_dataset_dicom_resize_npy/val/data_list1.txt --batch 1 --num_samples 2 --aspp --mixed_precision --network_name ResNet_3D_CPM_stride2