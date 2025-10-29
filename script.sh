config_path='configs/redet/redet_re50_refpn_1x_dior_r_le90.py'
sample="udc"
c_0_path="work_dirs/PU/EXP1/cycle0/epoch_12.pth"
s_0_path="work_dirs/PU/EXP1/cycle0/example_Task1"
c_1_path="work_dirs/PU/EXP1/cycle1/epoch_12.pth"
s_1_path="work_dirs/PU/EXP1/cycle1/example_Task1"
c_2_path="work_dirs/PU/EXP1/cycle2/epoch_12.pth"
s_2_path="work_dirs/PU/EXP1/cycle2/example_Task1"
c_3_path="work_dirs/PU/EXP1/cycle3/epoch_12.pth"
s_3_path="work_dirs/PU/EXP1/cycle3/example_Task1"
c_4_path="work_dirs/PU/EXP1/cycle4/epoch_12.pth"
s_4_path="work_dirs/PU/EXP1/cycle4/example_Task1"
work_dir="work_dirs/PU"

python train_dior.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 0 &&
python test_dior.py --config $config_path --checkpoint $c_0_path --eval mAP &&
python train_dior.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 1 &&
python test_dior.py --config $config_path --checkpoint $c_1_path --eval mAP &&
python train_dior.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 2 &&
python test_dior.py --config $config_path --checkpoint $c_2_path --eval mAP &&
python train_dior.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 3 &&
python test_dior.py --config $config_path --checkpoint $c_3_path --eval mAP &&
python train_dior.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 4 &&
python test_dior.py --config $config_path --checkpoint $c_4_path --eval mAP
