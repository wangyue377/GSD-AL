# Rotation Detection Demo

We provide a demo script to test a single image.

```shell
python demo/image_demo.py \
    ${IMG_ROOT} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE}
```

Examples:

```shell
python demo/image_demo.py demo/demo.jpg work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/oriented_rcnn_r50_fpn_1x_dota_v3.py work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/epoch_12.pth
```

```shell
python demo/image_demo.py demo/demo.jpg configs/redet/redet_re50_refpn_1x_dota_le90.py work_dirs/PU/EXP12/cycle4/epoch_12.pth
```

python demo/batch_demo.py demo/image/ configs/redet/redet_re50_refpn_1x_dota_le90.py
work_dirs/PU/EXP6/cycle4/epoch_12.pth --out_dir demo/output/ --device cuda:0 --score-thr 0.3

python demo/batch_demo.py demo/image/ configs/redet/redet_re50_refpn_1x_dota_le90.py E:
/MUS-CDB/DOTA_v1.0/cycle4/epoch_12.pth --out_dir demo/output/ --device cuda:0 --score-thr 0.3

python combine.py demo/111 output.jpg

E:/MUS-CDB-master/work_dirs/PU/EXP6/cycle0/example_Task1

python demo/batch_demo.py C:/Users/Administrator/Desktop/0706/ configs/redet/redet_re50_refpn_1x_dota_le90.py
demo/pt_file/UGF/epoch_12.pth --out_dir demo/out/ --device cuda:0 --score-thr 0.3