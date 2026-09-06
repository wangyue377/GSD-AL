# GSD-AL

![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg?style=plastic)
![PyTorch 1.8](https://img.shields.io/badge/PyTorch-1.8-EE4C2C.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/CUDA-11.1-green.svg?style=plastic)
![cuDNN 7.6.5](https://img.shields.io/badge/cudnn-7.6.5-green.svg?style=plastic)
[![LICENSE](https://img.shields.io/github/license/yuantn/mi-aod.svg)](https://github.com/yuantn/mi-aod/blob/master/LICENSE)

<!-- TOC -->

- [Abstract](#abstract)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Train and Test](#Train-and-Test)

<!-- TOC -->


PyTorch implementation of [
***Geometric-Semantic Diversity-based Active Learning for Oriented Object Detection in Remote Sensing***]().

The code will be updated later.

## Abstract

Oriented object detection in aerial remote sensing imagery faces challenges of costly annotation due to dense object
distributions and complex geometric variations. Active learning (AL) offers an effective approach to mitigate annotation
costs by selectively labeling the most informative samples. However, traditional AL methods predominantly focus on
uncertainty or representativeness criteria independently, lacking sufficient consideration for geometric and semantic
diversity, particularly crucial for rotated objects in remote sensing scenarios. To address these issues, we propose a
novel Geometric-Semantic Diversity-based Active Learning (GSD-AL) framework for oriented object detection. Specifically,
our method integrates instance-level uncertainty with both geometric and semantic diversity constraints. The geometric
diversity module exploits the spatial variability of rotated bounding boxes, emphasizing diversity in aspect ratios and
rotation angles to avoid redundancy. Concurrently, the semantic diversity module utilizes feature-based cosine
similarity, efficiently filtering instances to ensure semantic representativeness. Furthermore, we introduce a novel
dynamic class-balancing strategy that prioritizes uncertain instances from underrepresented classes by imposing a
maximum sampling limit for each class. Extensive experiments conducted on the DOTA benchmark demonstrate that our
proposed GSD-AL framework significantly reduces annotation costs while achieving comparable or superior detection
performance compared to state-of-the-art AL methods.

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n GSD-AL python=3.7 -y
    conda activate GSD-AL
    ```

2. Please install **PyTorch 1.8.0** and **torchvision 0.9.0** for **CUDA 11.1** following
   the [official instructions](https://pytorch.org/get-started/previous-versions/#v160), e.g.,

    ```shell
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```
3. Please install mmcv-full, we recommend you to install the pre-build package as below.
    ```shell
    pip install mmcv-full==1.4.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
    ```

4. Please clone the GSD-AL repository.

    ```shell
    git clone https://github.com/wangyue377/GSD-AL.git
    cd GSD-AL
    ```

5. Please download the required package.

    ```shell
    pip install -r requirements/build.txt
    ```

## Data Preparation

Please refer to [data_preparation.md](data_preparation.md) for dataset installation and segmentation.


<!-- You may also use the following commands directly:

```bash
cd $YOUR_DATASET_PATH
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar
``` -->

## Train and Test

Please modify the corresponding pretrained model path, they are located in:

```python
Line
16
of
configs / redet / redet_re50_refpn_1x_dota_le90.py: pretrained = '$PRETRAINED_MODEL_PATH/'
Line
16
of
configs / redet / redet_re50_refpn_1x_dota2_le90.py: pretrained = '$PRETRAINED_MODEL_PATH/'
```

We recommend you to use a GPU but not a CPU to train and test, because it will greatly shorten the time.

If you use only a single GPU, you can use the `script.sh` file directly as below:

```bash
chmod 700 ./script.sh
./script.sh 
```

The script uses the train.py and test.py functions to train and test where the value
of `config_path`,  `sample`, `c_0_path`, `s_0_path` and `work_dir` shound be changed according to your actual
situation (e.g., `config_path` should be replaced by the path of the config file in the `configs/_base_` folder).

Please note more parameters can be found and mofidy in the file of train.py and test.py .

We drew on the codebase of the MUS-CDB method.
