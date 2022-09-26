## Installation

Only tested with python3.7 so far

For installation with Dali, run :

```
pip3 install .[dali]
```

For normal installation, run :

```
pip3 install .
```

### Imagenet

Download imagenet dataset train (task 1 and 2) and val from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and put them in the `datasets` folder

Run the commands in this [file](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) from the `datasets` folder to setup the classes.

### Imagenet100

You need to already have the imagenet dataset in `datasets` folder in the required format.

In the `datasets` folder, run the following commands : 

```
git clone https://github.com/danielchyeh/ImageNet-100-Pytorch.git 
cd ImageNet-100-Pytorch/
python3 generate_IN100.py --source_folder ../imagenet/train --target_folder ../imagenet100/train
python3 generate_IN100.py --source_folder ../imagenet/val --target_folder ../imagenet100/val
```

## Running scripts

```
bash scheduler.sh
```