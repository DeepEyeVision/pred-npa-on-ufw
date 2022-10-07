# Prediction of NPA on UFW Images

This repository contains a research result of PraNet-based model that does semantic segmentation of NPA on UFW images.

## ENVIRONMENT SETUP
### REQUIREMENTS
```
python -V
3.7.9
```
```
pip install -r requirements.txt
```
Docker environment is required for running and compling quantized models.
```
docker build -t vino-q .
```

### DATASET
Sicne the dataset is highly privacy sensitive, we do not make the dataset, including any sample images, public.

## TRAIN
Here is an example of training command.
```
python train_optuna.py --gpus 0,1,2,3 -e 40 -t 20 --resume ../path/to/modle-to-resume.pth --nncf_config config/qint8_segm.json
```

| --option | description |
| --       | --  |
| --gpus | Gpu indices to use. Comma separated and without blank spaces. |
| -e    | Number of epochs. |
| -t | Number of trials of hyper parameters search. |
| --resume | Path to pretrained model. |
| --nncf_config | Path to NNCF quantization config file. |


## COMPILE and TEST
Compling model requires docker environment.
Start the docker container with the below command.
```
sh run.sh
```
Convert an onnx model to intermediate representation (IR) of OpenVINO.
```
sh convert sh <model-to-quantize.onnx>
```

The above commands produces ```xml``` and ```bin``` files.
To run the compiled model, 
```
python demo.py --resume path/to/model-best.xml 
```

## INFERENCE SPEED
| type    | inference (/image) |
| --      | --                 |
| PyTorch | 48 sec             |
| OpenVINO|  7 sec            | 

## ACKNOWLEDGEMENTS
We acknowledge the authors of [PraNet](https://github.com/DengPingFan/PraNet) to make the code public.

## LICENSE
We inherit the [PraNet License](https://github.com/DengPingFan/PraNet#7-license).
So, commercial use should get PraNet's and our permission.
