#!/bin/sh

VINO_CONVERTER="/opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/mo.py"
EXP_DIR=$(dirname "$1")
MODEL=$(basename "$1")
echo $EXP_DIR
# INPUT_MODEL="$2"

cd $EXP_DIR
python3 $VINO_CONVERTER --input_model $MODEL