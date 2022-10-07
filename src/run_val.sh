#!/bin/bash
# NPA
# individual
annotators=( "tampo" "takahashi" "yanagi" )
for indv_name in "${annotators[@]}"
do
    python3 demo.py --resume ../logs/2021_1025_2326/model-best.xml --save_pics --img_types npa --annotators $indv_name
done

# staple
python3 demo.py --resume ../logs/2021_1025_2326/model-best.xml --save_pics --img_types npa --annotators ${annotators[@]}
