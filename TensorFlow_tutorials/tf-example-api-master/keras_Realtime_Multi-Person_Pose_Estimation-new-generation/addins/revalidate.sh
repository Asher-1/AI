#!/bin/bash
for i in $(seq $1 $2 $3)
do
   python ./train_heads.py save_network_input_output HeadCount '' $i
done