#!/bin/bash
cd src
#Seed 0-9 is used in our paper
#Script to run the model for cora
python main.py --seed 0 --data cora
#Script to run the model for citeseer
python main.py --seed 0 --data citeseer
#Script to run the model for cora-ml
python main.py --seed 0 --data cora-ml
#Script to run the model for wikics
python main.py --seed 0 --data wikics --epoch 2000 --patience 2000
#Script to run the model for mmorpg
python main.py --seed 0 --data mmorpg

