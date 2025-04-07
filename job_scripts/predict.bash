#!/bin/bash

#Copyright (c) 20xx Idiap Research Institute, http://www.idiap.ch/
#Written by Alireza Mohammadshahi <alireza.mohammadshahi@idiap.ch>,

#This file is part of g2g-transformer.

#g2g-transformer is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License version 3 as
#published by the Free Software Foundation.

#g2g-transformer is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with g2g-transformer. If not, see <http://www.gnu.org/licenses/>.

# initial prediction (leave it blank if there is no initial parser)
initial=""
# original conllu input
file="test"
input="/home/preetam_pg/g2g-transformer/data/sishu/poetry.conllu"
modelpath="/home/preetam_pg/g2g-transformer/models/STBC"
output_path="/home/preetam_pg/g2g-transformer/pred/STBC"
# UD: conllu, penn: conllx, german: other
type="conllu"
language="en"
# keep punctuation or not
punct="yes"
if [ ! -d $output_path ]; then
  mkdir -p $output_path;
fi


echo "Predicting the input file"
if [ "$initial" = "" ]; then
  python run.py predict --modelpath $modelpath --fdata $input --fpred $output_path/$file.$type --input_type $type
else
  python run.py predict --fdata $input --finit $initial --modelpath $modelpath --use_predicted --input_type $type \
            --fpred $output_path/$file.$type
fi
echo "Finished Prediction"


if [ "$type" = "conllu" ]; then
    echo "Evaluating based on official UD script"
    python conll18_ud_eval.py $input $output_path/$file.$type -v
elif [ "$type" = "conllx" ]; then
    if [ "$punct" = "no" ]; then
        perl eval.pl -g $input -s $output_path/$file.$type -q
    else
        perl eval.pl -g $input -s $output_path/$file.$type -q -p
    fi
    echo "done"
fi