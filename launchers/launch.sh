#!/bin/bash

file_name=`python -c "import datetime; print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))"`
sbatch --output=./out/$1/$file_name.out --error=./out/$1/$file_name.err ./launchers/$1.sh
