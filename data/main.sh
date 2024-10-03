#!/usr/bin/bash

dir=$(cd $(dirname $0); pwd)
wget -nc https://archive.ics.uci.edu/static/public/359/news+aggregator.zip -P $dir
unzip -d $dir ${dir}/news+aggregator.zip
python3 ${dir}/preprocess.py
