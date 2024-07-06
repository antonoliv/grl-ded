#!/usr/bin/env bash

git clone https://github.com/antonoliv/master-dissertation
cd master-dissertation/src/grl
pip install -r requirements.txt --no-cache-dir
python install_scenarios.py
