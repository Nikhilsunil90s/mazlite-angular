#!/bin/bash
export LC_ALL=C.UTF-8;
export LANG=C.UTF-8;
export dataRoot='./data/local/all_user_data';
export DB_USER=root;
export DB_PASSWORD=234i343k3;
export DB_HOST=mazlite-db;
export DB_NAME=mazlite;
#pushd .
uvicorn main:app --reload --host 0.0.0.0 --port 5000

