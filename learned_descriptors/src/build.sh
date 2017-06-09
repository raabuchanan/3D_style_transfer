#!/usr/bin/env bash

g++ extractor.cpp -o extractor.o `pkg-config --cflags --libs opencv` -lsqlite3

nvcc matcher.cu -o matcher.o -I /usr/include/eigen3 `pkg-config --cflags --libs opencv` -lsqlite3

g++ verifier.cpp -o verifier.o `pkg-config --cflags --libs opencv` -lsqlite3
