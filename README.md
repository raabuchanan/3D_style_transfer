# 3D_style_transfer
Final Project for 3D Vision at ETH Zurich


## Learned Descriptors

Get the images from this link https://polybox.ethz.ch/index.php/s/82WvLFNBR4ACjir
place the entire glove folder inside the data folder

To build the C++ code run the build script

sudo chmod +x build.sh
./build.sh

Then run each step in the folloing order

./extractor.o
th descriptor.lua
./matcher.o

To view maches run
./verifier.o