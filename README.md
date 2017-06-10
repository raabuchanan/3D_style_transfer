# 3D_style_transfer
Final Project for 3D Vision at ETH Zurich


## Learned Descriptors

0. **Prerequisites:**
    * Ubuntu 16.04
    * OpenCV 3.1.0
    * CUDA v7.5
    * Lua 5.1 with the following Luarocks:
        * ``nn``
        * ``cunn``
        * ``image``
        * ``torch``
        * ``lsqlite3``

1. **Preparing data:** To get started quickly use the images proveded [here](https://polybox.ethz.ch/index.php/s/82WvLFNBR4ACjir). Place the entire directory named ``glove/mosaic`` inside the data folder. You can download the other directories aswell if you want to try out different styles. If you want to use your own data you will need to first run ``rename.sh`` to change the names of the files to sequential numbers.
2. **Building:** To build the C++ code run:
``cd ~/3D_style_transfer/learned_descriptors/src``
`sudo chmod +x build.sh`
`./build.sh`
This will build all of the C++ and Cuda code.
3. **Extracting features**
Run: `./extractor.o` which will extract up to 3000 features and record them in the ``glove_mosaic_learned.db`` database. The number of features can be changed but this will take longer to match later on. Currently the code is only set to find features in the first 5 images. If you actually want to attempt 3D reconstruciton you will need to extract feature from all the images and have at least 5000 features.
4. **Converting descriptors**
Here we use Lua to convert the image patches into 128D descriptors. Run: `th descriptor.lua`.
5. **Matching features**
Here exhaustive matching is performed using the GPU. Run `./matcher.o`
6. **Viewing matches**
To view matches between the first two images Run: `./verifier.o`

## Voxel Carving

0. **Prerequisites:**
    * Ubuntu 14.04
    * OpenCV 3.1.0
    * VTK

1. **Preparing data:** In order to use this module the user needs to have camera.txt, images.txt both without the HEADER (commented description). Additionaly the user needs to provide "sil" folder where the silhouettes are stored. Sample assets are provided in the "sample" folder.

2. **Building:** To build the C++ code run:
``cd ~/3D_style_transfer/Voxel-Carving/``
`mkdir build`
`cd build`
`cmake ..`
`make`
This will build all of the C++.

2. **Carving:**




