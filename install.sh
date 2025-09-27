apt install libgtest-dev libssl-dev python3 python3-pip
git submodule update --init --recursive

# please fill in the location of the nvcc compiler based on your CUDA toolkit installation status. 
# specifically, modify the `CUDA_VERSION` variables under `files/make/Makefile-*`.
sed -i "9s|.*|CXX=$(which nvcc)|" files/make/Makefile-PFalcon
sed -i "9s|.*|CXX=$(which nvcc)|" files/make/Makefile-Aegis
sed -i "9s|.*|CXX=$(which nvcc)|" files/make/Makefile-Mizar

# build pfalcon, aegis and mizar.
./scripts/quick_make.sh

# all the binary files will be installed under `./output/bin/`.