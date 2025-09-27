apt install libgtest-dev libssl-dev python3 python3-pip
git submodule update --init --recursive

# We need to locate the NVCC compiler binary and populate its path into the 
# Makefile under files/make. An automated method is provided here to detect 
# the NVCC compiler location. If this method fails, users can manually 
# locate the NVCC compiler using the whereis nvcc command.
sed -i "9s|.*|CXX=$(which nvcc)|" files/make/Makefile-PFalcon
sed -i "9s|.*|CXX=$(which nvcc)|" files/make/Makefile-Aegis
sed -i "9s|.*|CXX=$(which nvcc)|" files/make/Makefile-Mizar

# build pfalcon, aegis and mizar.
./scripts/quick_make.sh

# all the binary files will be installed under `./output/bin/`.
