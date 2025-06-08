#!/bin/bash
mkdir -p output/bin

echo "===================================================="
echo "Building Piranha for P-Falcon wo offline"
rm -rf build
cp ./files/make/Makefile-PFalcon ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13" -j32
mv ./piranha-rss output/bin/piranha-rss 

echo "===================================================="
echo "Building Piranha for Aegis wo offline"
rm -rf build
cp ./files/make/Makefile-Aegis ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13" -j32
mv ./piranha-aegis output/bin/piranha-aegis

echo "===================================================="
echo "Building Piranha for AegisOpt wo offline"
rm -rf build
cp ./files/make/Makefile-Aegis ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13 -DAEGISOPT" -j32
mv ./piranha-aegis output/bin/piranha-aegisopt

echo "===================================================="
echo "Building Piranha for Mizar wo offline"
rm -rf build
cp ./files/make/Makefile-Mizar ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13" -j32
mv ./piranha-mizar output/bin/piranha-mizar

echo "===================================================="
echo "Building Piranha for P-Falcon w offline"
rm -rf build
cp ./files/make/Makefile-PFalcon ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13 -DENABLE_OFFLINE" -j32
mv ./piranha-rss output/bin/piranha-rss-offline

echo "===================================================="
echo "Building Piranha for Aegis w offline"
rm -rf build
cp ./files/make/Makefile-Aegis ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13 -DENABLE_OFFLINE" -j32
mv ./piranha-aegis output/bin/piranha-aegis-offline

echo "===================================================="
echo "Building Piranha for AegisOpt w offline"
rm -rf build
cp ./files/make/Makefile-Aegis ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13 -DAEGISOPT -DENABLE_OFFLINE" -j32
mv ./piranha-aegis output/bin/piranha-aegisopt-offline

echo "===================================================="
echo "Building Piranha for Mizar w offline"
rm -rf build
cp ./files/make/Makefile-Mizar ./Makefile
make PIRANHA_FLAGS="-DFLOAT_PRECISION=13 -DENABLE_OFFLINE" -j32
mv ./piranha-mizar output/bin/piranha-mizar-offline