```shell
sudo apt install libgtest-dev libssl-dev
make -j8 PIRANHA_FLAGS="-DFLOAT_PRECISION=13 -DTWOPC"

export CUDA_VISIBLE_DEVICES=1
./piranha -p 0 -c config.json | tee p0.log
./piranha -p 1 -c config.json | tee p1.log
./piranha -p 2 -c config.json | tee p2.log

# Experiment for P-Falcon.
rm -rf build
make -j8 PIRANHA_FLAGS="-DFLOAT_PRECISION=13 -DPFALCON"
./piranha-mizar -p 0 -c config.json | tee p0.log
./piranha-mizar -p 1 -c config.json | tee p1.log
./piranha-mizar -p 2 -c config.json | tee p2.log

./piranha-aegis -p 0 -c config.json | tee p0.log
./piranha-aegis -p 1 -c config.json | tee p1.log
./piranha-aegis -p 2 -c config.json | tee p2.log

make -j8 PIRANHA_FLAGS="-DFLOAT_PRECISION=13"

python3 ./scripts/quick_exp.py -m func -p falcon aegis mizar -c 0 1 2
python3 ./scripts/quick_exp.py -m snni -p falcon mizar -c 0 0 1 --models lenet vgg16 --num_iterations 10 | tee memory_exp.log
python3 ./scripts/quick_exp.py -m train -p falcon aegis mizar -c 1 2 3 --models lenet vgg16 --num_iterations 10 --batch_size 32 | tee memory_exp.log
```
 