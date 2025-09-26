# download necessary datasets.
mkdir -p files/MNIST; mkdir -p files/CIFAR10
pip install torch torchvision
python3 scripts/download_mnist.py
python3 scripts/download_cifar.py

# run the experiments.
# by default, only the online phase of the protocol is executed. 
# if you wish to run an end-to-end experiment, please add the "-s" parameter.
python3 ./scripts/quick_exp.py -m func  -p falcon aegis mizar -c 0 1 2                                                          # func test
python3 ./scripts/quick_exp.py -m snni  -p falcon aegis mizar -c 0 1 2 --models lenet vgg16 --num_iterations 10                 # snni test
python3 ./scripts/quick_exp.py -m train -p falcon aegis mizar -c 0 1 2 --models lenet vgg16 --num_iterations 10 --batch_size 32 # snnt test