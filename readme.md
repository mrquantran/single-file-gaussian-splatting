export CXX=g++

pip install ninja (try uninstall)

xatlas

sudo apt-get install graphviz

export CXX=/usr/bin/g++-9
export CC=/usr/bin/gcc-9
sudo apt install gcc-9 g++-9

git clone https://github.com/NVlabs/nvdiffrast
pip install .

export TORCH_CUDA_ARCH_LIST="8.9" # gpu 4070ti
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

--
https://quan283.notion.site/Setup-15e2a3f8f2a180adbbd0ec6a57e39685?pvs=4