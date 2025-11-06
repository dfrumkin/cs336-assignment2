# (optional) sudo
sudo apt update

# Add NVIDIA devtools repository (if needed)
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs)/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(lsb_release -rs)/x86_64/ /"

sudo apt update
sudo apt install nsight-systems

# Verify
nsys --help
nsys status -e
