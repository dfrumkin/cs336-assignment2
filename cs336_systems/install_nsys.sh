# 1. Go to your persistent storage (e.g. /home/ubuntu or /mnt)
cd ~
mkdir -p tools && cd tools

# 2. Download the latest Nsight Systems .run installer (example version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-linux-public-2024.3.1.112-33147281.run

# 3. Make it executable
chmod +x nsight-systems-linux-public-*.run

# 4. Extract it (non-root, local install)
./nsight-systems-linux-public-*.run --target ~/tools/nsight-systems --extract-only

# 5. Add to PATH permanently
echo 'export PATH=$HOME/tools/nsight-systems/target-linux-x64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 6. Verify
nsys --version