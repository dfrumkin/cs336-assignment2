# 1. Go to your persistent storage (e.g. /home/ubuntu or /mnt)
cd ~
mkdir -p tools && cd tools

# 2. Download the latest Nsight Systems .run installer (example version)
wget https://chatgpt.com/c/690c9bb4-75f4-8326-a70b-2a85a72b27c4#:~:text=can%20download%3A%0Ahttps%3A/-,/,-developer.download.nvidia

# 3. Make it executable
chmod +x nsight-systems-linux-public-*.run

# 4. Extract it (non-root, local install)
./nsight-systems-linux-public-*.run --target ~/tools/nsight-systems --extract-only

# 5. Add to PATH permanently
echo 'export PATH=$HOME/tools/nsight-systems/target-linux-x64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 6. Verify
nsys --version
