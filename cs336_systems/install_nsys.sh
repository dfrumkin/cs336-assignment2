cd ~/tools || mkdir -p ~/tools && cd ~/tools

# Query the NVIDIA page and extract the latest .run link
NSYS_URL=$(curl -sL https://developer.nvidia.com/nsight-systems/get-started \
  | grep -o 'https://developer\.download\.nvidia\.com/devtools/nsight-systems/NsightSystems-linux-public-[^"]*\.run' \
  | sort -V | tail -n1)

echo "Latest Nsight Systems installer: $NSYS_URL"

# Download and install
wget -O nsys.run "$NSYS_URL"
chmod +x nsys.run
./nsys.run --target ~/tools/nsight-systems --extract-only

# Add to PATH
grep -q "nsight-systems" ~/.bashrc || echo 'export PATH=$HOME/tools/nsight-systems/target-linux-x64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nsys --version
