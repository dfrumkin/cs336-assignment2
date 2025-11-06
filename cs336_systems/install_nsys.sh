# Create a tools dir in persistent storage
cd ~/tools || mkdir -p ~/tools && cd ~/tools

# Fetch the latest Nsight Systems .run URL automatically
NSYS_URL=$(curl -s https://developer.download.nvidia.com/devtools/nsight-systems/ \
  | grep -o 'https[^"]*NsightSystems-linux-public-[^"]*\.run' \
  | sort -V | tail -n1)

echo "Latest Nsight Systems installer: $NSYS_URL"

# Download and install to a persistent folder
wget -O nsys.run "$NSYS_URL"
chmod +x nsys.run
./nsys.run --target ~/tools/nsight-systems --extract-only

# Add to PATH (only once)
grep -q "nsight-systems" ~/.bashrc || echo 'export PATH=$HOME/tools/nsight-systems/target-linux-x64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nsys --version