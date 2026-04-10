# Instructions to Run Modernized ConceptGraphs



```bash
mkdir data && cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
```
## Create Python environment and run shell if necessary
```bash
uv venv --python 3.10.12
source .venv/bin/activate

# Optional for CUDA 12.6 Downloaders
chmod +x ./use-cuda-126.sh
source ./use-cuda-126.sh

# Install the dependencies from the lock file
uv sync
```

We are going to install CUDA toolkit 12.6 (because it's native to Jetpack 6.2 which is used on Jetson Orin chips), but you do not need to worry about personal environments as you can have multiple CUDA drivers and simply change your CUDA path to download Python environment variables with this shell.
```bash
# Simply run the commands or copy and create a shell for per project usage.
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
echo "Using CUDA at $CUDA_HOME"
```
Check your Nvidia Driver version.

If you do not have CUDA 12.6, run:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install EXACTLY CUDA toolkit 12.6 (keeps 11.8 installed)
sudo apt-get install -y cuda-toolkit-12-6
```

## Clone the Repo

This is the ssh command, use https if you aren't familiar with ssh. There's no difference
```bash
git clone git@github.com:aitechroberts/neuro-nav.git
# Create your own branch for development
git branch -c your-branch-name
```



## Make the Rerun viewer work on WSL2 (Linux)
I don't have a pure Linux machine or boot into one if someone wants to add the changes necessary for that here below these commands. If you have further problems after this. Go [here](https://rerun.io/docs/getting-started/troubleshooting?utm_source=chatgpt.com#running-on-linux) to go to Rerun's troubleshooting page.

- WSL2
```bash
# Remove GL caps from ~/.bashrc (these force GL 3.3)
sed -i '/MESA_GL_VERSION_OVERRIDE/d;/MESA_GLSL_VERSION_OVERRIDE/d;/LIBGL_ALWAYS_INDIRECT/d' ~/.bashrc

# Prefer X11 over Wayland (Rerun docs suggest unsetting Wayland)
echo 'unset WAYLAND_DISPLAY' >> ~/.bashrc
echo 'export GDK_BACKEND=x11' >> ~/.bashrc
echo 'export XDG_SESSION_TYPE=x11' >> ~/.bashrc
source ~/.bashrc

sudo add-apt-repository ppa:kisak/kisak-mesa -y
sudo apt-get update
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools
echo 'export WGPU_BACKEND=vulkan' >> ~/.bashrc
source ~/.bashrc

# Rerun's docs suggest a fresh Mesa Vulkan add
```
- Linux
```bash
sudo apt-get -y install \
    libclang-dev \
    libatk-bridge2.0 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libssl-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    patchelf
```

#### How to Build Pytorch3D from Source if Necessary
Ensure your .venv is activated then run the command
```bash
# Generally takes around 15-30 mins according to community
uv pip install --no-build-isolation --no-deps \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Optional check to make sure everything worked
python - << 'PY'
import torch, pytorch3d
from pytorch3d import ops
print("OK:", torch.__version__, torch.version.cuda, pytorch3d.__file__)
PY
```

## Environment Variables
Be sure to set an OpenAI API Key or a dockerized vLLM container holding the model (easiest to use with direct support from HuggingFace)

```bash
export OPENAI_API_KEY=your-key-here
```



#### How to Fork the CG Repo from a Specific Commit

I forked the repo from concept-graphs and named it neuro-nav in the fork in the GitHub UI. Then cloned it to my personal development environment and ran the following commands.

```bash
git remote add upstream https://github.com/concept-graphs/concept-graphs.git
git fetch upstream --tags --prune
# Create temp branch from that specific commit using the full hash
git switch -c ali-dev-new-local a13fb6ea0b3fdff891b0e33aaf8b972a0cabeb29
git push -u origin ali-dev-new-local
git switch main
git reset --hard ali-dev-new-local
git push -f origin main
# Remove the temp branch
git push origin --delete ali-dev-new-local
```