
sudo apt update
sudo apt install build-essential dkms linux-headers-$(uname -r)
sudo apt install nvidia-driver-550


sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
##sudo apt install nvidia-utils-575-server
sudo apt install nvidia-driver-550
##sudo apt install cuda-12-4
sudo reboot

##make sure cuda is 12.4
##not recommended, but recommend to see nvidia website, downloading run file version... please do not use deb to install...
##During install, say NO to driver install if you already have a working driver.(on saturn cloud)
##install like this:
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

#for hard driver install:
sudo apt install build-essential dkms linux-headers-$(uname -r)
#then install the cuda version from official website

#how to block the open-source driver:
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot

##conda:
https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh



conda create -n dora python=3.10
pip install -r requirements.txt
pip install torch transformers datasets bitsandbytes huggingface-hub nvidia-ml-py3 
pip install accelerate deepspeed

pip install accelerate
~/.local/bin/accelerate config
export PATH="$HOME/.local/bin:$PATH"

deepspeed manual:
https://www.deepspeed.ai/docs/config-json/#parameter-offloading

for clould lab:(personal tokens of github)


for 5070:
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128