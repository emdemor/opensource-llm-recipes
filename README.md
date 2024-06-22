# GPU-Docker

## 0. Desinstale
sudo apt remove nvidia-*
sudo apt-get remove --purge nvidia-*
sudo apt remove --purge '^nvidia-.*'
sudo apt remove --purge '^libnvidia-.*'
sudo apt autoremove


## 1. Install Nvidia-Driver - 5.15, valid for cuda 11.7

```
export DISTRO=ubuntu2004
export ARCH=x86_64
sudo add-apt-repository -r restricted
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-$DISTRO-keyring.gpg
mv cuda-$DISTRO-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/ /" | sudo tee /etc/apt/sources.list.d/cuda-$DISTRO-$ARCH.list
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-$DISTRO.pin
sudo mv cuda-$DISTRO.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt update
sudo apt install -y nvidia-driver-515
```

<!-- sudo apt-mark hold nvidia-driver-515 -->

## 1. Install Cuda 11.7

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb

sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/

wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | sudo apt-key add -

sudo apt-get update

sudo apt-get -y install cuda-11-7
```

## 2. Instalar `nvidia-docker2`

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker
```
