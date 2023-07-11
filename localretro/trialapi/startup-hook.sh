conda env update --name base --file LocalRetro.yml
conda clean --all --force-pkgs-dirs --yes

eval "$(conda shell.bash hook)"
conda activate base
echo 'eval "$(conda shell.bash hook)" && conda activate LocalRetro' >> ~/.bashrc

pip install opencv-python
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev