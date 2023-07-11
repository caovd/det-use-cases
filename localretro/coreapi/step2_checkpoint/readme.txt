# On local machine
git clone https://github.com/kaist-amsg/LocalRetro.git
cd LocalRetro-main
det -m <master ip address> shell create -c .

# Open an existing Determined shell (optional)
det -m <master ip address> shell open shell_id

# On shell, ex., root@de70ad04e764  
cd /
cd run/determined/workdir

# Create a new conda env from LocalRetro.yml provided by customer
conda env create --file=LocalRetro.yml
conda init bash
# For changes to take effect, close and re-open your current shell.
det -m <master ip address> shell open <shell-id>
conda activate LocalRetro

# Install addtional libraries
apt-get install libxrender1 -y
pip install opencv-python
apt update && apt install -y libsm6 libxext6

# Go to Determined's working directory
cd /
cd run/determined/workdir

# Edit the experiment config file (Optional)
cd data/configs
nano default_config.json
cd ..
cd ..

# Run training
cd scripts
python Train.py -d USPTO_50K
