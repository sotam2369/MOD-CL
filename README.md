# Setup of ROAD-R (by MWIT)

## Docker setup with dataset pre installed

Go to root folder. Then, run the following commands.
```
# Go to the parent folder, and git clone the ROAD dataset if not done so already
cd ..
git clone https://github.com/gurkirt/road-dataset.git
cd ROAD-R-2023-fork

# Make the docker image
sudo docker build -f scripts/Dockerfile -t road-r .

# Run the docker
sudo docker run -it --ipc=host --gpus all --name road-r-cont -v "$(pwd)"/../road-dataset:/usr/src/road-dataset road-r

# Run the installation scripts

bash scripts/installation_nodataset.sh
```
