# Psychotherapy Llama model 
## Prerequisites
- Python 3.12 or higher
- UV package manager
- Docker
- Azure CLI

## Data version management
DVC data maangement is automatically installed, so you can version control your data.
- There is an option for data version management through DVC
- Once data is checked out, then in `.\src\fine-tuning\psy_ai_huggingface_model_registry_demo.ipynb` change the typical huggingface data loading to local one.

## Fine-tune
1. Run the command `uv run jupyter nbconvert --to notebook --execute .\src\fine-tuning\psy_ai_huggingface_model_registry_demo.ipynb`
1. Built model will appear in `src/fine-tuning/target` folder.

### (Optional) Prepare GGUF distribution files
* Change Dockerfile model file copying to GGUF file
* Install llama.cpp
* Install `requirements.txt`
* Convert you HF tensors to GGUF <br> `python convert_hf_to_gguf.py --outtype q4_0 --outfile <gguf_output_location>\<file_name>.gguf <model_folder_location>`

## MLFlow experiment tracking
Automatic MLFlow experiment tracking is integrated into the repo. Server will be started under adress `localhost:8080`.

## Local model serving

### Build Docker image
* Build <br> `docker build -t psy-model .`
* Change tag <br> `docker tag psy-model:latest psyserviceregistry.azurecr.io/psy-model`

### Launch Docker container
`docker run --runtime=nvidia --gpus=all -p 8000:8000 -p 11434:11434 psyserviceregistry.azurecr.io/psy-model`

## Azure Cloud Deployment
### Docker push
* Azure Registry login <br> `az acr login --name <psyserviceregistry>`
* Push <br>
`docker push psyserviceregistry.azurecr.io/psy-model`

### Azure AKS deployment
1. Create cluster with GPU pool node.
1. Install nvidia device plugin to a gpu node k8s. <br> `az aks command invoke \
    --resource-group myResourceGroup \
    --name myAKSCluster \
    --command "kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml"`
1. Restart node.
1. Label gpu pool node as `pool:gpunode`.
1. Deploy the pod using yml `k8s/deploy.yml`

## Automatic deployment
The repository is configured with **GitHub Actions** for automated deployment. All steps, including Docker build, push, and AKS deployment, are performed automatically upon code changes.

### Auto Deployment Requirements
This repo uses local Github Actions Runner, as you need a GPU to run model fine-tuning.
- Local system with GPU
- Install [Github Actions Runner for local launch](https://github.com/actions/runner/releases)
- Run `.github/workflows/start-runner.ps1` to start the local runner 
- Push changes and your runner will perform work    
