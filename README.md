# Psychotherapy Llama model 
## Prerequisites
- Python 3.12 or higher
- UV installed
- Docker

## Fine-tune
1. Run the command `uv run src/fine-tuning/psy_model_fine_tuning.py`.
1. Built model will appear in `src/fine-tuning/target` folder.

### (Optional) Prepare GGUF distribution files
* Change Dockerfile model file copying to GGUF file
* Install llama.cpp
* Install `requirements.txt`
* Convert you HF tensors to GGUF <br> `python convert_hf_to_gguf.py --outtype q4_0 --outfile <gguf_output_location>\<file_name>.gguf <model_folder_location>`

## Local model serving

### Build Docker image
* Build <br> `docker build -t psy-model .`
* Change tag <br> `docker tag psy-model:latest psyserviceregistry.azurecr.io/psy-model`

### Launch Docker container
`docker run --runtime=nvidia --gpus=all -p 8000:8000 -p 11434:11434 psy-model`

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
1. Deploy the pod using yml from `k8s`

## Automatic deployment
The repository is configured with github actions, all above actions will be one automatically by GitHub Actions deployment.
