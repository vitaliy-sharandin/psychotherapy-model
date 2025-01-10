# Psychotherapy model agent

## Fine-tune
Run the notebook, built model will appear in `target` folder.

## Prepare GGUF distribution file
* Install llama.cpp
* Install `requirements.txt`
* Convert you HF tensors to GGUF <br> `python convert_hf_to_gguf.py --outtype q4_0 --outfile <gguf_output_location>\<file_name>.gguf <model_folder_location>`

## Build Docker image
* Build <br> `docker build -t psy-model .`
* Change tag <br> `docker tag psy-model:latest 557125504363.dkr.ecr.us-west-2.amazonaws.com/psy-model:latest`

## Launch Docker container
`docker run --runtime=nvidia --gpus=all -p 8000:8000 psy-model`

## Docker push
* Registry login <br> `aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 557125504363.dkr.ecr.us-west-2.amazonaws.com`
* Push <br>
`docker push 557125504363.dkr.ecr.us-west-2.amazonaws.com/psy-model:latest`

## Azure AKS deployment
1. Create cluster with GPU pool node.
1. Install nvidia device plugin to a gpu node k8s. <br> `az aks command invoke \
    --resource-group myResourceGroup \
    --name myAKSCluster \
    --command "kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml"`
1. Restart node.
1. Label gpu pool node as `pool:gpunode`.
1. Deploy the pod using yml from `deployment`
