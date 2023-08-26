# Modeling Video as Stochastic Processes for Fine-Grained Video Representation Learning
Video as Stochastic Processes (VSP), a novel process-based contrastive learning framework for fine-grained video pepresentation Learning, which aims to discriminate between video processes and simultaneously capture the temporal dynamics in the processes. Specifically, we enforce the embeddings of the frame sequence of interest to approximate a goal-oriented stochastic process, i.e., Brownian bridge, in the latent space via a process-based contrastive loss.

---

## Environment

```
# Create and activate vsp Environment
conda create -y --name vsp python=3.7.9
conda activate vsp
# Pytorch and Auxiliary installation tools
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install -y conda-build ipython pandas scipy pip av -c conda-forge
```
