# Locally Adaptive Neural 3D Morphable Models
This repository provides the official implementation for the paper 
[Locally Adaptive Neural 3D Morphable Models](https://arxiv.org/pdf/2401.02937.pdf) 
accepted at CVPR 2024. 

## Overview
<img src="assets/architecture.png" alt="3D Morphable Model" width="100%" />

## Dependencies

Before you can run this project, you need to install the necessary dependencies. These dependencies are listed in the `requirements.txt` file. Follow these steps to install them:

### Step 1: Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/michaeltrs/LAMM3D_demo.git
cd LAMM3D_demo
```

### Step 2: Set Up a Conda Environment (Optional)

It's recommended to create a Conda environment for managing your Python projects. This helps in keeping your project dependencies isolated and ensures that they don't conflict with other projects or the global Python installation.

Create a Conda environment by running the following command:

```bash
conda create --name LAMM python=3.8
```
Activate the Conda environment:

```bash
conda activate LAMM
```

### Step 3: Install the Dependencies

With the Conda environment activated, install the project dependencies by running:

```bash
pip install -r requirements.txt
```

### Step 4: Running the Project

After the installation of dependencies is complete, you can run the project. Follow the specific instructions provided bellow for training LAMM on your own data or using pretrained models for inference.

## Training
Modify respective .yaml config files accordingly to define model architecture, data (also modify `data/implemented_datasets.yaml`) and save directories. 

### Dataset preparation
Follow the steps below to train LAMM3D on a new dataset. 
- Include the dataset as a new entry in `data/implemented_datasets.yaml`
```
DatasetName:
    basedir: (str) this is the base/root directory for the new data. All provided paths should be relative to this path'
    paths_train: (str) 'file with training paths (relative to basedir)'
    paths_eval: (str) 'file with evaluation paths (relative to basedir)'
    mean_std_file: (str) 'file (.pickle) with per-vertex mean and standard deviation values {'mean': numpy.array (N x 3), 'std': numpy.array (N x 3)}'
    template: (str) 'example (.obj) train/eval sample path'
    num_vertices: (int) number of vertices (N) of train and eval data samples
```
- add the dataset by name in new .yaml config file
```
DATASETS:
  train:
    dataset: DatasetName
    batch_size: training batch size

  eval:
    dataset: DatasetName
    batch_size: evaluation batch size
```

### Dimensionality Reduction (3D Mesh Reconstruction)
Train a model in 3D mesh dimensionality reduction by running:
```bash
python training/dim_reduction.py --config <path to .yaml>  --device <device id>
```

### Mesh Manipulation
For successfully training a model in mesh manipulation, it is critical to initialize from a checkpoint pretrained on 
dimensionality reduction. A path to pretrained checkpoint needs to be added in `CHECKPOINT.load_from_checkpoint` 
in the configuration file. Train a model in 3D mesh manipulation by running:
```bash
python training/manipulation.py --config <path to .yaml>  --device <device id>
```

## Inference

### Prepare a trained model for inference
When training a model in a new dataset several files need to be created to enable manual editing or randomly sampling mesh regions. 
If a checkpoint is downloaded from the section below, these files will be contained in the zip file.
These can also be gnerated for a new model by running:
```bash
python scripts/prepare_inference.py  --config <config .yaml file> --checkpoint <path to checkpoint.pth> --device < device id>
```
The following files are required: 
- `region_ids.pickle`: contains a dictionary with region numbers as keys. Each entry includes all vertices belonging to a region.
- `region_boundaries.pickle`: contains the region boundary vertices per region for a given mesh template and is used for smoothing 
region boundaries during large mesh deformations. 
- `mean_std.pickle`: per-vertex mean and standard deviation of location (XYZ) over the training dataset. It is used for 
normalizing inputs.
- `template.obj`: a data sample, indicates mesh topology and vertex connectivity.

Additionally, the following files will need to 
- `displacement_stats.pickle`: contains K (number of regions) Gaussian distributions fitted on control vertex displacements 
as described in sec.4.3 "Region sampling and disentanglement" of the paper.
- `gaussian_id.pickle`: contains a dictionary with "mean" and "sigma" for a gaussian distribution fitted to training 
data latent codes. Can be used to sample latent codes and generate novel 3D meshes.

### Pretrained Checkpoints
Download pre-trained model from [google drive](https://drive.google.com/drive/folders/16xrCBdvmn1POEHzISUy-SwSnht_Vc_b_?usp=sharing).

### Generate Random Samples 
A trained model can be used to generate new random mesh samples by running:
```bash
python scripts/generate_samples.py  --config <config .yaml file> --checkpoint_name <path to checkpoint.pth> --device <device id>
```

### Generate Random Sample Regions
A trained model can be used to generate new random regions for mesh samples by running:
```bash
python scripts/generate_sample_regions.py  --config <config .yaml file> --checkpoint_name <path to checkpoint.pth> --device <device id>
```
In case no mesh data are available use the ```--generate_random_source``` argument to randomly generate source 3D meshes.

### Manual Manipulation - Blender
Coming soon.
## BibTex
If you incorporate any data or code from this repository into your project, please acknowledge the source by citing the following work:

```
@misc{tarasiou2024locally,
      title={Locally Adaptive Neural 3D Morphable Models}, 
      author={Michail Tarasiou and Rolandos Alexandros Potamias and Eimear O'Sullivan and Stylianos Ploumpis and Stefanos Zafeiriou},
      year={2024},
      eprint={2401.02937},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
