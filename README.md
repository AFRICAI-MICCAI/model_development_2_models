# AFRICAI/MICCAI AI in Medical Imaging in Africa Summer School - 2023

<img src="Images/AFRICAI_banner.jpg" alt="Overview"/>

Materials for the 1st AFRICAI Summer School session for the session
"Model Development 2: Model-centric best practices and common pitfalls and open access infrastructures". 
For more information, see the AFRICAI website: https://africai.org/summer-school/. 
 
# 1. Open Access infrastructures
*Open source* in software development refers to the practice of making the source code of a project available to the public to be used, modified, and redistributed.

*Infrastructure* incompasses everything from frameworks to platforms, libraries, toolboses and notebook environments.

The LF AI & Data landscape explores open source projects in Artificial Intelligence and Data and their respective sub-domains. More information can be found here. https://landscapeapp.cncf.io/lfai/. 

Specifically in medical imaging, important (European) research infrastructres (RIs) are [Euro-BioImaging](https://www.eurobioimaging.eu/), [ERIC](https://research-and-innovation.ec.europa.eu/strategy/strategy-2020-2024/our-digital-future/european-research-infrastructures/eric_en), and (EOSC)[https://eosc-portal.eu/]. The EU also has a strategic roadmap on RIs: https://www.esfri.eu/. See the slides for further information: no notebooks on this topics are provided.

# 2. AI Toolboxes
Several open source/open access AI toolboxes are available for deep-learning projects. Some of the most Python popular solutions in medical imaging are:

- PyTorch, developed by Meta AI (https://pytorch.org/).
- TensorFlow, developed by Google Brains (https://www.tensorflow.org/).

We here focus on MONAI, see below, which is based on PyTorch and specifically designed for medical imaging. See the slides for further information: no notebooks on this topics are provided.

# 3. MONAI
MONAI (https://monai.io/) is an open-source framework built specifically for using AI in medical imaging. It provides a comprehensive set of tools and utilities for deep learning applications in medical image analysis. Developed with a focus on flexibility and ease of use, MONAI simplifies the process of building, training, and evaluating deep learning models for tasks like image segmentation, classification, and registration. Adopted by the medical imaging community, MONAI has a lot of backing, and many of the novel proposed
methods are nowadays developed in or added to MONAI.

We will focus in this tutorial on MONAI Core, which can be used for model development. For the other components, we refer you to:

- MONAI Label, MONAI's annotation platform: https://monai.io/label.html 
- MONAI Deploy, MONAI's solution to deploy trained models in the clinical workflow: https://monai.io/deploy.html 
- MONAI Stream, MONAI's SDK to build streaming inference pipelines: https://docs.monai.io/projects/stream/en/latest/index.html

In [**the main notebook of this session**](https://github.com/AFRICAI-MICCAI/model_development_2_models/blob/main/Notebooks/3-%20MONAI.ipynb), we will teach
you the basics on how to use MONAI. Specifically, demonstrated features include:

- Downloading and preparing data from MedNIST for training and validation.
- Using MONAI Transforms to homogenous data and perform data augmentations.
- Setup Pytorch Lightning for streamlined training and testing.
- Using predefined convolutional neural networks from MONAI to classify the 2D images.

Other relevant training materials:
- Tutorials from MONAI itself: https://github.com/Project-MONAI/tutorials/tree/main 
- The most recent MONAI bootcamp: https://github.com/Project-MONAI/monai-bootcamp, and also the video recordings at https://www.youtube.com/playlist?list=PLtoSVSQ2XzyAJAGzaHF0nUIkav0BnxhrJ. For the 2021 bootcamp, see https://github.com/Project-MONAI/MONAIBootcamp2021.

# 4. Monitoring
Various tools can be used to monitor your models during training, but also archive them in an organised way after training. 

- MLFlow (https://mlflow.org/), a model agnostic toolkit with a wide support of AI toolboxes. 
- Tensorboard (https://www.tensorflow.org/tensorboard), developed by Tensorflow, but compatible with some other toolboxes. 

We will focus on MLFlow, as it comes as part of the MONAI installation. We have included an optional [notebook](https://github.com/AFRICAI-MICCAI/model_development_2_models/blob/main/Notebooks/4-MONAI_MLFlow-[optional].ipynb) in this repository. See another example in the MONAI tutorials (https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/spleen_segmentation_mlflow.ipynb).

# 5. Model and hyperparameter selection
TO ADD APOSTOLIA

# 6. Processing
For model training, we need a compute power to run the analysis, i.e., GPUs. Google Colab is fine for small (educational)
experiments, but does not provide a lot of GPU space and limited time. Some environments we would advice to use instead:

- Amazon Web Services (AWS):
- Microsoft Azure

Note that most of these environments are not free, but some allows educational accounts depending on your university. Also,
some of the research infrastructures mentioned in the slides may offer some resources.
See the slides for further information: no notebooks on this topics are provided.

# 7. Pre-trained models
Models pretrained on another dataset can either be directly used on your own dataset or finetuned, i.e., the weights and biases are initialized
with the pretrained model weights. They may therefore result in both an efficiency and performance boost, but no guarantees. A collection of pretrained medical imaging models can be found in these platforms.

- MONAI's model zoo, see https://monai.io/model-zoo.html ([notebook](https://github.com/AFRICAI-MICCAI/model_development_2_models/blob/main/Notebooks/7-%20Pretrained-Models-MONAI_Model_Zoo.ipynb) included in this repository)
- Kaggle: https://www.kaggle.com/models
- Hugging face: tutorial https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt

# 8. Containerization
Containerization means bundling your software solution (e.g., trained model) in a deployable container to allow running it on any system without
requiring to install all kind of software. Popular solutions are:

- Docker, see also the Docker on MONAI at https://hub.docker.com/r/projectmonai/monai.
- Kubernetes, which MONAIs Inference Service also uses, see https://github.com/Project-MONAI/monai-deploy-app-server/blob/main/components/inference-service/README.md.

This only becomes relevant at the deployment stage, or when benchmarking as many challenges also make use of containers to use your algorithm. Hence if you want to use your model in other hospitals or in a federated learning setting, consider making use of these services. See the slides for further information: no notebooks on this topics are provided.

# Contact
Coordinators:

- Martijn P. A. Starmans (m.starmans@erasmusmc.nl)
- Apostolia Tsirikoglou (apostolia.tsirikoglou@ki.se)

Contributers:

- Mahlet Birhanu
- Douwe Spaanderman