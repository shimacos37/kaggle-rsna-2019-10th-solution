FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7

RUN apt-get update \
     && apt-get install --yes software-properties-common \
     && add-apt-repository ppa:neovim-ppa/stable \
     && apt-get update \
     && apt-get install --yes  --no-install-recommends \
     build-essential \
     cmake \
     curl \
     git \
     htop \
     libgtk2.0-dev \
     libboost-all-dev \
     libgl1-mesa-dev \
     libssl-dev \
     libreadline-dev \
     libyaml-dev \
     libxml2-dev \
     libxslt-dev \
     libncurses5-dev \
     libncursesw5-dev \
     software-properties-common \
     && apt-get clean

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && chmod +x ~/miniconda.sh \
     && ~/miniconda.sh -b -p /opt/conda \
     && rm ~/miniconda.sh \
     && /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing \
     && /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN  pip install --upgrade pip \
     && pip install --ignore-installed opencv-python \
     albumentations==0.3.2 \
     addict \
     lightgbm \
     setuptools \
     tqdm \
     joblib \
     keras \
     pyarrow \
     pyyaml \
     scikit-learn \
     scikit-image \
     future \
     torch \
     torchvision \
     efficientnet-pytorch \
     pretrainedmodels \
     segmentation-models-pytorch \
     pydicom \
     && git clone https://github.com/NVIDIA/apex \
     && cd apex \
     && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . \
     && cd .. \ 
     && rm -r apex

RUN pip install -U --no-deps pytorch-lightning==0.5.3.2 \
     && pip install test_tube \
     numpy==1.16.4 \
     scikit-learn==0.20.2 \
     tqdm==4.35.0 \
     tensorflow-gpu

WORKDIR /root/


