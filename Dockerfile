FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# install miniconda. based on https://hub.docker.com/r/continuumio/miniconda/~/dockerfile/
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/opt/conda/bin:$PATH
RUN apt-get update --fix-missing && \
  apt-get install -y wget bzip2 ca-certificates libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 git && \
  wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc &&  \
  conda update -n base conda && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# install fastai
COPY fastai /opt/fastai
RUN ["/bin/bash", "-c", "cd /opt/fastai && conda env update && source activate fastai && cd /opt/fastai && pip install -e ."]
# install quick nlp
COPY setup.py /opt/quicknlp/
COPY src /opt/quicknlp/src
# setup matplotlib
COPY matplotlibrc /root/.config/matplotlib
# setup jupyter lab
COPY jupyter_notebook_config.py /root/.jupyter/
RUN ["/bin/bash", "-c", "source activate fastai && cd /opt/quicknlp && pip install -e . && conda install jupyterlab && jupyter serverextension enable --py jupyterlab --sys-prefix"]
EXPOSE 8888
WORKDIR /workspace
CMD [ "/bin/bash", "-c", "source activate fastai && jupyter lab --allow-root"]



