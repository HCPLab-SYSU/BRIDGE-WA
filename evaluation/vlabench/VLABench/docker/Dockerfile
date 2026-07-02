ARG BASE_IMAGE=${BASE_IMAGE:-null}
ARG PUBLIC_IMAGE=${PUBLIC_IMAGE:-null}
FROM ${BASE_IMAGE} AS public
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive

COPY ./asset /tmp/build
RUN apt update &&\
    apt-get install -y --no-install-recommends tzdata && \
    echo "China/Shanghai" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/China/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    xargs apt install -y < /tmp/build/apt_packages.txt

RUN mkdir -p /app/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /app/miniconda3/miniconda.sh \
    && chmod +x /app/miniconda3/miniconda.sh \
    && /app/miniconda3/miniconda.sh -b -u -p /app/miniconda3 \
    && rm /app/miniconda3/miniconda.sh \
    && /app/miniconda3/bin/conda install python=${PYTHON_VERSION} \
    && /app/miniconda3/bin/conda config --set show_channel_urls yes \
    && /app/miniconda3/bin/conda config --set auto_update_conda false \
    && /app/miniconda3/bin/conda init --all

SHELL ["/app/miniconda3/bin/conda", "run", "/bin/bash", "-i", "-l", "-c"]
RUN pip install -r /tmp/build/requirements.txt
RUN mkdir -p /app && cd /app \
    && git clone https://github.com/OpenMOSS/VLABench.git \
    && cd VLABench \
    && pip install -r requirements.txt \
    && pip install -e . 

RUN cd /app/VLABench \
    && git submodule update --init \
    && sed -i 's|git@github.com:Physical-Intelligence/aloha.git|https://github.com/Physical-Intelligence/aloha.git|; s|git@github.com:Lifelong-Robot-Learning/LIBERO.git|https://github.com/Lifelong-Robot-Learning/LIBERO.git|' ./third_party/openpi/.gitmodules \
    && git submodule update --init --recursive \
    && sed -i 's/host: str = "10\.176\.52\.118"/host: str = "127\.0\.0\.1"/' third_party/openpi/examples/vlabench/eval.py 

RUN cd /app \
    && conda create -n lerobot python=3.10 \
    && conda init bash \
    && echo "source activate lerobot" > ~/.bashrc \
    && source ~/.bashrc \
    && conda activate lerobot \
    && git clone https://github.com/huggingface/lerobot.git \
    && cd lerobot \
    && conda install ffmpeg \
    && pip install --no-binary=av -e . 

RUN cd /app \
    && git clone https://github.com/kpertsch/rlds_dataset_builder.git \
    && cd rlds_dataset_builder \
    && conda env create -f environment_ubuntu.yml 


SHELL ["/bin/bash", "-i", "-l", "-c"]
RUN cd /app \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp -r ${HOME}/.local/ /app/.local/ \
    && source /app/.local/bin/env \
    && cd /app/VLABench/third_party/openpi \
    && uv venv --python 3.10 examples/vlabench/.venv \
    && source examples/vlabench/.venv/bin/activate \
    && uv pip sync examples/vlabench/requirements.txt \
    && uv pip install -e packages/openpi-client \
    && uv pip install -e /app/VLABench \
    && uv pip install plotly dash scikit-learn addict pandas networkx PyOpenGL==3.1.0 transformers torch accelerate pydantic httpx distro imageio jiter \
    && cd /app/VLABench \
    && git clone https://github.com/motion-planning/rrt-algorithms.git \
    && uv pip install -e rrt-algorithms 


##############################################
FROM ${PUBLIC_IMAGE} AS private
ARG TORCH_HOME
ARG _USER
ARG _UID
ARG _GID
ARG PW
ARG _HOME
ARG _PROJECTS=${_HOME}

COPY ./asset /tmp/build
RUN xargs apt install -y < /tmp/build/apt_packages.txt
# RUN pip install -r /tmp/build/requirements.txt

# Option1: Using unencrypted password/ specifying password
RUN usermod --password $(echo ${PW} | openssl passwd -1 -stdin) root
RUN useradd -m ${_USER} -d ${_HOME} --uid=${_UID} -s /bin/bash && echo "${_USER}:${PW}" | chpasswd


RUN chown ${_USER}:${_USER} -R /tmp/build
RUN chown ${_USER}:${_USER} -R /app


USER ${_UID}:${_GID}
WORKDIR ${_HOME}

RUN cd /app/VLABench \
    && sed -i 's#save_dir="/Your/path/to/trajectory/dataset"#save_dir="${HOME}/data/vlabench/trajectory/dataset"#' ./dataset_generation.sh

RUN /app/miniconda3/bin/conda config --set show_channel_urls yes \
    && /app/miniconda3/bin/conda config --set auto_update_conda false \
    && /app/miniconda3/bin/conda init --all

ENV TORCH_HOME=${TORCH_HOME}

COPY --chown=${_USER}:${_USER} ./asset/ssh .ssh
RUN chmod 0700 .ssh && chmod 600 .ssh/id_rsa && chmod 644 .ssh/id_rsa.pub \
    &&ssh-keyscan ssh.github.com >> .ssh/known_hosts \
    && cp /tmp/build/gitconfig ${_HOME}/.gitconfig \
    && cp /tmp/build/download-vs-code-server.sh ${_HOME}/

RUN rm -rf /tmp/build

ENV PYTHONPATH=.:${PYTHONPATH}

CMD "/bin/bash"


