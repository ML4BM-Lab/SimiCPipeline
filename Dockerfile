################## BASE IMAGE ######################
FROM python:3.10-slim-trixie@sha256:d62248f0057e027c5302c00736707a6f665c58ce7f8ee6b1da37d6382d0dc954 AS base

################## METADATA ######################
LABEL base.image="slim-trixie"
LABEL version="1"
LABEL software="poetry / python3.10 / magic-impute / SimiCPipeline"
LABEL software.version="Dec2025"
LABEL about.summary="Docker image to run SimiCPipeline using poetry package manager"
LABEL about.tags="Transcriptomics"
LABEL about.maintainer="GitHUb: @ML4BM-Lab; @irenemaring"

################## BASE INSTALLATION ######################
# Update/Install build dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends curl git vim && \
rm -rf /var/lib/apt/lists/*
RUN echo "alias ll='ls -la'" >> ~/.bashrc
# ################# POETRY ######################
FROM base AS poetry
RUN curl -sSL https://install.python-poetry.org | python3
ENV PATH="/root/.local/bin:${PATH}"

################## SIMIC ######################
FROM poetry AS simic
WORKDIR /home/

RUN git clone https://github.com/ML4BM-Lab/SimiCPipeline.git \
    && cd SimiCPipeline && \
    poetry install --with dev && \
    poetry run python -m pip install -e . && \
    poetry run python -m ipykernel install --user --name simicpipeline --display-name "SimiC Pipeline Kernel" && \
    poetry run python -c "import simicpipeline; print('SimiCPipeline successfully installed')"

WORKDIR /home/SimiCPipeline
EXPOSE 8888
CMD ["/bin/bash"]