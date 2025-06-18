FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
LABEL maintainer="Qi Li"
WORKDIR /home/TUS-REC2025/
ENV PATH="/opt/miniconda/bin:$PATH"
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda \
    && rm miniconda.sh \
    && apt-get purge -y --auto-remove curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && conda create -y -n freehand-US python=3.9.13 \
    && conda clean -afy \
    && conda run -n freehand-US pip install --no-cache-dir -r requirements.txt \
    && rm -f requirements.txt \
    && conda run -n freehand-US conda install -y pytorch3d --no-deps -c conda-forge \
    && conda clean -afy
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "freehand-US", "python", "test.py"]
