FROM nvcr.io/nvidia/pytorch:19.10-py3

ARG USERNAME=docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        fish man-db \
      && \
    rm -rf /var/lib/apt/lists/ && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME
ENTRYPOINT ["fixuid", "-q"]
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

RUN sudo -H /opt/conda/bin/pip install \
    thop==0.0.31-1910280903 \
    tqdm==4.32.1 \
    imageio==2.5.0 \
    imageio-ffmpeg==0.3.0

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

CMD ["fish"]
