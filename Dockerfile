FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3

RUN apt-get update && \
    apt-get install -y \
        subversion \
        unzip \
        make \
        automake \
        autoconf \
        libtool \
        zlib1g-dev \
        wget \
        git \
        python2.7 \
        python3 \
        zlib1g-dev \
        ca-certificates \
        patch \
        emacs-nox \
        sox \
        ffmpeg

RUN git clone https://github.com/kaldi-asr/kaldi.git /workspace/kaldi && \
    cd /workspace/kaldi/tools && \
    ./extras/install_mkl.sh && \
    make -j $(nproc) && \
    cd /workspace/kaldi/src && \
    ./configure --shared --use-cuda && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    find /workspace/kaldi -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    sed 's#`pwd`/../../..#/workspace/kaldi#g' /workspace/kaldi/egs/wsj/s5/path.sh > /workspace/path.sh && \
    echo "source /workspace/path.sh" >> /etc/bash.bashrc
