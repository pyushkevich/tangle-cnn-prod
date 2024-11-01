# Need CUDA!
# FROM anibali/pytorch:1.7.0-cuda11.0
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
USER root

# Some packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget curl gcc openslide-tools vim

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN apt-get install -y sudo
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && bash /usr/local/gcloud/google-cloud-sdk/install.sh --quiet \
  && chown -R user:user /home/user/.config

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Install VIPS
RUN apt-get install -y build-essential pkg-config glib2.0-dev
RUN apt-get install -y libexpat1-dev libtiff5-dev libjpeg-dev libgsf-1-dev
ARG VIPS_VERSION=8.8.0
ARG VIPS_URL=https://github.com/libvips/libvips/releases/download
RUN wget ${VIPS_URL}/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.gz \
  && tar xf vips-${VIPS_VERSION}.tar.gz \
  && cd vips-${VIPS_VERSION} \
  && ./configure \
  && make V=0 -j $(nproc) \
  && make install

# Needed by VIPS
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Install C3D
ARG C3D_URL=https://sourceforge.net/projects/c3d/files/c3d/Nightly/c3d-nightly-Linux-gcc64.tar.gz
RUN wget -q ${C3D_URL} \
 && tar -zxvf c3d-nightly-Linux-gcc64.tar.gz --strip 1 -C /usr/local

# Run pip
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy internals
COPY . /app/

# Switch to our user
USER user

# Add pythonpath
ENV PYTHONPATH=/app/wildcat.pytorch:$PYTHONPATH

