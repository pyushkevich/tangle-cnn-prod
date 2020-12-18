# Need CUDA!
FROM anibali/pytorch:cuda-10.0
USER root

# Some packages
RUN apt-get update
RUN apt-get install -y wget curl gcc openslide-tools python2.7 vim

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

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
  && make V=0 \
  && make install

# Needed by VIPS
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Install C3D
ARG C3D_URL=https://sourceforge.net/projects/c3d/files/c3d/Nightly/c3d-nightly-Linux-gcc64.tar.gz
RUN wget -q ${C3D_URL} \
 && tar -zxvf c3d-nightly-Linux-gcc64.tar.gz --strip 1 -C /usr/local

# Copy internals
COPY . /app/

# Run pip
RUN pip install -r /app/requirements.txt

# Switch to actual user again
USER user

# Add pythonpath
ENV PYTHONPATH=/app/wildcat.pytorch:$PYTHONPATH

