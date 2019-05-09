# Need CUDA!
FROM anibali/pytorch:cuda-10.0
USER root

# Some packages
RUN apt-get update
RUN apt-get install -y curl gcc openslide-tools python2.7

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && bash /usr/local/gcloud/google-cloud-sdk/install.sh --quiet \
  && chown -R user:user /home/user/.config

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Copy internals
COPY . /app/

# Run pip
RUN pip install -r /app/requirements.txt

# Switch to actual user again
USER user

# Add pythonpath
ENV PYTHONPATH=/app/wildcat.pytorch:$PYTHONPATH

