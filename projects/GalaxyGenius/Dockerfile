## https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/
## https://www.cherryservers.com/blog/docker-build-command

# FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y \
#     git \
#     python3-pip \
#     python3-dev \
#     python3-opencv \
#     libglib2.0-0

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set the working directory
WORKDIR /app

# Install any python packages you need
COPY requirements.txt ./

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the server
EXPOSE 8000
CMD ["uvicorn", "server:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
