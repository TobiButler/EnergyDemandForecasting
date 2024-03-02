# Use the official Ubuntu as the base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# install linux packages
RUN apt-get update && \
    apt-get install -y \
    curl \
    sudo \
    systemctl \
    gnupg2 \
    wget 

# Copy permission files into the container
COPY \Configuration \Configuration 

# # update permissions to allow install of conda and mamba
RUN mv "src/Configuration/Zscalar_certificate.cer" /usr/local/share/ca-certificates/Zscalar_certificate.crt 
RUN sudo update-ca-certificates

# # install mamba package manager
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" 
RUN bash Miniforge3.sh -b -p "${HOME}/conda" 
RUN rm Miniforge3.sh 

# SHELL ["/bin/bash", "-c"] 
# RUN bash "${HOME}/conda/etc/profile.d/conda.sh" 
# RUN bash "${HOME}/conda/etc/profile.d/mamba.sh" 


# set up virtual environment using mamba
RUN "${HOME}/conda/condabin/conda" config --set ssl_verify "/etc/ssl/certs/Zscalar_certificate.pem" && \
    "${HOME}/conda/condabin/mamba" env create --file Configuration/env.yml && \
    echo 'export PATH="$PATH:${HOME}/conda/envs/Test/bin"' >> ~/.bashrc

# ENV PATH="/${HOME}/conda/envs/pf_env/bin:${PATH}"

# location of python
# "${HOME}/conda/envs/pf_env/bin/python3.11"

# HERE: need to troubleshoot this. Need to be able to edit code in both repo and running container. Add argument to main method that specifies the directory to save the results. Need to save excel files to host directory...
# RUN "${HOME}/conda/envs/pf_env/bin/python3.11" "/Code/Master.py" 


# copy code into the container
COPY src /app

# Make port 8050 available to the world outside this container
EXPOSE 8050

# CMD ["${HOME}/conda/envs/Test/bin/python3.11", "app/test.py"]

# Set working directory
WORKDIR /

# Use volume mounting to make local code changes immediately available in the container
# VOLUME /

# Define default command when the container starts
# ENTRYPOINT ["/code/docker_script.sh"]

### how to use this file ###
# make sure docker desktop is running in the background
# 
