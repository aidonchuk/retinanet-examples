FROM nvcr.io/nvidia/pytorch:19.10-py3

ARG USER=alex
ARG UID=1000
ARG GID=1000
ARG PW=alex
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd


RUN apt-get -y update && apt-get -y upgrade && apt-get -y install curl && apt-get -y install wget && apt-get -y install git && apt-get -y install automake && apt-get install -y sudo && adduser ${USER} sudo
#RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..

# COPY --chown=${USER}:${USER}

RUN pip install git+https://github.com/bonlime/pytorch-tools.git@master

COPY . retinanet/
RUN pip install --no-cache-dir -e retinanet/
RUN pip install /workspace/retinanet/extras/tensorrt-6.0.1.5-cp36-none-linux_x86_64.whl
RUN pip install tensorboardx
RUN pip install albumentations
RUN pip install RPi.GPIO
RUN pip install setproctitle
RUN pip install paramiko
RUN pip install flask
RUN pip install mem_top
RUN pip install arrow
RUN pip install pycuda
RUN pip install torchvision
RUN pip install pretrainedmodels
RUN pip install efficientnet-pytorch
RUN pip install git+https://github.com/qubvel/segmentation_models.pytorch

RUN chown -R ${USER}:${USER} retinanet/

RUN apt-get install -y openssh-server && apt install -y tmux && apt-get -y install bison flex && apt-cache search pcre && apt-get -y install net-tools && apt-get -y install nmap
RUN apt-get -y install libpcre3 libpcre3-dev && apt-get -y install iputils-ping

RUN git clone https://github.com/swig/swig.git && cd swig && ./autogen.sh && ./configure && make && make install && cd ..

RUN wget https://www.baslerweb.com/fp-1551786516/media/downloads/software/pylon_software/pylon-5.2.0.13457-x86_64.tar.gz
RUN tar -xvzf pylon-5.2.0.13457-x86_64.tar.gz && cd pylon-5.2.0.13457-x86_64 && tar -C /opt -xzf pylonSDK-5.2.0.13457-x86_64.tar.gz && cd ..

RUN git clone https://github.com/basler/pypylon.git && cd pypylon && pip install . && cd ..

RUN mkdir /var/run/sshd
RUN echo 'root:pass' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22 5000 6000 6001 7000 7001 8000
CMD ["/usr/sbin/sshd", "-D"]

#USER ${UID}:${GID}
#WORKDIR /home/${USER}
