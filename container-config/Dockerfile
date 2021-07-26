FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /audiotx

COPY container-config/requirements.txt .
RUN pip install -r requirements.txt

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

CMD ["bash"]

#docker image build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file container-config/Dockerfile --tag audiotx:v1 .
#docker run  --shm-size=10g --gpus "device=0" -it -v $(pwd):/audiotx -v /home/purnima/appdir/Github/DATA/nsynth.64.76.dl:/data --rm audiotx:v1