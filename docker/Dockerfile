FROM registry.gitlab.com/nyker510/analysis-template/gpu:45e6cc9a5c4fb3a469b74bb197005c8536ab44b6
ENV PATH=${PATH}:/home/penguin/.local/bin

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt --user

WORKDIR /workdir
