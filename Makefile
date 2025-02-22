-include .env

export

PROJECT_NAME = gpu
PACKAGE_NAME = gpu

PWD := $(shell pwd)

DOCKER_IMG := $(PROJECT_NAME):latest
DOCKER_ENV := --env-file .env

DOCKER_RUN := docker run --rm -t

build:
	docker build -f Dockerfile  --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t $(DOCKER_IMG) .

start: build
	docker run --rm $(DOCKER_ENV) -v ./.cache:/home/jovyan/.cache -v ./project:/project -i --gpus all -p 8888:8888 -t $(DOCKER_IMG)
	

shell: build
	docker run --rm $(DOCKER_ENV) -v ./project:/project -i --gpus all -p 8888:8888 -t $(DOCKER_IMG) /bin/bash

test:
	echo $(shell id -g)