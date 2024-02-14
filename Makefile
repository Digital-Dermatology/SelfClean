.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := SelfClean
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD | tr / _)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)

COMMA := ,
DASH := -
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)

# check if `netstat` is installed
ifeq (, $(shell which netstat))
$(error "Netstat executable not found, install it with `apt-get install net-tools`")
endif

# Check if Jupyter Port is already use and define an alternative
ifeq ($(origin PORT), undefined)
  PORT_USED = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '8888' | tail -n 1)
  # Will fail if both ports 9999 and 10000 are used, I am sorry for that
  NEXT_TCP_PORT = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '[0-9]{4}' | sort | tail -n 1 | xargs -I '{}' expr {} + 1)
  ifeq ($(PORT_USED), 8888)
    PORT = $(NEXT_TCP_PORT)
  else
    PORT = 8888
  endif
endif

# docker
ifeq ($(origin CONTAINER_NAME), undefined)
  CONTAINER_NAME := default
endif

ifeq ($(origin LOCAL_DATA_DIR), undefined)
  LOCAL_DATA_DIR := /data/
endif

ifeq ($(origin DOCKER_SRC_DIR), undefined)
  DOCKER_SRC_DIR := "/workspace/"
endif

ifeq ($(origin LOCAL_DATA_DIR), undefined)
  LOCAL_DATA_DIR := /data/
endif

ifeq ($(origin GPU_ID), undefined)
  GPU_ID := all
  GPU_NAME := $(GPU_ID)
else
  GPU_NAME = $(subst $(COMMA),$(DASH),$(GPU_ID))
endif

ifeq ("$(GPU)", "false")
  GPU_ARGS := --gpus '"device="'
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_$(CONTAINER_NAME)
else
  GPU_ARGS := --gpus '"device=$(GPU_ID)"' --shm-size 200G --ipc=host
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_gpu_$(GPU_NAME)_$(CONTAINER_NAME)
endif

# count elements in comma-seperated GPU list
count = $(words $1)$(if $2,$(call count,$(wordlist 2,$(words $1),$1),$2))
GPU_LIST := $(subst $(COMMA),$(SPACE),$(GPU_ID))
NUM_GPUS := $(call count,$(GPU_LIST))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    NUM_CORES := $(shell nproc)
else ifeq ($(UNAME_S),Darwin)
    NUM_CORES := $(shell sysctl -n hw.ncpu)
else
	NUM_CORES := 1
endif
# optimal number of threads is #cores/#gpus
NUM_THREADS := $(shell expr $(NUM_CORES) / $(NUM_GPUS))

DOCKER_ARGS := -v $$PWD:/workspace/ -v $(LOCAL_DATA_DIR):/data/ -p $(PORT):8888 --rm
DOCKER_DGX := docker run \
			  -it \
              -u $(id -u):$(id -g) \
			  -v ${PWD}:/workspace/ \
			  -v /raid/fabian/:/raid/fabian/ \
			  -w /workspace \
			  -d \
			  --gpus='"device=0,1,2,3"' \
              --name $(PROJECTNAME)_multi_gpu \
			  --shm-size 200G \
			  --env-file .env
DOCKER_CMD := docker run $(DOCKER_ARGS) --env-file=.env $(GPU_ARGS) $(DOCKER_CONTAINER_NAME) -it $(PROJECTNAME):$(GIT_BRANCH)
TORCH_CMD := OMP_NUM_THREADS=$(NUM_THREADS) torchrun --standalone --nnodes 1 --node_rank 0 --nproc_per_node $(NUM_GPUS)

# SSH
PORT := 22
USERNAME := fgroger

###########################
# COMMANDS
###########################
# Thanks to: https://stackoverflow.com/a/10858332
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

###########################
# SSH UTILS
###########################
.PHONY: push_ssh
push_ssh: clean  ##@SSH pushes all the directories along with the files to a remote SSH server
	$(call check_defined, SSH_CONN)
	rsync -r --exclude='data/' --exclude='.git/' --exclude='.github/' --exclude='wandb/' --exclude='assets/' --progress -e 'ssh -p $(PORT)' $(PROJECT_DIR)/ $(USERNAME)@$(SSH_CONN):$(PROJECTNAME)/

.PHONY: pull_ssh
pull_ssh:  ##@SSH pulls directories from a remote SSH server
	$(call check_defined, SSH_CONN)
	scp -r -P $(PORT) $(USERNAME)@$(SSH_CONN):$(PROJECTNAME) .

###########################
# PROJECT UTILS
###########################
.PHONY: init
init:  ##@Utils initializes the project and pulls all the nessecary data
	@git submodule update --init --recursive

.PHONY: update_data_ref
update_data_ref:  ##@Utils updates the reference to the submodule to its latest commit
	@git submodule update --remote --merge

.PHONY: clean
clean:  ##@Utils clean the project
	@black .
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f .coverage coverage.xml report.xml
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f -R tmp/
	@rm -f -R cov_html/

.PHONY: install
install:  ##@Utils install the dependencies for the project
	python3 -m pip install -r requirements.txt

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker run an interactive bash inside the docker image (default: GPU=true)
	@echo "Running bash with GPU being $(GPU) and GPU_ID $(GPU_ID)"
	$(DOCKER_CMD) /bin/bash; \

start_jupyter: _build  ##@Docker start a jupyter notebook inside the docker image
	@echo "Starting jupyter notebook"
	@-docker rm $(PROJECTNAME)_gpu_$(GPU_ID)
	$(DOCKER_GPU_CMD) /bin/bash -c "jupyter notebook --allow-root --ip 0.0.0.0 --port 8888"

###########################
# TESTS
###########################
.PHONY: test
test: _build  ##@Test run all tests in the project
    # Ignore integration tests flag: --ignore=test/manual_integration_tests/
	$(DOCKER_CMD) /bin/bash -c "wandb offline && python -m pytest --cov-report html:cov_html --cov-report term --cov=src --cov-report xml --junitxml=report.xml ./ && coverage xml"
