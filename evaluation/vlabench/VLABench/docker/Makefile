SHELL=/bin/bash
PRIVATE_REGISTRY:=
# PRIVATE_REGISTRY:=docker.home.jjgong.com:20443/
BASE_IMAGE:=m.daocloud.io/docker.io/nvidia/cuda:11.8.0-devel-ubuntu20.04
# BASE_IMAGE:=${PRIVATE_REGISTRY}/nvidia/cuda:12.8.0-devel-ubuntu20.04


PROJECT_NAME:=vlabench_warmup_02
TEAM_NAME:=jerrikeph
PUBLIC_IMAGE:=${PRIVATE_REGISTRY}${TEAM_NAME}/${PROJECT_NAME}

# HTTPS_PROXY:=${https_proxy}
# HTTP_PROXY:=${http_proxy}
HTTPS_PROXY:=
HTTP_PROXY:=
_USER := $(shell whoami)
_UID := $(shell id -u)
_GID := $(shell id -g)
PW:=rootpass

LOCAL_NAME:=exp/${_USER}_${PROJECT_NAME}
PRIVATE_IMAGE:=${PRIVATE_REGISTRY}${LOCAL_NAME}
CONTRAINER_NAME=$(subst /,_,$(LOCAL_NAME))

PROJ_DIR:=${HOME}/projects
SHARE_DIR:=${HOME}/data
TORCH_HOME:=${SHARE_DIR}/torch_home

CHOME:= ${HOME}
TO_SHARE_DIR:=${SHARE_DIR}
TO_PROJ_DIR:=${CHOME}/projects
TO_TORCH_HOME:=${TO_SHARE_DIR}/torch_home

MAKEFILE_ := $(CURDIR)/$(firstword $(MAKEFILE_LIST))

USE_GPUS:=$(shell if [ -z "$$(lspci | grep NVIDIA)" ]; then echo ""; else echo "--gpus all"; fi)

#TODO uncomment run: _pull _instantiate_container _exec

run:  _check_image _instantiate_container _exec

run_root:  _check_image _instantiate_container _exec_root

run_refresh: kill run

kill:
	if docker container ls | grep -w ${CONTRAINER_NAME}; then \
		echo "container ${CONTRAINER_NAME} exists, killing it first"; \
		docker container kill  ${CONTRAINER_NAME}; fi

build_private: _setup_sshkey _setup_gitconfig
	DOCKER_BUILDKIT=1 docker build --no-cache --pull --target private \
		--build-arg PUBLIC_IMAGE=${PUBLIC_IMAGE} \
		--build-arg TORCH_HOME=${TO_TORCH_HOME} \
		--build-arg https_proxy=${HTTPS_PROXY} \
		--build-arg http_proxy=${HTTP_PROXY} \
		--build-arg _USER=${_USER}  \
		--build-arg _HOME=${CHOME}  \
		--build-arg _UID=${_UID} \
		--build-arg _GID=${_GID} \
		--build-arg PW=${PW} \
		-t ${PRIVATE_IMAGE} .;
	if [ ! -z "${PRIVATE_REGISTRY}" ]; then \
		docker push ${PRIVATE_IMAGE}; \
	fi;
	- tfinished building private image: ${PRIVATE_IMAGE}

edit_public: _pull_public
	$(eval EDIT_CONTRAINER_NAME=${CONTRAINER_NAME}_edit_public)
	xhost +local:docker; \
	docker run -it ${USE_GPUS} \
		--network host \
		--ipc host \
		--env DISPLAY \
		--device /dev/dri \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v ${SHARE_DIR}:${TO_SHARE_DIR} \
		-v ${PROJ_DIR}:${TO_PROJ_DIR} \
		--name ${EDIT_CONTRAINER_NAME} ${PUBLIC_IMAGE} /bin/bash
	docker commit ${EDIT_CONTRAINER_NAME} ${PUBLIC_IMAGE}
	docker push ${PUBLIC_IMAGE}
	docker container rm ${EDIT_CONTRAINER_NAME}

inspect_public: _pull_public
	$(eval CONTRAINER_NAME=${CONTRAINER_NAME}_inspect_public)
	xhost +local:docker; \
	docker run -it --rm ${USE_GPUS} \
		--network host \
		--ipc host \
		--env DISPLAY \
		--device /dev/dri \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v ${SHARE_DIR}:${TO_SHARE_DIR} \
		-v ${PROJ_DIR}:${TO_PROJ_DIR} \
		--name ${CONTRAINER_NAME} ${PUBLIC_IMAGE} /bin/bash

	
build_public:
	#TODO uncomment DOCKER_BUILDKIT=1 docker build --pull --target public --build-arg BASE_IMAGE=${BASE_IMAGE} \
	DOCKER_BUILDKIT=1 docker build --target public --build-arg BASE_IMAGE=${BASE_IMAGE} \
	--build-arg https_proxy=${HTTPS_PROXY} \
	--build-arg http_proxy=${HTTP_PROXY} \
	-t ${PUBLIC_IMAGE} .
	docker push ${PUBLIC_IMAGE}
	- tfinished building public image: ${PUBLIC_IMAGE}

_exec:
	docker exec -it ${CONTRAINER_NAME} bash

_exec_root:
	docker exec -itu 0 ${CONTRAINER_NAME} bash

_instantiate_container: 
	touch ${HOME}/.netrc
	touch ${HOME}/.bash_history
	if [ ! -d ${SHARE_DIR} ]; then \
		mkdir -p ${SHARE_DIR};fi 
	if [ ! -d ${HOME}/.cache ]; then \
		mkdir -p ${HOME}/.cache;fi 
	if [ ! -d ${PROJ_DIR} ]; then \
		mkdir -p ${PROJ_DIR};fi 
	if [ ! -d ${HOME}/vscode/vscode-server ]; then \
		mkdir -p ${HOME}/vscode/vscode-server;fi 
	if [ ! -d ${HOME}/vscode/vscode-remote-containers ]; then \
		mkdir -p ${HOME}/vscode/vscode-remote-containers; fi 
	if ! docker container ls | grep -w ${CONTRAINER_NAME}; then \
		xhost +local:docker; \
		docker run -itd --rm --shm-size=120g \
			--network host \
			--ipc host \
    		--env DISPLAY \
			--device /dev/dri \
    		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v ${SHARE_DIR}:${TO_SHARE_DIR} \
			-v ${HOME}/.cache:${CHOME}/.cache \
			-v ${HOME}/.netrc:${CHOME}/.netrc \
			-v ${HOME}/.bash_history:${CHOME}/.bash_history \
			-v ${PROJ_DIR}:${TO_PROJ_DIR} \
			-v ${HOME}/vscode/vscode-server:${CHOME}/.vscode-server \
			-v ${HOME}/vscode/vscode-remote-containers:${CHOME}/.vscode-remote-containers \
			--user ${_UID}:${_GID} \
			--name ${CONTRAINER_NAME} ${USE_GPUS} $(PRIVATE_IMAGE); \
	fi; \

_pull:
	if ! docker pull ${PRIVATE_IMAGE}; then \
		echo "pull ${PRIVATE_IMAGE} failed, try building private image first"; \
		${MAKE} -f ${MAKEFILE_} --no-print-directory build_private; \
	fi

_pull_public:
	if ! docker pull ${PUBLIC_IMAGE}; then \
		echo "pull ${PUBLIC_IMAGE} failed, try building public image first"; \
	fi

# attach is dangerous, exit from attached shell will terminate the container
_attach:
	docker attach ${CONTRAINER_NAME}


_setup_sshkey:
	touch ../.gitignore
	if [ -z "$$(cat ../.gitignore| grep ssh/)" ]; then echo "ssh/" >> ../.gitignore; fi 
	if [ ! -e ~/.ssh/id_rsa ] || [ ! -e ~/.ssh/id_rsa.pub ]; then \
		echo "sshkey don't exist, try generating it"; \
		ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N ""; \
		echo new sshkey generated; \
	fi; 
	mkdir -p asset/ssh; 
	cp ~/.ssh/id_rsa asset/ssh/id_rsa; 
	cp ~/.ssh/id_rsa.pub asset/ssh/id_rsa.pub; 
	printf "Host github.com\nHostname ssh.github.com\nPort 443\n" > asset/ssh/config
	echo To grant access to your github account, make sure to copy the content in [~/.ssh/id_rsa.pub] to github:;
	cat ~/.ssh/id_rsa.pub; 


_setup_gitconfig:
	touch ../.gitignore
	if [ -z "$$(cat ../.gitignore| grep gitconfig)" ]; then echo "gitconfig" >> ../.gitignore; fi 
	if [ ! -e ~/.gitconfig ]; then \
		echo "gitconfig don't exist, try configure it"; \
		read -p "(no space and special character allowed) Enter github username: " GIT_NAME; \
		read -p "(must be your real github account) Enter github email: " GIT_EMAIL; \
		git config --global user.name $${GIT_NAME:-${PROJECT_NAME}}; \
		git config --global user.email $${GIT_EMAIL:-${PROJECT_NAME}@gmail.com}; \
	fi;
	cp ~/.gitconfig asset/gitconfig; 
	

_check_image:
	if ! docker image ls | grep -w ${PRIVATE_IMAGE}; then \
		echo "${PRIVATE_IMAGE} image not found, try building private image first"; \
		${MAKE} -f ${MAKEFILE_} --no-print-directory build_private; \
	fi
	
