IMAGE_NAME=bluebook-bulldozers

test:
	python -m pytest -v

docker-test: build
	docker run -t $(IMAGE_NAME) "python -m pytest -v"

fetch-data:
	# some useful command to pull training data

build:
	docker build -t $(IMAGE_NAME) .

shell: build
	docker run -it $(IMAGE_NAME) /bin/bash

train: build
	docker run $(IMAGE_NAME) "some training command"

eval: build
	docker run $(IMAGE_NAME) "some inference command"
