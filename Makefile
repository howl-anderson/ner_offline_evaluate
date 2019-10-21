.PHONY: local_build_docker_image
local_build_docker_image:
	docker build -t offline_ner_evaluate:latest -f docker_v2/nightly/Dockerfile .

.PHONY: test_build_docker_image
test_build_docker_image:
	docker run -v $(shell pwd)/data:/data -v $(shell pwd)/model:/model -v $(shell pwd)/output:/output offline_ner_evaluate:latest

.PHONY: debug_build_docker_image
debug_build_docker_image:
	docker run -it -v $(shell pwd)/data:/data -v $(shell pwd)/model:/model -v $(shell pwd)/output:/output offline_ner_evaluate:latest /bin/bash