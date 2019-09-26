.DEFAULT_GOAL := help

download:
	@rsync -v -r -e ssh \
		--exclude out/ \
		--exclude data/ \
		--exclude __pycache__/ \
		nct01011@dt01.bsc.es:/home/nct01/nct01011/dogs-classification/* .

upload:
	@rsync -v -r -e ssh \
		--exclude __pycache__/ \
		--exclude .git/ \
		./ nct01011@dt01.bsc.es:/home/nct01/nct01011/dogs-classification

queue-task:
	@ssh nct01011@mt1.bsc.es 'cd /home/nct01/nct01011/dogs-classification && ./launchers/launch.sh train'

debug-task:
	@ssh nct01011@mt1.bsc.es 'cd /home/nct01/nct01011/dogs-classification && ./launchers/launch.sh debug'

help:
	@echo "run <make [download|upload]> to move files from/to server"