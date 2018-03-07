.PHONY: run
run: setup
	python ./transfer_learning_live_song_id.py

.PHONY: setup
setup: kapre
	pip install h5py keras tensorflow

.PHONY: kapre
kapre:
	-cd ..; git clone https://github.com/evhub/kapre; cd transfer-learning-live-song-id
	pip install -e ../kapre

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
