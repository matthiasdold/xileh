clean:
	rm -rf ./build ./src/xileh.egg-info
	find . -name __pycache__ | xargs rm -rf
	find . -name *.pyc | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf

install:
	pip install .

uninstall:
	pip uninstall xileh

test:
	pytest .
