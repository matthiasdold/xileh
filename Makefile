clean:
	rm -rf ./build ./src/xileh.egg-info .pytest_cache
	find . -name __pycache__ | xargs rm -rf
	find . -name *.pyc | xargs rm -rf

install:
	pip install .

uninstall:
	pip uninstall xileh

test:
	pytest .
