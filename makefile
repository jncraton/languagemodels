all: test

test: lint
	python3 -m doctest languagemodels/*.py

lint:
	flake8 --max-line-length 88 languagemodels/*.py

format:
	black languagemodels/*.py

doc:
	mkdir -p doc
	python3 -m pdoc -o doc languagemodels

paper.pdf: paper.md paper.bib
	pandoc $< --bibliography=paper.bib --citeproc -o $@

upload:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

clean:
	rm -rf tmp
	rm -rf languagemodels.egg-info
	rm -rf languagemodels/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf doc
