SHELL := /usr/bin/env bash
EXEC = python=3.10
PACKAGE = optsent
INSTALL = pip install -e .
ACTIVATE = source activate $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repo with latest version from GitHub.
.PHONY : update
update :
	@git pull origin main

## env       : setup environment and install dependencies.
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt
	@conda create -yn $(PACKAGE) $(EXEC)
	@$(ACTIVATE) ; $(INSTALL)

## test      : run testing pipeline.
.PHONY : test
test : mypy pylint pytest
mypy : env html/mypy/index.html
pylint : env html/pylint/index.html
pytest : env html/coverage/index.html
html/mypy/index.html : $(PACKAGE)/*.py
	@$(ACTIVATE) ; mypy \
	-p $(PACKAGE) \
	--html-report $(@D)
html/pylint/index.html : html/pylint/index.json
	@$(ACTIVATE) ; pylint-json2html -o $@ -e utf-8 $<
html/pylint/index.json : $(PACKAGE)/*.py test/*.py
	@mkdir -p $(@D)
	@$(ACTIVATE) ; pylint $(PACKAGE) test/*.py \
	--disable C0114,C0115,C0116,R0903,R0912,R0913,W0223,W0235,W1203 \
	--generated-members torch.* \
	--output-format=colorized,json:$@ \
	|| pylint-exit $$?
html/coverage/index.html : html/pytest/report.html
	@$(ACTIVATE) ; coverage html -d $(@D)
html/pytest/report.html : $(PACKAGE)/*.py test/*.py
	@$(ACTIVATE) ; coverage run --branch -m pytest \
	--html=$@ --self-contained-html


.PHONY: run
run :
	@$(ACTIVATE) ; python -m optsent $(filter-out $@, $(MAKECMDGOALS))
% : 
	@true
