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
env : conda $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt
	@$(ACTIVATE) ; $(INSTALL)
conda :
ifeq "$(shell $@ info --envs | grep $(PACKAGE) | wc -l)" "0"
	@$@ create -yn $(PACKAGE) $(EXEC)
endif

## test      : run testing pipeline.
.PHONY : test
test : pylint mypy
pylint : env html/pylint/index.html
mypy : env html/mypy/index.html
html/pylint/index.html : html/pylint/index.json
	@$(ACTIVATE) ; pylint-json2html -o $@ -e utf-8 $<
html/pylint/index.json : $(PACKAGE)/*.py
	@mkdir -p $(@D)
	@$(ACTIVATE) ; pylint $(PACKAGE) \
	--disable C0114,C0115,C0116,R0903,R0913,W0235,W1203 \
	--generated-members torch.* \
	--output-format=colorized,json:$@ \
	|| pylint-exit $$?
html/mypy/index.html : $(PACKAGE)/*.py
	@$(ACTIVATE) ; mypy \
	-p $(PACKAGE) \
	--html-report $(@D)
