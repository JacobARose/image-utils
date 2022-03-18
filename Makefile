.ONESHELL:

SHELL := /bin/bash
DATE_ID := $(shell date +"%y.%m.%d")
# Get package name from pwd
PACKAGE_NAME := $(shell basename $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
PACKAGE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
IMPORT_NAME := "imutils"

.DEFAULT_GOAL := help

DOCS_DIR = $(PACKAGE_DIR)/docs

PORT?=9898 ## Port at which to host local docs server. Used default value PORT=9898 if not overriden by user.


define PRINT_HELP_PYSCRIPT
import re, sys

class Style:
    BLACK = '\033[30m'
    BLUE = '\033[34m'
    BOLD = '\033[1m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    ENDC = '\033[0m'

print(f"{Style.BOLD}Please use `make <target>` where <target> is one of{Style.ENDC}")
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if line.startswith("# -------"):
		print(f"\n{Style.RED}{line}{Style.ENDC}")
	if match:
		target, help_msg = match.groups()
		if not target.startswith('--'):
			print(f"{Style.BOLD+Style.GREEN}{target:20}{Style.ENDC} - {help_msg}")
endef

export PRINT_HELP_PYSCRIPT


# If you want a specific Python interpreter define it as an envvar
# $ export PYTHON_ENV=
ifdef PYTHON_ENV
	PYTHON := $(PYTHON_ENV)
else
	PYTHON := python3
endif


#################################### Functions ###########################################
# Function to check if package is installed else install it.
define install-pkg-if-not-exist
	@for pkg in ${2} ${3}; do \
		if ! command -v "$${pkg}" >/dev/null 2>&1; then \
			echo "installing $${pkg}"; \
			$(PYTHON) -m pip install $${pkg}; \
		fi;\
	done
endef

# Function to create python virtualenv if it doesn't exist
define create-venv
	$(call install-pkg-if-not-exist,virtualenv)

	@if [ ! -d ".$(PACKAGE_NAME)_venv" ]; then \
		$(PYTHON) -m virtualenv ".$(PACKAGE_NAME)_venv" -p $(PYTHON) -q; \
		echo "\".$(PACKAGE_NAME)_venv\": Created successfully!"; \
	fi;
	@echo "Source virtual environment before tinkering"
	@echo "Manually run: \`source .$(PACKAGE_NAME)_venv/bin/activate\`"
endef


help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
	echo "IMPORT_NAME =" $(IMPORT_NAME)
	echo "PACKAGE_NAME =" $(PACKAGE_NAME)
	echo "PACKAGE_DIR =" $(PACKAGE_DIR)
	echo "DOCS_DIR =" $(DOCS_DIR)



# -------------------------------- Builds and Installations -----------------------------
# TODO: Refactor venv-based functions to use conda instead.

dev-venv: venv ## Install the package in development mode including all dependencies inside a virtualenv (container).
	@$(PYTHON_VENV) -m pip install .[dev];
	echo -e "\n--------------------------------------------------------------------"
	echo -e "Usage:\nPlease run:\n\tsource .$(PACKAGE_NAME)_venv/bin/activate;"
	echo -e "\t$(PYTHON) -m pip install .[dev];"
	echo -e "Start developing..."

install: clean ## Check if package exist, if not install the package
	@$(PYTHON) -c "import $(PACKAGE_NAME)" >/dev/null 2>&1 || $(PYTHON) -m pip install .;

venv:  ## Create virtualenv environment on local directory.
	@$(create-venv)


# -------------------------------------- Clean Up  --------------------------------------
.PHONY: clean
# clean: clean-build clean-docs clean-pyc clean-test clean-docker ## Remove all build, test, coverage and Python artefacts
clean: clean-docs clean-pyc

#clean-build: ## Remove build artefacts
#	rm -fr build/
#	rm -fr dist/
#	rm -fr .eggs/
#	find . -name '*.egg-info' -exec rm -fr {} +
#	find . -name '*.egg' -exec rm -fr {} +
#	find . -name '*.xml' -exec rm -fr {} +

clean-docs: ## Remove all files in ${DOCS_DIR}
	rm -rf $(DOCS_DIR)

#clean-docs: ## Remove docs/_build artefacts, except PDF and singlehtml
#	# Do not delete <module>.pdf and singlehtml files ever, but can be overwritten.
#	find docs/compiled_docs ! -name "$(PACKAGE_NAME).pdf" ! -name 'index.html' -type f -exec rm -rf {} +
#	rm -rf docs/compiled_docs/doctrees
#	rm -rf docs/compiled_docs/html
#	rm -rf $(DOCS_DIR)/modules.rst
#	rm -rf $(DOCS_DIR)/$(PACKAGE_NAME)*.rst
#	rm -rf $(DOCS_DIR)/README.md

clean-pyc: ## Remove Python file artefacts
	echo "Currently cleaning from root: $(PWD)"
	find . -name '*.ipynb_checkpoints' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	#find . -name '__pycache__' -exec rm -fr {}


# ---------------------------------------- Tests -----------------------------------------
test: ## Run tests quickly with pytest
	$(PYTHON) -m pytest -sv
	# $(PYTHON) -m nose -sv

# ---------------------------- Documentation Generation ----------------------
.PHONY: --docs-depencencies

--docs-depencencies: ## Installs pdoc3 if it's not installed already
	$(call install-pkg-if-not-exist,pdoc3)

docs: --docs-depencencies clean ## Generate nested html api documentation using pdoc3
#	pdoc3 --html --output-dir $(DOCS_DIR) $(PACKAGE_NAME)
	pdoc3 --html --output-dir $(DOCS_DIR) $(IMPORT_NAME)


docs-server: docs ## Start up a local server using pdoc3
	pdoc3 --http localhost:$(PORT) $(IMPORT_NAME)


