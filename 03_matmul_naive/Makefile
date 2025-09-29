PYTHON = python

TEST_SCRIPT = src/test.py

.PHONY: all build run test clean


all: build test

build:
	@echo "--- Building C++/CUDA extension ---"
	$(PYTHON) setup.py build_ext --inplace


run:
	@echo "--- Running test script ---"
	$(PYTHON) $(TEST_SCRIPT)

test: build
	@echo "--- Running test script ---"
	$(PYTHON) $(TEST_SCRIPT)

clean:
	@echo "--- Cleaning up build artifacts ---"
	-rmdir /s /q build
	-del *.pyd
	@echo "Cleanup complete."