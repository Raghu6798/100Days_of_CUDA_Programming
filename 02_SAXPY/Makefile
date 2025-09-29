PYTHON = python

# Define the name of your test script.
TEST_SCRIPT = src/test.py

# ==============================================================================
#  Main Commands
# ==============================================================================

# Phony targets are commands that don't represent a file.
# This is good practice to prevent conflicts with actual files named "build", "test", etc.
.PHONY: all build run test clean

# The default command when you just type "make".
# It will first run the 'build' command, then the 'test' command.
all: build test

# Build the C++/CUDA extension in-place.
build:
	@echo "--- Building C++/CUDA extension ---"
	$(PYTHON) setup.py build_ext --inplace

# Run the test script.
# This is a simple target for just running the script without rebuilding.
run:
	@echo "--- Running test script ---"
	$(PYTHON) $(TEST_SCRIPT)

# A combination target that first cleans, then builds, then runs the test.
# This is the most common command you will use for a full verification.
test: build
	@echo "--- Running test script ---"
	$(PYTHON) $(TEST_SCRIPT)

# Clean up all build artifacts and compiled files.
# This is essential for ensuring a clean build from scratch.
clean:
	@echo "--- Cleaning up build artifacts ---"
	-rmdir /s /q build
	-del *.pyd
	@echo "Cleanup complete."