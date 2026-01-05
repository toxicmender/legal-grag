# Test Fixtures Directory

This directory contains test data files used by pytest fixtures.

## Structure

- Sample PDF files for testing PDF reading functionality
- Mock data files for various test scenarios
- Reference data for integration tests

## Usage

Fixtures are defined in `tests/conftest.py` and can reference files in this directory.

## Note

For actual PDF testing, place sample PDF files here. The fixtures in `conftest.py` 
use temporary files by default to avoid requiring actual PDF files in the repository.

