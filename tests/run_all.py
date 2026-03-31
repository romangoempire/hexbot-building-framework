#!/usr/bin/env python3
"""Discover and run all hex_bot tests. Exit 0 if all pass, 1 otherwise."""

import os
import sys
import unittest

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print('\n' + '=' * 60)
    run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = run - failures - errors - skipped
    print(f'Ran {run} tests: {passed} passed, {failures} failed, '
          f'{errors} errors, {skipped} skipped')
    print('=' * 60)

    if result.wasSuccessful():
        print('ALL TESTS PASSED')
        return 0
    else:
        print('SOME TESTS FAILED')
        return 1


if __name__ == '__main__':
    sys.exit(main())
