"""Shared test utilities."""

from __future__ import annotations

_RULE = "=" * 78


def header(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def report(test: str, **kwargs) -> None:
    """Print a clearly formatted summary of a test's configuration/result."""
    header(test)
    for key, value in kwargs.items():
        print(f"  {key:<22} {value}")
    print(_RULE)


def run_all_tests(module_globals: dict) -> None:
    """Run every test_*() function in a module directly, no pytest needed.

    Call as ``run_all_tests(globals())`` from a file's
    ``if __name__ == "__main__":`` block, so ``python test_whatever.py``
    runs the whole file and prints every test's diagnostics unconditionally
    (pytest normally captures stdout and only shows it on failure).
    """
    test_fns = {
        name: obj
        for name, obj in module_globals.items()
        if name.startswith("test_") and callable(obj)
    }
    for fn in test_fns.values():
        fn()
    print(f"\n{len(test_fns)}/{len(test_fns)} tests passed.\n")
