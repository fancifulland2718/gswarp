# Contributing to gswarp

Thanks for your interest in contributing! Here are a few guidelines.

## Reporting Issues

- Search existing issues before opening a new one.
- Include your OS, Python version, PyTorch version, Warp version, and GPU model.
- Provide a minimal reproducible example when possible.

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused — one feature or fix per PR.
3. Add tests if you are changing kernel logic or public API behavior.
4. Make sure the existing test suite still passes.
5. Follow the existing code style (no auto-formatter enforced, but keep it consistent).

## Code Style

- Python 3.10+ syntax (union types with `|`, etc.).
- Type annotations for public function signatures.
- Warp kernels and `@wp.func` helpers stay in `_warp_backend.py` (single-module requirement).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).
