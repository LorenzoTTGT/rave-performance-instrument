# RAVE repository workflow

Read [docs/BUILDING.md](docs/BUILDING.md) for native build, staging, versioning and installation procedures.

- `VERSION` owns the delivered version. Preserve plugin codes and bundle identifiers.
- Use `build.sh` / `build.ps1`; run tests before staging. Build outputs and staged packages remain ignored.
- Host installation is explicit, from a verified staged package only. Do not copy raw build outputs into plugin directories or remove unrelated plugins.
- Use repository-pinned formatting tools. Prettier enforcement covers changed supported files; keep unrelated formatting changes separate. C++ uses `.clang-format`, Python Black, and CMake cmake-format.
- Validate packaging changes with relocated loading and disposable installation, including replacement/rollback checks. Report untested architectures and reduced checks accurately.
