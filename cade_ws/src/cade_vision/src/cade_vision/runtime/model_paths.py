"""Model path and loading helpers."""

from pathlib import Path


def package_model_path(filename: str) -> str:
    import cade_vision

    package_dirs = [
        Path(path).resolve()
        for path in reversed(list(getattr(cade_vision, "__path__", [])))
    ]
    package_file = getattr(cade_vision, "__file__", None)
    if package_file:
        package_dirs.append(Path(package_file).resolve().parent)

    for package_dir in package_dirs:
        candidate = package_dir / "models" / filename
        if candidate.exists():
            return str(candidate)

    if package_dirs:
        return str(package_dirs[0] / "models" / filename)
    return filename


def preferred_model_path(filename: str) -> str:
    """Prefer a TensorRT engine next to the packaged .pt when available."""
    pt_path = Path(package_model_path(filename))
    if pt_path.suffix == ".pt":
        engine_path = pt_path.with_suffix(".engine")
        if engine_path.exists():
            return str(engine_path)
    return str(pt_path)


def load_yolo_model(model_path: str, device: str, label: str):
    from ultralytics import YOLO

    print(f"Loading {label} model on {device}: {model_path}")
    model = YOLO(model_path)
    if Path(str(model_path)).suffix.lower() != ".engine":
        model.to(device)
    else:
        print(f"{label} TensorRT engine loaded; device is selected by engine runtime")
    print(f"{label} model loaded")
    return model

