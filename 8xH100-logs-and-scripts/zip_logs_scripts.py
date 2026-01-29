import zipfile
from pathlib import Path

zip_name = "logs-and-scripts.zip"

logs_dir = Path("logs")
py_files = Path(".").glob("*.py")

with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_STORED) as z:
    # Add logs/ recursively
    if logs_dir.exists():
        for path in logs_dir.rglob("*"):
            if path.is_file():
                z.write(path, arcname=path)
    else:
        print("Warning: logs/ directory not found")

    # Add *.py files from current directory
    for path in py_files:
        if path.is_file():
            z.write(path, arcname=path.name)

print(f"Created {zip_name}")
