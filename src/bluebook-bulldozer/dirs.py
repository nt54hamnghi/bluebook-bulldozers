import pathlib

ROOT_DIR = pathlib.Path(__file__).parents[2]
CONTENT_DIR = ROOT_DIR / "src/bluebook-bulldozer/stages/content"
DATA_DIR = ROOT_DIR / "resrc/data"
LOG_DIR = ROOT_DIR / "log"

print(ROOT_DIR)
