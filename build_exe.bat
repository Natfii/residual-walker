@echo off
rem Build the portable launcher: dist\ResidualWalker.exe (~10 MB, stdlib only).
rem Requires the app venv to exist (run launcher.py or setup once first).
cd /d "%~dp0"
.venv\Scripts\pip install --quiet pyinstaller
.venv\Scripts\pyinstaller --onefile --console --name ResidualWalker ^
  --distpath dist --workpath build --specpath build launcher.py
echo.
echo Built: dist\ResidualWalker.exe
