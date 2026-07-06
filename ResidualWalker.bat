@echo off
rem Residual Walker — for machines that already have Python 3.10+.
rem No Python? Use ResidualWalker.exe instead (or build it: build_exe.bat).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 launcher.py %*
) else (
  python launcher.py %*
)
