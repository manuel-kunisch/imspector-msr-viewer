@echo off
rem MSR Viewer launcher - uses the conda environment "py312".
rem Double-click to start, or drag .msr files / folders onto this file.
set "ENV=%USERPROFILE%\anaconda3\envs\py312"
if not exist "%ENV%\pythonw.exe" (
  echo Conda environment not found: %ENV%
  echo Edit the ENV line in %~nx0 to point to a conda env with PyQt5, numpy and tifffile.
  pause
  exit /b 1
)
set "PATH=%ENV%;%ENV%\Library\mingw-w64\bin;%ENV%\Library\usr\bin;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"
start "" "%ENV%\pythonw.exe" -B "%~dp0msr_viewer.py" %*
