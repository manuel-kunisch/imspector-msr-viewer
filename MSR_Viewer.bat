@echo off
rem MSR Viewer launcher - runs msr_viewer.py in a conda environment with PyQt5, numpy and tifffile.
rem Double-click to start, or drag .msr files / folders onto this file.
rem The environment is MSR_VIEWER_ENV if set, else the first of the folders below that exists.
setlocal
set "ENV=%MSR_VIEWER_ENV%"
if defined ENV if exist "%ENV%\pythonw.exe" goto found
for %%E in (
  "%USERPROFILE%\.conda\envs\msr"
  "%USERPROFILE%\anaconda3\envs\msr"
  "%USERPROFILE%\miniconda3\envs\msr"
  "%USERPROFILE%\miniforge3\envs\msr"
  "%ProgramData%\anaconda3\envs\msr"
  "%ProgramData%\miniconda3\envs\msr"
  "%USERPROFILE%\anaconda3\envs\py312"
) do if exist "%%~E\pythonw.exe" (
  set "ENV=%%~E"
  goto found
)
echo No conda environment "msr" found (looked in %USERPROFILE%\.conda\envs, anaconda3, miniconda3, ...).
echo Create one:  conda create -n msr python=3.12 numpy tifffile pyqt
echo or set MSR_VIEWER_ENV to the folder of an environment with PyQt5, numpy and tifffile.
pause
exit /b 1
:found
set "PATH=%ENV%;%ENV%\Library\mingw-w64\bin;%ENV%\Library\usr\bin;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"
start "" "%ENV%\pythonw.exe" -B "%~dp0msr_viewer.py" %*
