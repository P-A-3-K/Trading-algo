@echo off
rem QuantLab: one-command start for Windows. Double-click this file.
rem First run: creates a private Python environment in .venv and installs the packages.
rem Later runs: packages are only reinstalled when requirements.txt has changed.
setlocal
cd /d "%~dp0"
title QuantLab

set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto nopython

if exist ".venv\Scripts\python.exe" goto venvready
echo First run: creating a private Python environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 goto nopython
:venvready

fc /b requirements.txt .venv\requirements.installed >nul 2>nul
if not errorlevel 1 goto start
echo Installing packages - first run, or after an update. This takes a minute ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto pipfailed
copy /y requirements.txt .venv\requirements.installed >nul

:start
".venv\Scripts\python.exe" -m quantlab.app %*
if errorlevel 1 pause
exit /b

:nopython
echo.
echo Python was not found, or the .venv folder could not be created.
echo Install Python from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click start.bat again.
pause
exit /b 1

:pipfailed
echo.
echo Installing the packages failed - see the message above.
echo Check your internet connection and try again.
pause
exit /b 1
