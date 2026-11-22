@echo off
rem QuantLab: get the newest version from GitHub, then start the app. Double-click this file.
setlocal
cd /d "%~dp0"
echo Getting the newest version ...
git pull --ff-only
if errorlevel 1 goto failed
call "%~dp0start.bat" %*
exit /b

:failed
echo.
echo Update failed. If it says your local changes would be overwritten, you edited a file
echo that was also updated: run "git status" to see which, or ask for help.
echo The app was not started.
pause
exit /b 1
