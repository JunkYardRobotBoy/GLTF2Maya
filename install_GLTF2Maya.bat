@echo off
rem Installs GLTF2Maya for Maya 2026 / 2027:
rem   - GLTF2Maya.py  -> Documents\maya\<ver>\plug-ins\   (the plug-in)
rem   - GLTF2Maya.py  -> Documents\maya\scripts\          (so "import GLTF2Maya" works in Python)
setlocal
set SRC=%~dp0GLTF2Maya.py
set MAYADIR=%USERPROFILE%\Documents\maya
if not exist "%MAYADIR%" set MAYADIR=%USERPROFILE%\OneDrive\Documents\maya
if not exist "%MAYADIR%" (
    echo Could not find Documents\maya. Start Maya once, then run this again.
    pause & exit /b 1
)
if not exist "%MAYADIR%\scripts" mkdir "%MAYADIR%\scripts"
copy /Y "%SRC%" "%MAYADIR%\scripts\" >nul && echo Installed to %MAYADIR%\scripts
for %%V in (2026 2027) do (
    if exist "%MAYADIR%\%%V" (
        if not exist "%MAYADIR%\%%V\plug-ins" mkdir "%MAYADIR%\%%V\plug-ins"
        copy /Y "%SRC%" "%MAYADIR%\%%V\plug-ins\" >nul && echo Installed to %MAYADIR%\%%V\plug-ins
    )
)
echo.
echo In Maya: Plug-in Manager ^> GLTF2Maya.py ^> Loaded + Auto load.
echo Python shelf button / Script Editor (Python tab):
echo     import GLTF2Maya; GLTF2Maya.show_ui()
pause
