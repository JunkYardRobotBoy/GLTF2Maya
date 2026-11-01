@echo off
rem Installs GLTF2Maya for Maya 2026 / 2027:
rem   - GLTF2Maya.py   -> Documents\maya\<ver>\plug-ins\     (the plug-in)
rem   - GLTF2Maya.py   -> Documents\maya\scripts\            (so "import GLTF2Maya" works in Python)
rem   - icons\*.png    -> Documents\maya\<ver>\prefs\icons\  (shelf icon)
rem The plug-in adds an Import button to the glTF shelf (shared with Maya2glTF) when it loads.
setlocal
set SRC=%~dp0
set MAYADIR=%USERPROFILE%\Documents\maya
if not exist "%MAYADIR%" set MAYADIR=%USERPROFILE%\OneDrive\Documents\maya
if not exist "%MAYADIR%" (
    echo Could not find Documents\maya. Start Maya once, then run this again.
    pause & exit /b 1
)
if not exist "%MAYADIR%\scripts" mkdir "%MAYADIR%\scripts"
copy /Y "%SRC%GLTF2Maya.py" "%MAYADIR%\scripts\" >nul && echo Installed to %MAYADIR%\scripts
for %%V in (2026 2027) do (
    if exist "%MAYADIR%\%%V" (
        if not exist "%MAYADIR%\%%V\plug-ins" mkdir "%MAYADIR%\%%V\plug-ins"
        copy /Y "%SRC%GLTF2Maya.py" "%MAYADIR%\%%V\plug-ins\" >nul && echo Installed to %MAYADIR%\%%V\plug-ins
        if not exist "%MAYADIR%\%%V\prefs\icons" mkdir "%MAYADIR%\%%V\prefs\icons"
        copy /Y "%SRC%icons\GLTF2Maya*.png" "%MAYADIR%\%%V\prefs\icons\" >nul && echo Installed icons to %MAYADIR%\%%V\prefs\icons
    )
)
echo.
echo In Maya: Plug-in Manager ^> GLTF2Maya.py ^> Loaded + Auto load.
echo An Import button appears on the glTF shelf. Script Editor (Python tab):
echo     import GLTF2Maya; GLTF2Maya.show_ui()
pause
