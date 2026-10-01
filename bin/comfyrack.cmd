@echo off
REM Runs the plugin's OWN bundled copy.
set "PLUGIN_ROOT=%~dp0.."
if "%COMFYRACK_PYTHON%"=="" set "COMFYRACK_PYTHON=python"
"%COMFYRACK_PYTHON%" -c "import sys; sys.path.insert(0, r'%PLUGIN_ROOT%\src'); from comfyrack.cli.main import main; sys.exit(main())" %*
