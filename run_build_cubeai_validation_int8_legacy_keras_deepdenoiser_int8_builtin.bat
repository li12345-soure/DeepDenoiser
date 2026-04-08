@echo off
setlocal

cd /d "%~dp0"

set TF_USE_LEGACY_KERAS=1
set PYTHON_EXE=.\deepdenoiser_tflite_env\Scripts\python.exe
set EXPORT_SCRIPT=.\export_tf_raw_debug.py
set CHECKPOINT_DIR=.\model\190614-104802
set INPUT_GLOB=.\Dataset\pred\*.npz
set OUT_DIR=.\cubeai_validation
set INT8_MODEL=.\deepdenoiser_int8_builtin.tflite

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%EXPORT_SCRIPT%" (
  echo [ERROR] Script not found: %EXPORT_SCRIPT%
  exit /b 1
)

if not exist "%INT8_MODEL%" (
  echo [ERROR] INT8 model not found: %INT8_MODEL%
  echo [INFO] If your file has a different name, edit this line in the bat file:
  echo        set INT8_MODEL=.\deepdenoiser_int8_builtin.tflite
  exit /b 1
)

"%PYTHON_EXE%" .\build_cubeai_validation_set_direct.py ^
  --python_exe "%PYTHON_EXE%" ^
  --export_script "%EXPORT_SCRIPT%" ^
  --checkpoint_dir "%CHECKPOINT_DIR%" ^
  --input_glob "%INPUT_GLOB%" ^
  --out_dir "%OUT_DIR%" ^
  --tflite_model "%INT8_MODEL%" %*

endlocal
