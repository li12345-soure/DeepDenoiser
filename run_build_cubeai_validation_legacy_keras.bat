@echo off
setlocal

REM Force TensorFlow 2.16+ to use legacy Keras 2 if tf_keras is installed
set TF_USE_LEGACY_KERAS=1

REM Optional: make logs quieter / deterministic-ish
set TF_CPP_MIN_LOG_LEVEL=1

if "%~1"=="" (
    .\deepdenoiser_tflite_env\Scripts\python.exe .\build_cubeai_validation_set_direct.py ^
      --python_exe .\deepdenoiser_tflite_env\Scripts\python.exe ^
      --export_script .\export_tf_raw_debug.py ^
      --checkpoint_dir .\model\190614-104802 ^
      --input_glob ".\Dataset\pred\*.npz" ^
      --out_dir .\cubeai_validation
) else (
    .\deepdenoiser_tflite_env\Scripts\python.exe .\build_cubeai_validation_set_direct.py ^
      --python_exe .\deepdenoiser_tflite_env\Scripts\python.exe ^
      --export_script .\export_tf_raw_debug.py ^
      --checkpoint_dir .\model\190614-104802 ^
      --input_glob ".\Dataset\pred\*.npz" ^
      --out_dir .\cubeai_validation %*
)

endlocal
