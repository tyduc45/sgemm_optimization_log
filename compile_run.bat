@echo off
cmake -S . -B build
if errorlevel 1 exit /b %errorlevel%
cmake --build build --config Release --parallel
if errorlevel 1 exit /b %errorlevel%
.\build\Release\modern_cuda.exe
