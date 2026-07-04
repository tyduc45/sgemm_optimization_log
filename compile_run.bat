if not exist build md build 
cd build 
cmake .. 
cmake --build .
.\Debug\morden_cuda.exe
