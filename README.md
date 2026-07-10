# sgemm_optimization_log

项目结构：

- `include/`：类型定义和函数声明；
- `src/`：CUDA kernel、GPU/CPU 端函数实现及程序入口。

在 Visual Studio Developer PowerShell 中构建并运行：

```powershell
cmake -S . -B build
cmake --build build --config Release --parallel
.\build\Release\modern_cuda.exe
```
