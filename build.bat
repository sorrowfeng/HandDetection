@echo off
setlocal

:: 定义项目根目录
set "PROJECT_DIR=%~dp0"

echo 开始打包应用程序...
echo 项目目录: %PROJECT_DIR%

:: 激活虚拟环境
echo 正在激活虚拟环境...
call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo 错误: 无法激活虚拟环境！
    goto :error
)

:: 检查PyInstaller是否安装
echo 正在检查PyInstaller...
pyinstaller --version
if errorlevel 1 (
    echo 错误: PyInstaller未安装！
    echo 请运行: pip install pyinstaller
    goto :error
)

:: 执行打包命令
echo 正在执行打包...
pyinstaller --noconfirm --onefile "%PROJECT_DIR%\main.py" ^
  --name=hand_detector ^
  --add-data "%PROJECT_DIR%\resources;mediapipe"

if errorlevel 1 (
    echo 错误: 打包过程中发生错误！
    goto :error
)

:: 输出打包成功信息
echo.
echo ==================================================
echo 打包成功！
echo 可执行文件位于: %PROJECT_DIR%\dist\hand_detector.exe
echo ==================================================
echo.

:: 清理临时文件
echo 正在清理临时文件...
rmdir /S /Q "%PROJECT_DIR%\build"
del /F /Q "%PROJECT_DIR%\hand_detector.spec"

echo 打包过程已完成！

goto :end

:error
echo 打包过程失败！
exit /b 1

:end
endlocal