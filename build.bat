@echo off
rem ============================================================
rem  Local AI Installer - 打包脚本 (Windows)
rem  生成 dist\LocalAI-Installer.exe (单文件, 免 Python 环境)
rem  及 dist\mini-agent.exe (自带轻量聊天 Agent)
rem ============================================================
chcp 65001 >nul
cd /d %~dp0

set PY=python
where python >nul 2>&1 || set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

echo [1/3] 检查 PyInstaller...
%PY% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo 未安装, 正在安装 PyInstaller...
    %PY% -m pip install pyinstaller
)

echo [2/3] 清理旧构建...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/3] 打包...
echo  -- 打包 mini-agent.exe ...
%PY% -m PyInstaller --noconfirm --clean ^
    --onefile --console --name "mini-agent" ^
    mini_agent.py
if errorlevel 1 goto :fail

echo  -- 打包 LocalAI-Installer.exe ...
%PY% -m PyInstaller --noconfirm --clean ^
    --onefile --console ^
    --name "LocalAI-Installer" ^
    --add-data "mini_agent.py;." ^
    --add-data "dist\mini-agent.exe;." ^
    installer.py
if errorlevel 1 goto :fail

echo.
echo 打包完成:
echo   dist\LocalAI-Installer.exe   (安装器)
echo   dist\mini-agent.exe          (自带聊天 Agent)
echo 使用方法: 双击 LocalAI-Installer.exe 运行, 或加 --help 查看参数
pause
exit /b 0

:fail
echo 打包失败!
pause
exit /b 1
