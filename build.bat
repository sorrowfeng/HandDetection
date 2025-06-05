@echo off
setlocal

:: ������Ŀ��Ŀ¼
set "PROJECT_DIR=%~dp0"

echo ��ʼ���Ӧ�ó���...
echo ��ĿĿ¼: %PROJECT_DIR%

:: �������⻷��
echo ���ڼ������⻷��...
call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ����: �޷��������⻷����
    goto :error
)

:: ���PyInstaller�Ƿ�װ
echo ���ڼ��PyInstaller...
pyinstaller --version
if errorlevel 1 (
    echo ����: PyInstallerδ��װ��
    echo ������: pip install pyinstaller
    goto :error
)

:: ִ�д������
echo ����ִ�д��...
pyinstaller --noconfirm --onefile "%PROJECT_DIR%\main.py" ^
  --name=hand_detector ^
  --add-data "%PROJECT_DIR%\resources;mediapipe"

if errorlevel 1 (
    echo ����: ��������з�������
    goto :error
)

:: �������ɹ���Ϣ
echo.
echo ==================================================
echo ����ɹ���
echo ��ִ���ļ�λ��: %PROJECT_DIR%\dist\hand_detector.exe
echo ==================================================
echo.

:: ������ʱ�ļ�
echo ����������ʱ�ļ�...
rmdir /S /Q "%PROJECT_DIR%\build"
del /F /Q "%PROJECT_DIR%\hand_detector.spec"

echo �����������ɣ�

goto :end

:error
echo �������ʧ�ܣ�
exit /b 1

:end
endlocal