# 打包发布流程

```cpp
./.venv/Scripts/activate.bat

pyinstaller --noconfirm --onefile main.py ^
  --name=hand_detector ^
  --add-data "resources;mediapipe"
```