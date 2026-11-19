# Quadruped Robot — 12-DOF 四足机器人

12 自由度四足机器人控制系统：4 腿 × 3 自由度（`hip_roll / hip_pitch / knee`），支持 `trot / walk / bound / stand` 步态。

## 硬件与控制架构

| 层级 | 运行平台 | 核心职责 | 频率 |
|---|---|---|---|
| 上层规划 | Raspberry Pi 5 (Python) | 4 振子耦合 Hopf CPG (RK4)、摆线足端轨迹、全身 3-DOF 解析逆解、串口重传 | 50 Hz 下发 |
| 下层实时控制 | ESP32 (C++17 / Arduino) | MPU6050 I2C 14B 读取、Mahony 四元数融合 + 冲击门控、FSR 消抖、3-DOF 逆解、12 路 16-bit LEDC PWM、100 ms 看门狗缓停 | 100 Hz 控制 / 50 Hz 遥测 |

## 通信协议（UART 115200）

帧格式：`0xAA 0x55 | len | cmd | seq | payload... | CRC16-CCITT (BE) | 0x0D 0x0A`
- `0x01 CMD_JOINTS`：12 × `int16` 关节目标角（`0.01 deg`，50 Hz 流式下发，每 20 ms 一帧，丢帧不重传）
- `0x02 CMD_MODE`：模式切换（`IDLE / STAND / TROT / ESTOP`），带 `seq + CMD_ACK` 确认与超时重传
- `0x03 CMD_ACK`：关键命令应答
- `0x04 CMD_TELEMETRY`：ESP32 50 Hz 回传姿态欧拉角、角速度、4 路 FSR 与控制器状态
- `0x05 CMD_FOOT_TARGETS`：4 × 髋坐标系足端目标（`0.1 mm`），由 ESP32 端结合 Mahony 姿态进行 100 Hz 实时稳态补偿与 3-DOF 逆解

## 目录结构

```text
pi/
  kinematics.py      # 3-DOF FK/IK、解析 Jacobian、全身位姿逆解
  imu_filter.py      # MPU6050 14 字节编解码、陀螺仪零偏标定、Mahony AHRS
  cpg.py             # RK4 四振子耦合 Hopf CPG + 速度指令足端轨迹规划
  uart_link.py       # CRC16 封包解包、坏帧重同步、ReliableCommander ACK 重传
  sim_runner.py      # 100 Hz 闭环仿真入口（含外力推搡、2.5g 冲击与断链看门狗验证）
esp32/
  hal.h              # Arduino ESP32 (v2/v3) 与本地 Host MockHal 双端硬件抽象
  kinematics.h       # 与 Python 对齐的 C++ 3-DOF FK/IK 与姿态补偿
  imu_mpu6050.h/.cpp # MPU6050 I2C 驱动、零偏标定、Mahony AHRS
  uart_parser.h/.cpp # 防溢出 CRC16 字节流状态机与遥测/ACK 封包
  joint_controller.h/.cpp # 100 Hz 主控循环、FSR 消抖、16-bit PWM 映射、看门狗缓停
tests/
  cpp/test_esp32_firmware.cpp # C++ 固件原生单测 + 二进制管道 --bridge / --ik-batch
  test_*.py          # Python 单元测试、闭环仿真测试与 Python<->C++ 跨语言联调测试
```

## 本地编译、测试与运行

1. 编译并运行 ESP32 C++ 固件原生测试（host `g++`）：
```pwsh
New-Item -ItemType Directory -Force build | Out-Null
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iesp32 esp32/imu_mpu6050.cpp esp32/uart_parser.cpp esp32/joint_controller.cpp tests/cpp/test_esp32_firmware.cpp -o build/test_esp32_firmware.exe
.\build\test_esp32_firmware.exe
```

2. 运行全部 Python 单元测试 + 跨语言 C++ 二进制联调测试：
```pwsh
python -m unittest discover -s tests -v
```

3. 运行 100 Hz 四足闭环步态仿真（含外力扰动 + 断链保护演示）：
```pwsh
python -m pi.sim_runner --duration 4.0 --inject-disturbance --inject-dropout
```
