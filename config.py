import os

# 路径配置
DATA_DIR = "data"
MODEL_DIR = "model"
SCALERS_DIR = "scalers"
RESULTS_DIR = "results"

# 输入文件 (假设单位: 速度km/h, 角速度deg/s, 油门0-1, 刹车MPa, 方向盘转角deg)
INPUT_FILES = {
    "longitudinal_speed": "Vx.txt",
    "lateral_speed": "Vy.txt",
    "yaw_rate": "AVz.txt",
    "throttle": "Throttle.txt",
    "brake": "Pbk_Con.txt",
    "steer": "Steer_SW.txt"
}

# 输出文件 (假设单位: 加速度g, 角加速度rad/s^2)
OUTPUT_FILES = {
    "longitudinal_acc": "Ax.txt",
    "lateral_acc": "Ay.txt",
    "yaw_acc": "AAz.txt"
}

# 【统一】物理模型默认参数，与模型一保持一致
PHYSICS_PARAMS = {
    "m": 1500,              # 车辆总质量 (kg)
    "Iz": 2500,             # 车辆绕z轴转动惯量 (kg·m²)
    "a": 1.14,              # 前轴到质心距离 (m)
    "b": 1.40,              # 后轴到质心距离 (m)
    "Caf": 80000,           # 前轮侧偏刚度 (N/rad)
    "Car": 80000,           # 后轮侧偏刚度 (N/rad)
    "steering_ratio": 19.8, # 方向盘到前轮的传动比
    "F_drive_max": 8000.0,  # 最大驱动力 (N)
    "M_brake": 2500.0       # 制动系数 (N/MPa)
}

# 训练参数
TRAIN_PARAMS = {
    "time_delay": 10,
    "epochs": 500,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "patience": 30,
    "validation_split": 0.15,
    "test_split": 0.15,
    "use_residual_network": True,  # 控制是否使用残差网络
}

# 确保目录存在
for directory in [DATA_DIR, MODEL_DIR, SCALERS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
