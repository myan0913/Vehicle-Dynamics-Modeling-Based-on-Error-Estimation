import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 【修正】从 main.py 导入模型, 从 config 导入所有需要的配置
from config import (
    TRAIN_PARAMS, RESULTS_DIR, MODEL_DIR, SCALERS_DIR, DATA_DIR,
    OUTPUT_FILES, INPUT_FILES, PHYSICS_PARAMS
)
from main import HybridVehicleDynamicsModel

# --- Matplotlib 设置 (保持不变) ---
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16


def set_matplotlib_english():
    """设置matplotlib支持英文显示"""
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = True
    except:
        print("警告: 未能设置Times New Roman字体，将使用默认字体")


# --- 【新增】从 main.py 复制的数据处理函数 ---
def load_data():
    """加载原始数据文件，并对齐到最小长度。返回原始单位的数据。"""
    X_list, y_list = [], []
    input_keys = ["longitudinal_speed", "lateral_speed", "yaw_rate", "throttle", "brake", "steer"]
    output_keys = ["longitudinal_acc", "lateral_acc", "yaw_acc"]

    for key in input_keys:
        path = os.path.join(DATA_DIR, INPUT_FILES[key])
        if not os.path.exists(path): print(f"警告: 文件不存在 {path}"); return None, None
        X_list.append(np.loadtxt(path).reshape(-1, 1))

    for key in output_keys:
        path = os.path.join(DATA_DIR, OUTPUT_FILES[key])
        if not os.path.exists(path): print(f"警告: 文件不存在 {path}"); return None, None
        y_list.append(np.loadtxt(path).reshape(-1, 1))

    X_raw, y_raw = np.hstack(X_list), np.hstack(y_list)
    min_len = min(X_raw.shape[0], y_raw.shape[0])
    return X_raw[:min_len], y_raw[:min_len]


def create_sequences(X, y, time_delay):
    """根据时间延迟创建序列数据。"""
    X_seq, y_seq = [], []
    # 确保与 main.py 中的逻辑完全一致
    for i in range(len(X) - time_delay):
        X_seq.append(X[i:i + time_delay])
        y_seq.append(y[i + time_delay - 1])
    return np.array(X_seq), np.array(y_seq)


# --- 评估与绘图函数 (功能保持不变) ---
def evaluate_model(y_true, y_pred):
    """评估模型性能，假设输入数据单位为 (g, g, rad/s^2)"""
    metrics = {}
    feature_names = ['Longitudinal Acceleration (g)', 'Lateral Acceleration (g)', 'Yaw Acceleration (rad/s^2)']

    for i in range(y_true.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])

        metrics[f'{feature_names[i]}_RMSE'] = rmse
        metrics[f'{feature_names[i]}_R2'] = r2
        metrics[f'{feature_names[i]}_MAE'] = mae

    return metrics


def plot_detailed_comparison(y_true, y_pred, save_path=None, max_samples=2000):
    """绘制详细对比图，假设输入数据单位为 (g, g, rad/s^2)"""
    set_matplotlib_english()

    if y_true.shape[0] > max_samples:
        indices = np.linspace(0, y_true.shape[0] - 1, max_samples, dtype=int)
        y_true_plot, y_pred_plot = y_true[indices], y_pred[indices]
    else:
        y_true_plot, y_pred_plot = y_true, y_pred

    feature_names = ['Longitudinal Acceleration', 'Lateral Acceleration', 'Yaw Acceleration']
    feature_units = ['(g)', '(g)', '(rad/s$^2$)']

    n_features = y_true.shape[1]
    fig, axs = plt.subplots(n_features, 2, figsize=(18, 6 * n_features), gridspec_kw={'width_ratios': [3, 1]})

    if n_features == 1:
        axs = axs.reshape(1, 2)

    for i in range(n_features):
        axs[i, 0].plot(y_true_plot[:, i], 'b-', label='Ground Truth', linewidth=2)
        axs[i, 0].plot(y_pred_plot[:, i], 'r--', label='Prediction', linewidth=2)
        axs[i, 0].set_xlabel('Sample Number')
        axs[i, 0].set_ylabel(f'Value {feature_units[i]}')
        axs[i, 0].set_title(f'{feature_names[i]} Time Series Comparison')
        axs[i, 0].legend()
        axs[i, 0].grid(True)
        axs[i, 1].scatter(y_true_plot[:, i], y_pred_plot[:, i], alpha=0.5, edgecolors='k')
        min_val = min(np.min(y_true_plot[:, i]), np.min(y_pred_plot[:, i]))
        max_val = max(np.max(y_true_plot[:, i]), np.max(y_pred_plot[:, i]))
        axs[i, 1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal Fit')
        axs[i, 1].set_xlabel('Ground Truth')
        axs[i, 1].set_ylabel('Prediction')
        axs[i, 1].set_title(f'{feature_names[i]} Scatter Plot')
        axs[i, 1].grid(True)
        axs[i, 1].axis('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"详细比较图已保存至: {save_path}")
    plt.close()


# --- 主测试函数 (核心修改部分) ---
def test_model():
    """主测试函数，加载模型，并用原始单位数据进行评估"""
    # 1. 设置参数
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    time_delay = TRAIN_PARAMS["time_delay"]
    use_residual_network = TRAIN_PARAMS["use_residual_network"]
    G_CONST = 9.81  # 用于单位转换的常量

    # 【修正】根据配置确定模型名称，以加载正确的文件
    mode_name = "hybrid_cnn_lstm" if use_residual_network else "physics_enhanced_cnn_lstm"
    print(f"正在测试模式: {mode_name}")

    # 2. 【修正】加载原始数据 (X和y均为原始单位)
    print("正在加载原始数据...")
    X_raw, y_original_full = load_data()
    if X_raw is None:
        print("错误: 无法加载数据文件。请检查config.py中的路径和文件名。")
        return

    # 3. 【修正】准备模型输入 X (与训练时完全一致)
    print("正在准备模型输入数据 (X)...")
    # 单位转换 (原始单位 -> 物理SI单位)
    X_si = X_raw.copy()
    X_si[:, 0] *= (1 / 3.6)  # km/h -> m/s
    X_si[:, 1] *= (1 / 3.6)  # km/h -> m/s
    X_si[:, 2] *= (np.pi / 180)  # deg/s -> rad/s
    X_si[:, 5] *= (np.pi / 180)  # deg -> rad (steer wheel)

    # 创建时间序列 (我们只需要 X_seq)
    X_seq, _ = create_sequences(X_si, y_original_full, time_delay)
    num_features = X_seq.shape[2]

    # 对齐 y_original (原始单位) 和 X_seq
    y_original_aligned = y_original_full[time_delay - 1: len(X_seq) + time_delay - 1]

    # 4. 【修正】划分测试集 (与 main.py 完全一致)
    val_split_ratio = TRAIN_PARAMS["validation_split"]
    test_split_ratio = TRAIN_PARAMS["test_split"]

    dataset_size = len(X_seq)
    test_size = int(test_split_ratio * dataset_size)
    train_val_size = dataset_size - test_size
    val_size = int(val_split_ratio * train_val_size)
    train_size = train_val_size - val_size

    X_test = X_seq[train_size + val_size:]
    y_test_original = y_original_aligned[train_size + val_size:]
    print(f"测试集样本数: {len(X_test)}")

    # 5. 【修正】加载标准化器、模型并进行预测
    model_path = os.path.join(MODEL_DIR, f"model_{mode_name}_best.pth")
    scaler_X_path = os.path.join(SCALERS_DIR, f"scaler_X_{mode_name}.pkl")
    scaler_y_path = os.path.join(SCALERS_DIR, f"scaler_y_{mode_name}.pkl")

    try:
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
    except FileNotFoundError:
        print(f"错误: 找不到scaler文件。请确保 '{scaler_X_path}' 和 '{scaler_y_path}' 存在。")
        print("请先运行 main.py 进行训练。")
        return

    # 初始化模型
    model = HybridVehicleDynamicsModel(
        time_delay=time_delay,
        num_features=num_features,
        physics_params=PHYSICS_PARAMS,
        use_residual_network=use_residual_network
    ).to(device)

    # 【关键】将 scalers 设置到模型中，这是新模型的要求
    model.set_scalers(scaler_X, scaler_y, device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{model_path}'。请先运行 main.py 进行训练。")
        return

    # 准备测试数据
    X_test_reshaped = X_test.reshape(-1, num_features)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # 模型预测
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    # 6. 【关键】将模型预测(SI单位)转换为原始单位，以和 y_test_original 比较
    print("正在对齐预测结果的单位...")
    # 反标准化，得到SI单位的预测结果 (m/s^2, m/s^2, rad/s^2)
    predictions_si = scaler_y.inverse_transform(predictions_scaled)

    predictions_original = predictions_si.copy()
    predictions_original[:, 0:2] /= G_CONST  # m/s^2 -> g

    # 7. 在原始单位上评估模型性能
    print("\n--- 模型性能评估 (测试集, 原始单位: g, g, rad/s²) ---")
    metrics = evaluate_model(y_test_original, predictions_original)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 8. 在原始单位上可视化结果
    print("\n正在生成详细的性能对比图...")
    plot_save_path = os.path.join(RESULTS_DIR, f"test_set_detailed_comparison_{mode_name}.png")
    plot_detailed_comparison(
        y_test_original,
        predictions_original,
        save_path=plot_save_path
    )


if __name__ == "__main__":
    print("开始测试混合车辆动力学模型...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_model()
    print("\n测试完成!")
