import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import *
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl


# ==================================
# 物理模型 (应用小角度假设)
# ==================================
class VehicleDynamicsModel:
    def __init__(self, params=None):
        self.params = PHYSICS_PARAMS.copy() if params is None else params

    def calculate_accelerations_batch(self, states_batch, inputs_batch, params_batch):
        vx = states_batch[:, 0]
        vy = states_batch[:, 1]
        r = states_batch[:, 2]
        throttle = inputs_batch[:, 0]
        brake = inputs_batch[:, 1]
        delta = inputs_batch[:, 2]
        m, Iz, a, b, Caf, Car, F_drive_max, M_brake = (
            params_batch[k] for k in ["m", "Iz", "a", "b", "Caf", "Car", "F_drive_max", "M_brake"]
        )
        vx_safe = torch.clamp(vx, min=0.5)
        F_drive = throttle * F_drive_max
        F_brake = brake * M_brake
        Fx_total_wheels = F_drive - F_brake
        alpha_f = delta - (vy + a * r) / vx_safe
        alpha_r = (b * r - vy) / vx_safe
        Fyf = 2 * Caf * alpha_f
        Fyr = 2 * Car * alpha_r
        F_lateral_component_on_x = Fyf * delta
        vx_dot = vy * r + (1 / m) * (Fx_total_wheels + F_lateral_component_on_x)
        vy_dot = -vx_safe * r + (1 / m) * (Fyf + Fyr)
        r_dot = (1 / Iz) * (a * Fyf - b * Fyr)
        return torch.stack([vx_dot, vy_dot, r_dot], dim=1)


# ==================================
# 深度学习模型 (适配新物理模型)
# ==================================


class SequentialParameterNetwork(nn.Module):
    """
    一个更先进的序列处理网络，灵感来源于模型2的 CNN+BiLSTM 结构。
    它直接处理时间序列数据，以更好地捕捉动态特征。
    """

    def __init__(self, input_feature_dim, hidden_size=128, lstm_layers=2, dropout=0.2):
        super(SequentialParameterNetwork, self).__init__()

        # 1. 卷积层：用于提取局部时序特征
        # 输入形状: (batch, features, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_feature_dim, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # 2. BiLSTM层：用于捕捉长距离时序依赖
        # 输入形状: (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 3. 输出头：与原模型保持一致
        # LSTM输出的特征维度是 hidden_size * 2 (因为是双向的)
        lstm_output_dim = hidden_size * 2

        # 参数辨识头 (8个参数)
        self.param_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()  # 输出在 0-1 之间
        )

        # 误差补偿头 (3个输出)
        self.error_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        # x 的输入形状: (batch_size, seq_len, input_feature_dim)

        # 卷积层需要 (batch, features, seq_len)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.relu1(self.conv1(x_conv))
        x_conv = self.relu2(self.conv2(x_conv))

        # 将形状转换回LSTM需要的 (batch, seq_len, features)
        x_lstm = x_conv.permute(0, 2, 1)

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)

        # 我们使用LSTM在所有时间步上的输出的平均值作为整个序列的特征表示
        # 这是比只用最后一个时间步的输出更稳健的做法
        # lstm_out 形状: (batch, seq_len, hidden_size * 2)
        sequence_features = torch.mean(lstm_out, dim=1)

        # 通过两个头得到最终输出
        param_corrections = self.param_head(sequence_features)
        error_predictions = self.error_head(sequence_features)

        return param_corrections, error_predictions


class HybridVehicleDynamicsModel(nn.Module):
    def __init__(self, time_delay, num_features, physics_params, use_residual_network=True):
        super(HybridVehicleDynamicsModel, self).__init__()
        self.physics_model = VehicleDynamicsModel(physics_params)

        # 使用新的、更强大的序列网络
        self.nn_model = SequentialParameterNetwork(
            input_feature_dim=num_features,
            hidden_size=128,  # 可以调整的超参数
            lstm_layers=2,  # 可以调整的超参数
            dropout=0.2  # 可以调整的超参数
        )

        self.use_residual_network = use_residual_network

        if self.use_residual_network:
            # 残差网络的输入维度也需要相应修改
            # SequentialParameterNetwork的输出维度是固定的，所以残差网络的输入也应该是序列
            self.residual_network = nn.Sequential(
                nn.Linear(time_delay * num_features, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
            )
            self.alpha = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))

        self.identifiable_params = [
            "m", "Iz", "a", "b", "Caf", "Car", "F_drive_max", "M_brake"
        ]
        self.base_params = {k: torch.tensor(physics_params[k]) for k in self.identifiable_params}
        self.steering_ratio = torch.tensor(physics_params["steering_ratio"])

        self.register_buffer('scaler_X_mean', None)
        self.register_buffer('scaler_X_scale', None)
        self.register_buffer('scaler_y_mean', None)
        self.register_buffer('scaler_y_scale', None)

    def set_scalers(self, scaler_X: StandardScaler, scaler_y: StandardScaler, device):
        self.scaler_X_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=device)
        self.scaler_X_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=device)
        self.scaler_y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
        self.scaler_y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)
        for key, val in self.base_params.items():
            self.base_params[key] = val.to(device)
        self.steering_ratio = self.steering_ratio.to(device)

    def forward(self, x):
        # x 的输入形状: (batch_size, time_delay, num_features)
        batch_size, device = x.shape[0], x.device
        if self.scaler_X_mean is None:
            raise RuntimeError("Scalers must be set before forward pass.")

        # 新的nn_model可以直接处理序列数据，无需展平
        param_corrections_raw, error_predictions_physical = self.nn_model(x)

        # 将Sigmoid输出 [0, 1] 映射到修正因子范围 [0.5, 1.5]
        param_corrections = param_corrections_raw * 1.0 + 0.5

        # --- 后续逻辑保持不变 ---
        current_x_standardized = x[:, -1, :]
        current_x_physical = current_x_standardized * self.scaler_X_scale + self.scaler_X_mean

        vx, vy, r = current_x_physical[:, 0], current_x_physical[:, 1], current_x_physical[:, 2]
        throttle, brake, steer_sw_rad = current_x_physical[:, 3], current_x_physical[:, 4], current_x_physical[:, 5]
        delta = steer_sw_rad / self.steering_ratio

        # 注意：这里的param_corrections已经不是原先的定义了，需要根据新的映射关系调整
        # 原来是 * 1.0 + 0.5，现在我们保持这个逻辑，但要注意param_corrections_raw才是原始输出
        corrected_params = {
            key: self.base_params[key].expand(batch_size) * param_corrections[:, i]
            for i, key in enumerate(self.identifiable_params)
        }

        states_batch = torch.stack([vx, vy, r], dim=1)
        inputs_batch = torch.stack([throttle, brake, delta], dim=1)

        physics_outputs_physical = self.physics_model.calculate_accelerations_batch(
            states_batch, inputs_batch, corrected_params
        )
        physics_outputs_physical += error_predictions_physical
        physics_outputs_standardized = (physics_outputs_physical - self.scaler_y_mean) / self.scaler_y_scale

        if self.use_residual_network:
            # 残差网络仍然使用展平的输入
            x_flat = x.reshape(batch_size, -1)
            direct_predictions_standardized = self.residual_network(x_flat)
            sigmoid_alpha = torch.sigmoid(self.alpha).unsqueeze(0)
            hybrid_outputs = sigmoid_alpha * physics_outputs_standardized + (
                    1 - sigmoid_alpha) * direct_predictions_standardized
        else:
            hybrid_outputs = physics_outputs_standardized

        return hybrid_outputs




# ===================================
# 工具函数
# ===================================
def set_matplotlib_chinese():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未能设置中文字体。")


def plot_training_history(train_losses, val_losses, save_path=None):
    set_matplotlib_chinese()
    plt.figure(figsize=(10, 6));
    plt.plot(train_losses, label='Training Loss');
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title('Training and Validation Loss');
    plt.legend();
    plt.grid(True)
    if save_path: plt.savefig(save_path)
    plt.close()


def plot_prediction_comparison(y_true, y_pred, feature_names, save_path=None):
    set_matplotlib_chinese()
    n_features = y_true.shape[1]
    fig, axs = plt.subplots(n_features, 1, figsize=(15, 4 * n_features), sharex=True)
    if n_features == 1: axs = [axs]
    for i in range(n_features):
        axs[i].plot(y_true[:, i], label='True', alpha=0.8);
        axs[i].plot(y_pred[:, i], label='Predicted', alpha=0.8, linestyle='--')
        axs[i].set_ylabel(feature_names[i]);
        axs[i].set_title(f'{feature_names[i]} Comparison');
        axs[i].legend();
        axs[i].grid(True)
    axs[-1].set_xlabel('Sample Index');
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()


def evaluate_model(y_true, y_pred, feature_names):
    metrics = {}
    for i in range(y_true.shape[1]):
        metrics[f'{feature_names[i]}_RMSE'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        metrics[f'{feature_names[i]}_R2'] = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f'{feature_names[i]}_MAE'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
    return metrics


# ====================================
# 数据处理函数
# ====================================
def load_data():
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
    X_seq, y_seq = [], []
    for i in range(len(X) - time_delay):
        X_seq.append(X[i:i + time_delay])
        y_seq.append(y[i + time_delay - 1])
    return np.array(X_seq), np.array(y_seq)


# =================================
# 主函数
# =================================
def main():
    torch.manual_seed(42);
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 1. 从config.py获取所有参数 ---
    time_delay = TRAIN_PARAMS["time_delay"]
    epochs = TRAIN_PARAMS["epochs"]
    batch_size = TRAIN_PARAMS["batch_size"]
    learning_rate = TRAIN_PARAMS["learning_rate"]
    weight_decay = TRAIN_PARAMS["weight_decay"]
    patience = TRAIN_PARAMS["patience"]
    val_split_ratio = TRAIN_PARAMS["validation_split"]
    test_split_ratio = TRAIN_PARAMS["test_split"]
    use_residual_network = TRAIN_PARAMS["use_residual_network"]
    mode_name = "hybrid_cnn_lstm" if use_residual_network else "physics_enhanced_cnn_lstm"
    print(f"当前模式: {mode_name}")

    # --- 2. 数据加载 ---
    X_raw, y_raw = load_data()
    if X_raw is None:
        print("错误: 无法加载数据文件。请检查config.py中的路径和文件名。正在生成示例数据...")
        n_samples = 2000
        X_raw = np.random.rand(n_samples, 6) * np.array([200, 20, 45, 1, 1, 360])
        y_raw = np.random.rand(n_samples, 3) * np.array([2, 2, 1])

    # --- 3. 单位转换 (原始单位 -> 物理SI单位) ---
    X_si = X_raw.copy()
    X_si[:, 0] *= (1 / 3.6)
    X_si[:, 1] *= (1 / 3.6)
    X_si[:, 2] *= (np.pi / 180)
    X_si[:, 5] /= 19.8
    y_si = y_raw.copy()
    y_si[:, 0] *= 9.81
    y_si[:, 1] *= 9.81

    # --- 4. 创建时间序列 ---
    X_seq, y_seq = create_sequences(X_si, y_si, time_delay)
    num_features = X_seq.shape[2]

    # --- 5. 按时间顺序分割数据集 ---
    dataset_size = len(X_seq)
    test_size = int(test_split_ratio * dataset_size)
    train_val_size = dataset_size - test_size
    val_size = int(val_split_ratio * train_val_size)
    train_size = train_val_size - val_size
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size + val_size], y_seq[train_size:train_size + val_size]
    X_test, y_test = X_seq[train_size + val_size:], y_seq[train_size + val_size:]
    print(f"数据集大小: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")

    # --- 6. 标准化 ---
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    y_val_scaled = scaler_y.transform(y_val)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
    os.makedirs(SCALERS_DIR, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(SCALERS_DIR, f"scaler_X_{mode_name}.pkl"))
    joblib.dump(scaler_y, os.path.join(SCALERS_DIR, f"scaler_y_{mode_name}.pkl"))

    # --- 7. 创建DataLoader ---
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                torch.tensor(y_val_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- 8. 模型初始化与训练 ---
    model = HybridVehicleDynamicsModel(
        time_delay, num_features, PHYSICS_PARAMS, use_residual_network
    ).to(device)
    model.set_scalers(scaler_X, scaler_y, device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5, verbose=True)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))
        scheduler.step(val_losses[-1])
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_{mode_name}_best.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停触发: 验证损失在 {patience} 个轮次内未改善。")
                break
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_training_history(train_losses, val_losses,
                          save_path=os.path.join(RESULTS_DIR, f'training_history_{mode_name}.png'))

    # --- 9. 测试与评估 ---
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"model_{mode_name}_best.pth")))
    model.eval()
    all_predictions_scaled, all_targets_si = [], []
    with torch.no_grad():
        for batch_X, batch_y_si in test_loader:
            batch_X = batch_X.to(device)
            outputs_scaled = model(batch_X)
            all_predictions_scaled.append(outputs_scaled.cpu().numpy())
            all_targets_si.append(batch_y_si.numpy())
    all_predictions_scaled = np.vstack(all_predictions_scaled)
    all_targets_si = np.vstack(all_targets_si)
    all_predictions_si = scaler_y.inverse_transform(all_predictions_scaled)
    all_predictions_report = all_predictions_si.copy()
    all_predictions_report[:, :2] /= 9.81
    all_targets_report = all_targets_si.copy()
    all_targets_report[:, :2] /= 9.81
    feature_names = ['纵向加速度 (g)', '横向加速度 (g)', '横摆角加速度 (rad/s²)']
    metrics = evaluate_model(all_targets_report, all_predictions_report, feature_names)
    print("\n--- 模型性能评估 ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    plot_prediction_comparison(
        all_targets_report, all_predictions_report, feature_names,
        save_path=os.path.join(RESULTS_DIR, f'prediction_comparison_{mode_name}.png')
    )
    print(f"\n评估完成，结果图已保存至 '{RESULTS_DIR}' 目录。")


if __name__ == "__main__":
    main()
