```markdown
# Hybrid Physics-Informed Vehicle Dynamics Prediction
# 混合物理感知车辆动力学预测系统

本项目实现了一个先进的混合驱动（Hybrid Data-driven & Physics-based）车辆动力学预测模型，它结合了传统的车辆二自由度动力学方程与现代深度学习架构（CNN-LSTM），旨在利用驾驶员输入和车辆状态精确预测车辆的动力学响应（纵向、横向及横摆加速度）。

## 🌟 项目核心亮点

1.  **混合建模架构 (Hybrid Architecture)**:
    *   **CNN-LSTM 主干**: 利用一维卷积提取局部特征，LSTM 处理时序依赖。
    *   **物理嵌入**: 在神经网络的前向传播中内置了基于 `physics.py` 的动力学方程。
2.  **两种融合模式**:
    *   **残差网络模式 (Residual Mode)**: 物理模型作为基准，神经网络学习物理模型的误差（Residual），极大提高了模型的泛化能力和收敛速度。
    *   **特征增强模式 (Feature Mode)**: 物理计算结果作为高阶特征输入全连接层。
3.  **完整的工程链路**:
    *   包含从数据加载、滑动窗口序列化、标准化、训练、验证到测试评估的全流程。

## 📂 文件结构说明

```text
.
├── config.py           # [配置] 路径设置、物理参数(质量/摩擦系数等)、训练超参数
├── physics.py          # [物理] 纯物理模型实现(VehicleDynamicsModel)及基准绘图
├── main.py             # [核心] 混合模型定义(HybridVehicleDynamicsModel)与训练主程序
├── test.py             # [测试] 加载训练好的模型并在测试集上评估性能
├── data/               # [数据] 存放输入输出的 .txt 原始数据
├── model/              # [产出] 存放训练好的 .pth 模型权重
├── scalers/            # [产出] 存放 sklearn 的 StandardScaler 文件
└── results/            # [产出] 存放训练过程 Loss 曲线及预测对比图表
```

## 🛠️ 环境依赖

项目基于 Python 和 PyTorch 实现。请确保安装以下依赖库：

```bash
pip install numpy torch matplotlib scikit-learn joblib
```

## 🚀 快速开始

### 1. 数据准备
请在 `data/` 目录下放入以下格式的 `.txt` 文件（无表头，单列数据）：
*   **输入特征**: `Vx.txt` (纵向速度), `Vy.txt` (横向速度), `AVz.txt` (横摆角速度), `Throttle.txt` (油门), `Pbk_Con.txt` (刹车), `Steer_SW.txt` (方向盘转角)。
*   **目标标签**: `Ax.txt` (纵向加速度), `Ay.txt` (横向加速度), `AAz.txt` (横摆角加速度)。

### 2. 配置参数
在 `config.py` 中调整车辆物理参数以匹配实际车型，并确认训练参数：
```python
PHYSICS_PARAMS = {
    "m": 1500,      # 车辆质量 (kg)
    "Caf": 80000,   # 前轮侧偏刚度
    # ...
}
```

### 3. 运行流程

*   **基准测试 (Optional)**: 运行纯物理模型查看基础效果。
    ```bash
    python physics.py
    ```
*   **模型训练**: 运行主程序进行数据处理和模型训练。
    ```bash
    python main.py
    ```
*   **模型测试**: 加载最佳模型并在测试集上生成详细评估报告。
    ```bash
    python test.py
    ```

## 🧠 模型技术细节

*   **输入处理**: 采用滑动窗口（Time Delay）截取时间序列。
*   **物理层 (Internal Physics)**: 模型内部包含一个不可学习的物理层，它将归一化的 Tensor 反变换为 SI 单位物理量，计算理论加速度，确保模型输出符合基本物理定律。
*   **单位系统**:
    *   输入/输出文件：常用工程单位 (km/h, g, deg)。
    *   内部计算：严格使用 **SI 标准单位** (m/s, m/s², rad)。

## 🤝 贡献指南 (Contributing)

我们非常欢迎外部开发者为本项目做出贡献！无论是修复 Bug、改进物理模型、优化神经网络架构，还是完善文档，您的帮助都非常重要。

在开始之前，请仔细阅读以下指南。

### 1. 如何参与贡献 (Workflow)

我们采用标准的 GitHub 协作流程：

1.  **Fork 本仓库**: 点击右上角的 "Fork" 按钮将仓库复制到您的 GitHub 账户。
2.  **Clone 到本地**:
    ```bash
    git clone https://github.com/你的用户名/car_dynamics_hybrid.git
    ```
3.  **创建分支 (Branch)**:
    *   请避免直接在 `master` 或 `main` 分支上工作。
    *   根据修改类型创建分支，例如：
        *   功能开发: `feat/add-transformer-layer`
        *   Bug 修复: `fix/calculation-error`
        *   文档改进: `docs/update-readme`
4.  **提交代码 (Commit)**:
    *   确保代码可以运行，并通过了基本的测试。
    *   Commit 信息应简洁明了（建议使用英文），例如: `feat: add residual connection to LSTM`。
5.  **推送到远程 (Push)**:
    ```bash
    git push origin feat/你的分支名
    ```
6.  **提交 Pull Request (PR)**: 在 GitHub 页面点击 "New Pull Request"，并按照下方的格式填写描述。

### 2. 代码规范 (Code Guidelines)

为了保持代码库的整洁和可维护性，请遵守以下规范：

*   **代码风格**:
    *   遵循 Python **PEP 8** 编码规范。
    *   推荐使用 4 个空格缩进。
    *   类名使用 `CamelCase`，函数和变量名使用 `snake_case`。
*   **物理一致性 (重要)**:
    *   **变量命名**: 涉及物理量时，请务必在变量名或注释中明确物理含义。
    *   **单位**: 项目内部计算严禁混用单位。所有核心计算逻辑必须使用 **SI 标准单位** (m, s, rad, N, kg)。
        *   ❌ `v = 100` (含义不明)
        *   ✅ `v_ms = 100 / 3.6` (明确转换为 m/s)
*   **注释**:
    *   复杂的物理公式实现（如 `physics.py` 中的动力学方程）必须添加注释，说明公式来源或物理意义。
    *   神经网络模块需注明输入输出的 Tensor 形状 (Shape)。
*   **导入顺序**: 标准库 -> 第三方库 (torch, numpy) -> 本地模块 (config, physics)。

### 3. Pull Request (PR) 描述格式

提交 PR 时，请复制并填写以下模板：

```markdown
### PR 类型
- [ ] ✨ 新功能 (Feature)
- [ ] 🐛 Bug 修复 (Bug Fix)
- [ ] ♻️ 代码重构 (Refactor)
- [ ] 📚 文档更新 (Documentation)

### 描述
简要描述这个 PR 做了什么修改。如果是修复 Bug，请说明导致 Bug 的原因。

### 关联 Issue
Closes #Issue编号 (如果有)

### 测试计划
请描述你是如何测试这项修改的：
- [ ] 已运行 `python test.py` 并通过所有测试
- [ ] 已运行 `python physics.py` 确认物理基准未被破坏
- [ ] 新增了单元测试 (可选)

### 截图 (如有界面变动或性能提升)
在此处粘贴截图或图表。
```

### 4. Issue 提交规范

如果您发现了 Bug 或有新的功能建议，请在 Issues 页面提交。

*   **Bug Report**: 请包含 Bug 描述、复现步骤、环境信息（OS, PyTorch版本）及报错截图。
*   **Feature Request**: 请说明功能背景、建议方案及相关参考资料（如论文链接）。

---
```