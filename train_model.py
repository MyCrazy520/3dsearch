import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先加载黑体（Windows内置），备用DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
plt.rcParams['font.family'] = 'sans-serif'
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 全局配置与路径初始化 =====================
# 核心路径配置（严格按要求定义）
FREECAD_PATH = r"D:\FreeCAD"
STEP_ROOT_DIR = r"D:\graduate\cad\pointnet\Opencascade\step\标准件\GB国家标准"
FEATURE_SAVE_PATH = r"./step_features"  # 几何特征/PointNet特征保存目录
MODEL_SAVE_PATH = r"./trained_models"  # 模型保存目录
VIS_SAVE_PATH = r"./visualization"  # 可视化结果保存目录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PATH"] += os.pathsep + os.path.join(FREECAD_PATH, "bin")  # 加入FreeCAD bin路径

# 超参数配置
POINT_NUM = 1024  # 点云采样点数（PointNet标准输入）
BATCH_SIZE = 32
EPOCHS_POINTNET = 50
LR_POINTNET = 1e-3
EPOCHS_LGB = 100
LR_LGB = 0.1
NUM_CLASSES = None  # 自动统计类别数
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 创建保存目录
for dir_path in [FEATURE_SAVE_PATH, MODEL_SAVE_PATH, VIS_SAVE_PATH]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# ===================== 2. STEP几何特征提取（FreeCAD/OpenCASCADE） =====================
def step2point_cloud(step_file_path, num_points=POINT_NUM):
    """
    从STEP文件提取几何特征并转换为标准化点云
    依赖：FreeCAD bin（OpenCASCADE内核）、open3d点云处理
    :param step_file_path: STEP文件路径
    :param num_points: 输出点云采样点数
    :return: (num_points, 3) 标准化点云数组，失败返回None
    """
    try:
        # 初始化FreeCAD/OpenCASCADE环境
        sys.path.append(os.path.join(FREECAD_PATH, "bin"))
        import FreeCAD
        import Part
        FreeCAD.Console.PrintLog = lambda *args: None  # 屏蔽日志输出

        # 加载STEP文件并提取形状
        doc = FreeCAD.newDocument("temp")
        shape = Part.read(step_file_path)
        if shape is None:
            return None

        # 转换为点云（离散化几何特征）
        mesh = shape.tessellate(0.1)  # 细化度，值越小点越密
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh[0])

        # 标准化点云（平移到原点+单位球归一化）
        pcd = pcd.voxel_down_sample(voxel_size=0.05)  # 下采样去重
        points = np.asarray(pcd.points)
        if len(points) < num_points:
            # 点数不足时补零
            points = np.pad(points, ((0, num_points - len(points)), (0, 0)), mode='constant')
        else:
            # 随机采样到指定点数
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]

        # 几何特征标准化（关键：消除尺度/位置影响）
        points = points - np.mean(points, axis=0)  # 中心化
        points = points / np.max(np.linalg.norm(points, axis=1))  # 单位球归一化
        return points.astype(np.float32)

    except Exception as e:
        print(f"提取{step_file_path}特征失败: {str(e)}")
        return None
    finally:
        # 清理FreeCAD临时文档
        FreeCAD.closeDocument("temp")


def load_step_dataset(root_dir):
    """
    按目录分类加载STEP数据集，生成点云+标签+文件路径
    目录结构：root_dir/类别1/xxx.step, root_dir/类别2/xxx.step
    :return: points_list(点云列表), labels(数字标签), file_paths(文件路径列表), class2idx(类别到数字映射)
    """
    class2idx = {}
    points_list = []
    labels = []
    file_paths = []  # 新增：记录每个有效样本的STEP文件完整路径
    class_idx = 0

    # 遍历一级目录作为类别（按GB国家标准分类）
    for class_name in tqdm(os.listdir(root_dir), desc="加载STEP数据集"):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class2idx[class_name] = class_idx

        # 遍历类别下所有STEP文件
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(".step") or file_name.lower().endswith(".stp"):
                step_path = os.path.join(class_dir, file_name)
                point_cloud = step2point_cloud(step_path)
                if point_cloud is not None:
                    points_list.append(point_cloud)
                    labels.append(class_idx)
                    file_paths.append(step_path)  # 同步记录有效文件路径
        class_idx += 1

    # 全局类别数赋值
    global NUM_CLASSES
    NUM_CLASSES = class_idx
    # 转换为数组/列表（文件路径保持列表，便于后续过滤）
    points_array = np.array(points_list)
    labels_array = np.array(labels)
    print(f"数据集加载完成：{len(points_array)}个样本，{NUM_CLASSES}个类别")
    print(f"类别映射：{class2idx}")
    # 保存类别映射（后续预测用）
    np.save(os.path.join(MODEL_SAVE_PATH, "class2idx.npy"), class2idx)
    # 返回值新增file_paths
    return points_array, labels_array, file_paths, class2idx


# ===================== 3. PointNet模型定义（特征提取核心） =====================
class TNet(nn.Module):
    """PointNet基础变换网络，提取空间变换特征"""

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # 初始化变换矩阵为单位矩阵
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + iden
        return x.view(batch_size, self.k, self.k)


class PointNetFeatureExtractor(nn.Module):
    """PointNet特征提取网络（两阶段：局部特征+全局特征）"""

    def __init__(self, feature_dim=1024):
        super(PointNetFeatureExtractor, self).__init__()
        self.tnet3 = TNet(k=3)  # 点云坐标变换
        self.tnet64 = TNet(k=64)  # 特征空间变换
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        n_pts = x.size(2)
        # 第一阶段：坐标变换+局部特征提取
        trans3 = self.tnet3(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans3)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # 第二阶段：特征变换+全局特征提取
        trans64 = self.tnet64(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans64)
        x = x.transpose(2, 1)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        # 全局池化（最大池化提取核心特征）
        global_feat = torch.max(x, 2, keepdim=True)[0]
        global_feat = global_feat.view(batch_size, -1)
        return global_feat  # 输出(bs, 1024)全局特征


class PointNetClassifier(nn.Module):
    """PointNet分类器（基于特征提取网络，用于预训练特征提取器）"""

    def __init__(self, num_classes, feature_dim=1024):
        super(PointNetClassifier, self).__init__()
        self.feature_extractor = PointNetFeatureExtractor(feature_dim)
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        global_feat = self.feature_extractor(x)  # 提取全局特征
        # 分类头
        x = self.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, global_feat  # 输出预测值+全局特征


# ===================== 4. 数据集类与PointNet训练 =====================
class STEPDataset(Dataset):
    """STEP点云数据集类，适配PyTorch DataLoader"""

    def __init__(self, points, labels):
        self.points = torch.from_numpy(points).transpose(2, 1)  # (N, 3, 1024) PointNet输入格式
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]


def train_pointnet(points, labels):
    """
    训练PointNet模型，提取并保存全局特征（用于后续LightGBM训练）
    :return: pointnet_feat(PointNet提取的特征数组), labels(标签数组)
    """
    # 构建数据集并划分训练/验证集
    dataset = STEPDataset(points, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型、优化器、损失函数
    model = PointNetClassifier(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_POINTNET, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 学习率衰减

    # 训练过程
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    print(f"\n开始训练PointNet（设备：{DEVICE}，epochs：{EPOCHS_POINTNET}）")
    for epoch in range(EPOCHS_POINTNET):
        # 训练轮
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # 统计指标
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += data.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证轮
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output, _ = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += data.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        # 学习率衰减
        scheduler.step()

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "pointnet_best.pth"))

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1:2d} | 训练损失：{train_loss:.4f} | 训练精度：{train_acc:.4f} | 验证损失：{val_loss:.4f} | 验证精度：{val_acc:.4f}")

    # 加载最优模型，提取全量数据的PointNet特征
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "pointnet_best.pth")))
    model.eval()
    full_dataset = DataLoader(STEPDataset(points, labels), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    pointnet_feat = []
    with torch.no_grad():
        for data, _ in full_dataset:
            data = data.to(DEVICE)
            _, feat = model(data)
            pointnet_feat.append(feat.cpu().numpy())
    pointnet_feat = np.concatenate(pointnet_feat, axis=0)

    # 保存PointNet特征和训练曲线
    np.save(os.path.join(FEATURE_SAVE_PATH, "pointnet_global_feat.npy"), pointnet_feat)
    plot_training_curve(train_losses, val_losses, train_accs, val_accs, "PointNet训练曲线", "pointnet_train_curve.png")
    print(f"\nPointNet训练完成，最优验证精度：{best_val_acc:.4f}")
    print(f"PointNet全局特征保存至：{os.path.join(FEATURE_SAVE_PATH, 'pointnet_global_feat.npy')}")
    return pointnet_feat, labels


# ===================== 5. 特征净化（关键优化步骤） =====================
def feature_purification(feat, labels, file_paths, threshold=3.0):
    """
    特征净化：移除异常特征样本，同步过滤文件路径，提升后续LightGBM训练效果
    方法：基于类内特征的Z-score过滤，保留类内正常分布的特征
    :param feat: 原始特征数组 (N, D)
    :param labels: 标签数组 (N,)
    :param file_paths: 原始文件路径列表 (N,)
    :param threshold: Z-score阈值，超过则视为异常
    :return: clean_feat(净化后特征), clean_labels(净化后标签), clean_file_paths(净化后文件路径)
    """
    print(f"\n开始特征净化（原始样本数：{len(feat)}，类别数：{NUM_CLASSES}）")
    clean_feat_list, clean_labels_list, clean_file_paths_list = [], [], []
    for cls in range(NUM_CLASSES):
        # 提取当前类别的特征、标签、文件路径掩码
        cls_mask = labels == cls
        cls_feat = feat[cls_mask]
        cls_labels = labels[cls_mask]
        cls_file_paths = [file_paths[i] for i in np.where(cls_mask)[0]]
        if len(cls_feat) == 0:
            continue
        # 计算类内特征的Z-score（按特征维度）
        scaler = StandardScaler()
        cls_feat_scaled = scaler.fit_transform(cls_feat)
        # 计算每个样本的平均Z-score
        z_score = np.mean(np.abs(cls_feat_scaled), axis=1)
        # 过滤异常样本（Z-score < 阈值）
        normal_idx = z_score < threshold
        clean_cls_feat = cls_feat[normal_idx]
        clean_cls_labels = cls_labels[normal_idx]
        clean_cls_file_paths = [cls_file_paths[i] for i in np.where(normal_idx)[0]]
        # 加入结果列表
        clean_feat_list.append(clean_cls_feat)
        clean_labels_list.append(clean_cls_labels)
        clean_file_paths_list.extend(clean_cls_file_paths)
    # 合并净化后的数据
    clean_feat = np.concatenate(clean_feat_list, axis=0)
    clean_labels = np.concatenate(clean_labels_list, axis=0).astype(int)
    print(f"特征净化完成，净化后样本数：{len(clean_feat)}（剔除{len(feat) - len(clean_feat)}个异常样本）")
    # ========== 新增代码：保存全量净化后文件路径 ==========
    clean_file_paths_npy = os.path.join(FEATURE_SAVE_PATH, "clean_file_paths.npy")
    np.save(clean_file_paths_npy, clean_file_paths_list)  # 保存为npy格式，兼容列表
    print(f"全量净化后文件路径保存至：{clean_file_paths_npy}")
    # ====================================================
    # 保存净化后特征
    np.save(os.path.join(FEATURE_SAVE_PATH, "clean_pointnet_feat.npy"), clean_feat)
    np.save(os.path.join(FEATURE_SAVE_PATH, "clean_labels.npy"), clean_labels)
    return clean_feat, clean_labels, clean_file_paths_list


# ===================== 6. LightGBM训练（分层采样+分类） =====================
def train_lightgbm(feat, labels, file_paths):
    """
    LightGBM训练：分层采样拆分数据集，训练分类模型，保存模型，同步返回测试集文件路径
    核心：分层采样保证训练/测试集类别分布一致，适配小样本/不平衡数据集
    :param feat: 净化后特征数组
    :param labels: 净化后标签数组
    :param file_paths: 净化后文件路径列表
    :return: lgb_model(训练好的LightGBM模型), y_test(测试集标签), y_pred(测试集预测), test_file_paths(测试集文件路径)
    """
    # 分层采样拆分训练/测试集（按类别比例拆分，test_size=0.2）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for train_idx, test_idx in sss.split(feat, labels):
        X_train, X_test = feat[train_idx], feat[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        test_file_paths = [file_paths[i] for i in test_idx]  # 提取测试集对应的文件路径
    print(f"\nLightGBM分层采样完成：训练集{len(X_train)}样本，测试集{len(X_test)}样本")

    # 构建LightGBM数据集
    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False)

    # LightGBM参数配置（分类任务）
    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "learning_rate": LR_LGB,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "device": "gpu" if torch.cuda.is_available() else "cpu",
        "verbose": -1  # 关闭默认日志，由log_evaluation控制打印
    }

    # 定义回调函数（整合早停+日志打印，高版本LightGBM标准写法）
    callbacks = [
        # 早停回调：10轮无提升则停止训练，打印早停日志
        lgb.callback.early_stopping(
            stopping_rounds=10,
            verbose=True,
            min_delta=0.001
        ),
        # 日志打印回调：每10轮打印一次训练/验证集指标
        lgb.callback.log_evaluation(
            period=10  # 与原verbose_eval=10逻辑完全一致
        )
    ]

    # 训练LightGBM（移除verbose_eval，回调统一传入callbacks）
    print(f"开始训练LightGBM（epochs：{EPOCHS_LGB}）")
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=EPOCHS_LGB,
        valid_sets=[lgb_train, lgb_test],
        callbacks=callbacks  # 核心：所有回调逻辑统一在此传入
    )

    # 预测与评估
    y_pred_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nLightGBM测试集精度：{acc:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=list(class2idx.keys())))

    # 保存LightGBM模型
    lgb_model.save_model(os.path.join(MODEL_SAVE_PATH, "lightgbm_best.txt"))
    print(f"LightGBM模型保存至：{os.path.join(MODEL_SAVE_PATH, 'lightgbm_best.txt')}")
    # 返回值新增test_file_paths
    return lgb_model, y_test, y_pred, test_file_paths


# ===================== 7. 结果可视化与保存 =====================
def plot_training_curve(train_loss, val_loss, train_acc, val_acc, title, save_name):
    """绘制模型训练曲线（损失+精度）"""
    plt.figure(figsize=(12, 4))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="训练损失", color="blue")
    plt.plot(val_loss, label="验证损失", color="red")
    plt.title(f"{title} - 损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="训练精度", color="blue")
    plt.plot(val_acc, label="验证精度", color="red")
    plt.title(f"{title} - 精度曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_SAVE_PATH, save_name), dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("模型混淆矩阵", fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    # 标注数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_SAVE_PATH, save_name), dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(lgb_model, class2idx, save_name):
    """绘制LightGBM特征重要性"""
    plt.figure(figsize=(12, 6))
    feat_imp = pd.Series(lgb_model.feature_importance(), index=[f"feat_{i}" for i in range(lgb_model.num_feature())])
    feat_imp = feat_imp.sort_values(ascending=False)[:50]  # 取前50个重要特征
    feat_imp.plot(kind="bar", color="green")
    plt.title("LightGBM Top50 特征重要性", fontsize=14)
    plt.xlabel("特征索引")
    plt.ylabel("特征重要性")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_SAVE_PATH, save_name), dpi=300, bbox_inches="tight")
    plt.close()


# ===================== 8. 主函数（全流程执行） =====================
if __name__ == "__main__":
    # 步骤1：加载STEP数据集并提取几何点云特征（接收新增的file_paths）
    points, labels, file_paths, class2idx = load_step_dataset(STEP_ROOT_DIR)
    if len(points) == 0:
        raise ValueError("数据集加载失败，无有效STEP样本")

    # 步骤2：训练PointNet，提取全局特征
    pointnet_feat, labels = train_pointnet(points, labels)

    # 步骤3：特征净化（关键优化，传递file_paths并接收净化后的clean_file_paths）
    clean_feat, clean_labels, clean_file_paths = feature_purification(pointnet_feat, labels, file_paths)

    # 步骤4：LightGBM分层采样训练（传递clean_file_paths并接收测试集test_file_paths）
    lgb_model, y_test, y_pred, test_file_paths = train_lightgbm(clean_feat, clean_labels, clean_file_paths)

    # 步骤5：结果可视化与保存（全量结果，新增STEP文件路径列）
    # 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, list(class2idx.keys()), "model_confusion_matrix.png")
    # LightGBM特征重要性
    plot_feature_importance(lgb_model, class2idx, "lgb_feature_importance.png")
    # 保存最终预测结果（新增「STEP文件路径」列，放在第一列更直观）
    pred_result = pd.DataFrame({
        "STEP文件路径": test_file_paths,
        "真实标签": [list(class2idx.keys())[i] for i in y_test],
        "预测标签": [list(class2idx.keys())[i] for i in y_pred]
    })
    pred_result.to_csv(os.path.join(VIS_SAVE_PATH, "prediction_result.csv"), index=False, encoding="utf-8-sig")

    print(f"\n===================== 模型训练全流程完成 =====================")
    print(f"1. 特征文件保存至：{FEATURE_SAVE_PATH}")
    print(f"2. 训练模型保存至：{MODEL_SAVE_PATH}")
    print(f"3. 可视化结果保存至：{VIS_SAVE_PATH}")
    print(f"4. 带STEP文件路径的预测结果保存至：{os.path.join(VIS_SAVE_PATH, 'prediction_result.csv')}")