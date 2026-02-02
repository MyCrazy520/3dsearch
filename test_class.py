import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # 修复核心：添加nn别名
import lightgbm as lgb
import open3d as o3d
from tqdm import tqdm

# ===================== 配置项（与训练代码保持一致，无需修改） =====================
FREECAD_PATH = r"D:\FreeCAD"
POINT_NUM = 1024  # 与训练一致的点云采样数
FEATURE_SAVE_PATH = r"./step_features"
MODEL_SAVE_PATH = r"./trained_models"
VIS_SAVE_PATH = r"./visualization"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PATH"] += os.pathsep + os.path.join(FREECAD_PATH, "bin")
sys.path.append(os.path.join(FREECAD_PATH, "bin"))

# 忽略无关警告
import warnings

warnings.filterwarnings('ignore')


# ===================== 1. 复用训练代码的核心函数（点云提取+模型定义） =====================
def step2point_cloud(step_file_path, num_points=POINT_NUM):
    """从STEP文件提取并标准化点云（与训练代码完全一致，保证特征统一）"""
    try:
        import FreeCAD
        import Part
        FreeCAD.Console.PrintLog = lambda *args: None  # 屏蔽FreeCAD日志

        # 加载STEP文件并转换为形状
        doc = FreeCAD.newDocument("temp")
        shape = Part.read(step_file_path)
        if shape is None:
            return None

        # 离散化为点云并预处理
        mesh = shape.tessellate(0.1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh[0])
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        points = np.asarray(pcd.points)

        # 补零/采样到指定点数
        if len(points) < num_points:
            points = np.pad(points, ((0, num_points - len(points)), (0, 0)), mode='constant')
        else:
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]

        # 标准化（中心化+单位球归一化，与训练一致）
        points = points - np.mean(points, axis=0)
        points = points / np.max(np.linalg.norm(points, axis=1))
        return points.astype(np.float32)

    except Exception as e:
        print(f"[错误] 处理{os.path.basename(step_file_path)}失败：{str(e)[:50]}...")
        return None
    finally:
        try:
            FreeCAD.closeDocument("temp")
        except:
            pass


# ===================== 2. 复用PointNet模型定义（特征提取用） =====================
class TNet(nn.Module):
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
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + iden
        return x.view(batch_size, self.k, self.k)


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=1024):
        super(PointNetFeatureExtractor, self).__init__()
        self.tnet3 = TNet(k=3)
        self.tnet64 = TNet(k=64)
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
        trans3 = self.tnet3(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans3)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        trans64 = self.tnet64(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans64)
        x = x.transpose(2, 1)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        global_feat = torch.max(x, 2, keepdim=True)[0]
        return global_feat.view(batch_size, -1)


class PointNetClassifier(nn.Module):
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
        global_feat = self.feature_extractor(x)
        x = self.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, global_feat


# ===================== 3. 核心测试类定义 =====================
class STEPTypePredictor:
    """STEP文件类型预测器：加载训练好的模型，实现单/多文件预测"""

    def __init__(self):
        # 初始化加载所有训练资源
        self.class2idx = self._load_class2idx()  # 类别->数字映射
        self.idx2class = {v: k for k, v in self.class2idx.items()}  # 数字->类别映射
        self.num_classes = len(self.class2idx)
        self.pointnet_model = self._load_pointnet()  # 加载PointNet特征提取器
        self.lgb_model = self._load_lightgbm()  # 加载LightGBM分类器
        print(f"[初始化完成] 加载{self.num_classes}个类别映射，模型设备：{DEVICE}")

    def _load_class2idx(self):
        """加载训练阶段保存的类别映射文件"""
        class2idx_path = os.path.join(MODEL_SAVE_PATH, "class2idx.npy")
        if not os.path.exists(class2idx_path):
            raise FileNotFoundError(f"未找到类别映射文件：{class2idx_path}\n请先运行训练代码生成模型")
        return np.load(class2idx_path, allow_pickle=True).item()

    def _load_pointnet(self):
        """加载训练好的PointNet模型（仅用于提取特征）"""
        pointnet_path = os.path.join(MODEL_SAVE_PATH, "pointnet_best.pth")
        if not os.path.exists(pointnet_path):
            raise FileNotFoundError(f"未找到PointNet模型：{pointnet_path}\n请先运行训练代码生成模型")
        # 初始化模型并加载权重
        model = PointNetClassifier(self.num_classes).to(DEVICE)
        model.load_state_dict(torch.load(pointnet_path, map_location=DEVICE))
        model.eval()  # 推理模式（关闭Dropout/BatchNorm训练模式）
        return model

    def _load_lightgbm(self):
        """加载训练好的LightGBM模型"""
        lgb_path = os.path.join(MODEL_SAVE_PATH, "lightgbm_best.txt")
        if not os.path.exists(lgb_path):
            raise FileNotFoundError(f"未找到LightGBM模型：{lgb_path}\n请先运行训练代码生成模型")
        return lgb.Booster(model_file=lgb_path)

    def _extract_pointnet_feat(self, point_cloud):
        """从单一点云提取PointNet全局特征（与训练一致）"""
        if point_cloud is None:
            return None
        # 转换为PointNet输入格式：(1, 3, 1024) [batch, channel, points]
        point_tensor = torch.from_numpy(point_cloud).transpose(1, 0).unsqueeze(0).to(DEVICE).float()
        # 推理提取特征（无梯度计算，提升速度）
        with torch.no_grad():
            _, global_feat = self.pointnet_model(point_tensor)
        return global_feat.cpu().numpy()[0]  # 输出(1024,)特征向量

    def predict_single(self, step_file_path):
        """
        预测单个STEP文件的类型
        :param step_file_path: STEP/STP文件完整路径
        :return: dict - {文件路径, 预测类别, 预测概率, 状态}，失败返回None
        """
        if not os.path.exists(step_file_path):
            print(f"[错误] 文件不存在：{step_file_path}")
            return None
        if not step_file_path.lower().endswith((".step", ".stp")):
            print(f"[错误] 非STEP文件：{os.path.basename(step_file_path)}")
            return None

        # 步骤1：提取点云
        point_cloud = step2point_cloud(step_file_path)
        if point_cloud is None:
            return None

        # 步骤2：提取PointNet特征
        feat = self._extract_pointnet_feat(point_cloud)
        if feat is None:
            return None

        # 步骤3：LightGBM预测（含概率）
        pred_proba = self.lgb_model.predict(feat.reshape(1, -1))[0]
        pred_idx = np.argmax(pred_proba)
        pred_class = self.idx2class[pred_idx]
        pred_score = round(float(pred_proba[pred_idx]), 4)

        # 返回预测结果
        return {
            "文件路径": step_file_path,
            "预测类别": pred_class,
            "预测概率": pred_score,
            "状态": "成功"
        }

    def predict_batch(self, step_file_list, save_result=True):
        """
        批量预测STEP文件类型
        :param step_file_list: STEP文件路径列表
        :param save_result: 是否将结果保存为CSV（默认True，保存至visualization目录）
        :return: pd.DataFrame - 所有文件的预测结果
        """
        if not isinstance(step_file_list, list) or len(step_file_list) == 0:
            raise ValueError("输入必须为非空的文件路径列表")

        # 批量预测（带进度条）
        results = []
        for file_path in tqdm(step_file_list, desc="批量预测STEP文件"):
            res = self.predict_single(file_path)
            if res is None:
                res = {
                    "文件路径": file_path,
                    "预测类别": "失败",
                    "预测概率": 0.0,
                    "状态": "失败"
                }
            results.append(res)

        # 转换为DataFrame
        result_df = pd.DataFrame(results)
        # 统计结果
        success_num = len(result_df[result_df["状态"] == "成功"])
        fail_num = len(result_df) - success_num
        print(f"\n[批量预测完成] 总文件：{len(step_file_list)} | 成功：{success_num} | 失败：{fail_num}")

        # 保存结果到CSV
        if save_result:
            if not os.path.exists(VIS_SAVE_PATH):
                os.makedirs(VIS_SAVE_PATH)
            save_path = os.path.join(VIS_SAVE_PATH, "step_prediction_result_test.csv")
            result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"[结果保存] 预测结果已保存至：{save_path}")

        return result_df


# ===================== 4. 测试示例（直接运行即可，已实现递归查询） =====================
if __name__ == "__main__":
    # 初始化预测器（自动加载所有模型）
    predictor = STEPTypePredictor()

    # -------------------- 批量预测（递归查询目标文件夹所有子目录的STEP文件） --------------------
    test_dir = r"D:\graduate\cad\pointnet\Opencascade\modesearch\testdata"  # 你的测试根文件夹
    batch_file_list = []

    if os.path.exists(test_dir):
        # 核心：os.walk递归遍历根文件夹+所有子文件夹
        # root：当前遍历的文件夹路径，dirs：当前文件夹下的子文件夹列表，files：当前文件夹下的文件列表
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                # 筛选.step/.stp后缀文件（忽略大小写）
                if file.lower().endswith((".step", ".stp")):
                    full_file_path = os.path.join(root, file)  # 拼接文件完整路径
                    batch_file_list.append(full_file_path)
        print(f"[文件检索完成] 递归找到 {len(batch_file_list)} 个STEP/STP文件")
    else:
        print(f"[错误] 测试文件夹不存在：{test_dir}")

    # 执行批量预测
    if batch_file_list:
        batch_result = predictor.predict_batch(batch_file_list, save_result=True)
        # 打印前5条结果
        print("\n[批量预测前5条结果]")
        print(batch_result.head())