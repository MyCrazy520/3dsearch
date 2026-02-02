# -------------------------- 完整依赖导入（解决所有未定义错误） --------------------------
import os
import sys
import numpy as np
import pandas as pd
import torch  # PyTorch核心库
import torch.nn as nn  # PyTorch神经网络模块（解决nn未定义）
import lightgbm as lgb
import open3d as o3d
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 全局配置（与训练脚本保持一致）
FREECAD_PATH = r"D:\FreeCAD"
POINT_NUM = 1024
FEATURE_SAVE_PATH = r"./step_features"
MODEL_SAVE_PATH = r"./trained_models"
VIS_SAVE_PATH = r"./visualization"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PATH"] += os.pathsep + os.path.join(FREECAD_PATH, "bin")
sys.path.append(os.path.join(FREECAD_PATH, "bin"))

# -------------------------- 复用训练脚本的核心类/函数（保证逻辑一致） --------------------------
# TNet变换网络（与训练脚本完全一致）
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

# PointNet特征提取器（与训练脚本完全一致）
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
        global_feat = global_feat.view(batch_size, -1)
        return global_feat

# PointNet分类器（仅用于加载预训练权重，与训练脚本完全一致）
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

# STEP转点云（与训练脚本完全一致，保证特征提取一致性）
def step2point_cloud(step_file_path, num_points=POINT_NUM):
    try:
        import FreeCAD
        import Part
        FreeCAD.Console.PrintLog = lambda *args: None
        doc = FreeCAD.newDocument("temp")
        shape = Part.read(step_file_path)
        if shape is None:
            return None
        mesh = shape.tessellate(0.1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh[0])
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        points = np.asarray(pcd.points)
        if len(points) < num_points:
            points = np.pad(points, ((0, num_points - len(points)), (0, 0)), mode='constant')
        else:
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]
        points = points - np.mean(points, axis=0)
        points = points / np.max(np.linalg.norm(points, axis=1))
        return points.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] 提取{os.path.basename(step_file_path)}点云失败: {str(e)}")
        return None
    finally:
        try:
            import FreeCAD
            FreeCAD.closeDocument("temp")
        except:
            pass

# -------------------------- 相似性匹配核心测试类 --------------------------
class STEPSimilarityMatcher:
    """
    STEP文件相似性匹配测试类
    核心：基于预训练PointNet提取特征，LightGBM计算类别匹配得分，按得分降序排列
    """
    def __init__(self):
        """初始化：加载预训练模型、类别映射、净化后特征库"""
        self.class2idx = None  # 类别->数字映射
        self.idx2class = None  # 数字->类别映射（反向）
        self.pointnet_model = None  # PointNet特征提取模型
        self.lgb_model = None  # LightGBM分类模型
        self.clean_feats = None  # 净化后的特征库（用于精准相似性匹配）
        self.clean_labels = None  # 特征库对应标签
        self.clean_file_paths = None  # 特征库对应STEP文件路径
        self.num_classes = 0

        # 加载所有预训练资源
        self._load_class_mapping()
        self._load_pointnet_model()
        self._load_lgb_model()
        self._load_clean_feature_lib()
        print(f"[INIT] 相似性匹配器初始化完成")
        print(f"[INIT] 类别数：{self.num_classes} | 特征库样本数：{len(self.clean_feats)} | 运行设备：{DEVICE}")

    def _load_class_mapping(self):
        """加载类别映射文件（训练时保存的class2idx.npy）"""
        class2idx_path = os.path.join(MODEL_SAVE_PATH, "class2idx.npy")
        if not os.path.exists(class2idx_path):
            raise FileNotFoundError(f"类别映射文件不存在：{class2idx_path}，请先执行训练脚本")
        self.class2idx = np.load(class2idx_path, allow_pickle=True).item()
        self.idx2class = {v: k for k, v in self.class2idx.items()}
        self.num_classes = len(self.class2idx)

    def _load_pointnet_model(self):
        """加载预训练PointNet模型（训练时保存的pointnet_best.pth）"""
        pointnet_path = os.path.join(MODEL_SAVE_PATH, "pointnet_best.pth")
        if not os.path.exists(pointnet_path):
            raise FileNotFoundError(f"PointNet模型文件不存在：{pointnet_path}，请先执行训练脚本")
        # 初始化模型并加载权重
        self.pointnet_model = PointNetClassifier(self.num_classes).to(DEVICE)
        self.pointnet_model.load_state_dict(torch.load(pointnet_path, map_location=DEVICE))
        self.pointnet_model.eval()  # 评估模式，关闭Dropout/BatchNorm训练特性

    def _load_lgb_model(self):
        """加载预训练LightGBM模型（训练时保存的lightgbm_best.txt）"""
        lgb_path = os.path.join(MODEL_SAVE_PATH, "lightgbm_best.txt")
        if not os.path.exists(lgb_path):
            raise FileNotFoundError(f"LightGBM模型文件不存在：{lgb_path}，请先执行训练脚本")
        self.lgb_model = lgb.Booster(model_file=lgb_path)

    def _load_clean_feature_lib(self):
        """加载净化后的特征库、标签、文件路径（训练时特征净化的结果）"""
        # 加载特征和标签
        clean_feat_path = os.path.join(FEATURE_SAVE_PATH, "clean_pointnet_feat.npy")
        clean_label_path = os.path.join(FEATURE_SAVE_PATH, "clean_labels.npy")
        if not os.path.exists(clean_feat_path) or not os.path.exists(clean_label_path):
            raise FileNotFoundError("净化特征文件不存在，请先执行训练脚本的特征净化步骤")
        self.clean_feats = np.load(clean_feat_path)
        self.clean_labels = np.load(clean_label_path)

        # ========== 修改后：加载训练脚本保存的全量净化路径 ==========
        clean_file_paths_path = os.path.join(FEATURE_SAVE_PATH, "clean_file_paths.npy")
        if not os.path.exists(clean_file_paths_path):
            raise FileNotFoundError("全量净化路径文件不存在，请先重新执行训练脚本的特征净化步骤")
        self.clean_file_paths = np.load(clean_file_paths_path, allow_pickle=True).tolist()
        # ===========================================================
        # 最终校验（双重保障，理论上不会触发）
        if len(self.clean_file_paths) != len(self.clean_feats):
            raise ValueError(f"路径数量与特征数量不匹配！路径：{len(self.clean_file_paths)}，特征：{len(self.clean_feats)}")

    def extract_single_feature(self, step_file_path):
        """
        提取单个STEP文件的PointNet全局特征（与训练时一致的特征提取流程）
        :param step_file_path: 单个STEP文件路径
        :return: (1024,) 全局特征数组，失败返回None
        """
        # STEP转标准化点云
        point_cloud = step2point_cloud(step_file_path)
        if point_cloud is None:
            return None
        # 转换为PointNet输入格式 (1, 3, 1024)
        point_tensor = torch.from_numpy(point_cloud).transpose(0, 1).unsqueeze(0).to(DEVICE).float()
        # 提取全局特征（关闭梯度计算）
        with torch.no_grad():
            _, feat = self.pointnet_model(point_tensor)
        # 转换为numpy数组 (1024,)
        return feat.cpu().numpy().squeeze()

    def _calculate_cosine_similarity(self, query_feat, feat_lib):
        """
        计算余弦相似度（衡量特征相似性，值越大越相似，范围[-1,1]）
        :param query_feat: 查询特征 (D,)
        :param feat_lib: 特征库 (N, D)
        :return: (N,) 余弦相似度数组
        """
        # 归一化特征（避免尺度影响）
        query_feat = query_feat / np.linalg.norm(query_feat + 1e-8)
        feat_lib = feat_lib / (np.linalg.norm(feat_lib + 1e-8, axis=1, keepdims=True))
        # 计算余弦相似度
        similarity = np.dot(feat_lib, query_feat)
        return similarity

    def match_single_file(self, step_file_path, top_k=5, save_result=False):
        """
        单STEP文件相似性匹配：返回Top-K相似的STEP文件/类别，按得分降序排列
        :param step_file_path: 待匹配的STEP文件路径
        :param top_k: 返回前K个相似结果，默认5
        :param save_result: 是否保存匹配结果到CSV，默认False
        :return: DataFrame(相似结果)，含列：匹配文件路径、相似类别、余弦相似度、LGBM得分、综合得分
        """
        if not os.path.exists(step_file_path) or not step_file_path.lower().endswith((".step", ".stp")):
            raise ValueError(f"无效的STEP文件路径：{step_file_path}")

        print(f"\n[MATCH] 开始匹配文件：{os.path.basename(step_file_path)}")
        # 步骤1：提取待匹配文件的特征
        query_feat = self.extract_single_feature(step_file_path)
        if query_feat is None:
            raise RuntimeError(f"无法提取{step_file_path}的特征，匹配终止")
        query_feat_2d = query_feat.reshape(1, -1)  # LGBM输入格式

        # 步骤2：计算两类得分
        # 2.1 LightGBM类别匹配得分（概率，对应各类别的相似性）
        lgb_score = self.lgb_model.predict(query_feat_2d, num_iteration=self.lgb_model.best_iteration)[0]
        # 2.2 余弦相似度（与特征库中每个样本的精准相似性）
        cos_sim = self._calculate_cosine_similarity(query_feat, self.clean_feats)
        # 2.3 综合得分（归一化后加权，兼顾类别概率和精准特征相似性，权重可调整）
        lgb_score_norm = (lgb_score - lgb_score.min()) / (lgb_score.max() - lgb_score.min() + 1e-8)
        cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        combined_score = 0.4 * lgb_score_norm[self.clean_labels] + 0.6 * cos_sim_norm  # 余弦相似度权重更高

        # 步骤3：构造结果数据
        result_data = {
            "匹配文件路径": self.clean_file_paths,
            "相似类别": [self.idx2class[idx] for idx in self.clean_labels],
            "余弦相似度": cos_sim,
            "LGBM类别得分": [lgb_score[idx] for idx in self.clean_labels],
            "综合得分": combined_score
        }
        result_df = pd.DataFrame(result_data)

        # 步骤4：按综合得分降序排列，取Top-K
        result_df_sorted = result_df.sort_values(by="综合得分", ascending=False).head(top_k).reset_index(drop=True)
        # 保留4位小数，提升可读性
        for col in ["余弦相似度", "LGBM类别得分", "综合得分"]:
            result_df_sorted[col] = result_df_sorted[col].round(4)

        # 步骤5：保存结果（若需要）
        if save_result:
            save_path = os.path.join(VIS_SAVE_PATH, f"相似性匹配结果_{os.path.basename(step_file_path)}.csv")
            result_df_sorted.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"[SAVE] 匹配结果已保存至：{save_path}")

        # 打印匹配结果
        print(f"[RESULT] 前{top_k}个相似结果（按综合得分降序）：")
        print(result_df_sorted.to_string(index=False))
        return result_df_sorted

    def match_batch_files(self, step_dir, top_k=5, save_result=True):
        """
        批量匹配文件夹下的所有STEP文件
        :param step_dir: STEP文件所在文件夹
        :param top_k: 每个文件返回前K个相似结果，默认5
        :param save_result: 是否保存批量结果到CSV，默认True
        :return: DataFrame(所有文件的匹配结果)
        """
        if not os.path.isdir(step_dir):
            raise NotADirectoryError(f"无效的文件夹路径：{step_dir}")

        # 遍历文件夹下所有STEP文件
        step_files = [os.path.join(step_dir, f) for f in os.listdir(step_dir)
                      if f.lower().endswith((".step", ".stp"))]
        if len(step_files) == 0:
            raise FileNotFoundError(f"文件夹{step_dir}下未找到STEP/STP文件")

        print(f"\n[BATCH MATCH] 开始批量匹配，共{len(step_files)}个STEP文件")
        batch_result = []
        for step_file in tqdm(step_files, desc="批量匹配进度"):
            try:
                # 单文件匹配
                single_result = self.match_single_file(step_file, top_k=top_k, save_result=False)
                # 增加待匹配文件列
                single_result["待匹配文件"] = os.path.basename(step_file)
                batch_result.append(single_result)
            except Exception as e:
                print(f"[SKIP] 跳过文件{os.path.basename(step_file)}：{str(e)}")
                continue

        # 合并批量结果
        if len(batch_result) == 0:
            raise RuntimeError("批量匹配无有效结果")
        final_batch_df = pd.concat(batch_result, ignore_index=True)
        # 调整列顺序，更直观
        col_order = ["待匹配文件", "匹配文件路径", "相似类别", "余弦相似度", "LGBM类别得分", "综合得分"]
        final_batch_df = final_batch_df[col_order]

        # 保存批量结果
        if save_result:
            save_path = os.path.join(VIS_SAVE_PATH, "批量相似性匹配结果.csv")
            final_batch_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"\n[SAVE] 批量匹配结果已保存至：{save_path}")

        print(f"\n[BATCH DONE] 批量匹配完成，有效匹配{len(batch_result)}个文件")
        return final_batch_df

# -------------------------- 测试示例 --------------------------
if __name__ == "__main__":
    # 1. 初始化相似性匹配器
    matcher = STEPSimilarityMatcher()

    # 2. 单文件相似性匹配（示例：替换为你的测试STEP文件路径）
    test_step_file = r"D:\graduate\cad\pointnet\Opencascade\modesearch\testdata\0\国家标准GB_GB_T19066.2-2020A11508.step"
    matcher.match_single_file(test_step_file, top_k=10, save_result=True)

    # 3. 批量文件相似性匹配（示例：替换为你的STEP文件夹路径）
    # test_step_dir = r"D:\graduate\cad\pointnet\Opencascade\step\标准件\GB国家标准\测试集"
    # matcher.match_batch_files(test_step_dir, top_k=5, save_result=True)