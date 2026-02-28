# -------------------------- 完整依赖导入（新增FAISS） --------------------------
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import open3d as o3d
import faiss  # 新增FAISS依赖
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 全局配置（与训练脚本保持一致）
FREECAD_PATH = r"D:\FreeCAD"
POINT_NUM = 1024
FEATURE_SAVE_PATH = r"./step_features"
MODEL_SAVE_PATH = r"./trained_models"
VIS_SAVE_PATH = r"./visualization"
# 新增FAISS配置
FAISS_INDEX_PATH = os.path.join(MODEL_SAVE_PATH, "step_feature_faiss_index.index")  # FAISS索引保存路径
FAISS_NLIST = 100  # IVF索引聚类数（经验值：样本数^(1/3)，如10万样本设为50）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FAISS设备自动选择
FAISS_DEVICE = faiss.StandardGpuResources() if torch.cuda.is_available() else None

os.environ["PATH"] += os.pathsep + os.path.join(FREECAD_PATH, "bin")
sys.path.append(os.path.join(FREECAD_PATH, "bin"))


# -------------------------- 复用原有核心类/函数（完全不变） --------------------------
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
        global_feat = global_feat.view(batch_size, -1)
        return global_feat


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


# -------------------------- 基于FAISS的相似性匹配核心类（改造后） --------------------------
class STEPSimilarityMatcher:
    """
    STEP文件相似性匹配测试类（FAISS优化版）
    核心：基于预训练PointNet提取特征，FAISS实现高效近邻检索，LightGBM辅助类别得分，按综合得分降序排列
    优化点：将原O(N)线性检索改为FAISS近邻检索，支持百万级特征库毫秒级响应
    """

    def __init__(self):
        """初始化：加载预训练模型、类别映射、净化后特征库，构建/加载FAISS索引"""
        self.class2idx = None
        self.idx2class = None
        self.pointnet_model = None
        self.lgb_model = None
        self.clean_feats = None  # 净化后的特征库 (N, 1024)
        self.clean_labels = None  # 特征库对应标签 (N,)
        self.clean_file_paths = None  # 特征库对应STEP文件路径 (N,)
        self.num_classes = 0
        self.faiss_index = None  # FAISS检索索引

        # 加载所有预训练资源
        self._load_class_mapping()
        self._load_pointnet_model()
        self._load_lgb_model()
        self._load_clean_feature_lib()
        # 核心：构建或加载FAISS索引
        self._build_or_load_faiss_index()
        print(f"[INIT] 相似性匹配器（FAISS版）初始化完成")
        print(f"[INIT] 类别数：{self.num_classes} | 特征库样本数：{len(self.clean_feats)} | 运行设备：{DEVICE}")
        print(f"[INIT] FAISS检索设备：{'GPU' if FAISS_DEVICE else 'CPU'} | 索引路径：{FAISS_INDEX_PATH}")

    def _load_class_mapping(self):
        class2idx_path = os.path.join(MODEL_SAVE_PATH, "class2idx.npy")
        if not os.path.exists(class2idx_path):
            raise FileNotFoundError(f"类别映射文件不存在：{class2idx_path}，请先执行训练脚本")
        self.class2idx = np.load(class2idx_path, allow_pickle=True).item()
        self.idx2class = {v: k for k, v in self.class2idx.items()}
        self.num_classes = len(self.class2idx)

    def _load_pointnet_model(self):
        pointnet_path = os.path.join(MODEL_SAVE_PATH, "pointnet_best.pth")
        if not os.path.exists(pointnet_path):
            raise FileNotFoundError(f"PointNet模型文件不存在：{pointnet_path}，请先执行训练脚本")
        self.pointnet_model = PointNetClassifier(self.num_classes).to(DEVICE)
        self.pointnet_model.load_state_dict(torch.load(pointnet_path, map_location=DEVICE))
        self.pointnet_model.eval()

    def _load_lgb_model(self):
        # 注意：与训练脚本保持一致，若训练时保存为.bin则修改为lightgbm_best.bin
        lgb_path = os.path.join(MODEL_SAVE_PATH, "lightgbm_best.txt")
        if not os.path.exists(lgb_path):
            lgb_path = os.path.join(MODEL_SAVE_PATH, "lightgbm_best.bin")
            if not os.path.exists(lgb_path):
                raise FileNotFoundError(f"LightGBM模型文件不存在：{lgb_path}，请先执行训练脚本")
        self.lgb_model = lgb.Booster(model_file=lgb_path)

    def _load_clean_feature_lib(self):
        clean_feat_path = os.path.join(FEATURE_SAVE_PATH, "clean_pointnet_feat.npy")
        clean_label_path = os.path.join(FEATURE_SAVE_PATH, "clean_labels.npy")
        clean_file_paths_path = os.path.join(FEATURE_SAVE_PATH, "clean_file_paths.npy")
        if not all(os.path.exists(p) for p in [clean_feat_path, clean_label_path, clean_file_paths_path]):
            raise FileNotFoundError("净化特征/标签/路径文件缺失，请先执行训练脚本的特征净化步骤")

        self.clean_feats = np.load(clean_feat_path).astype(np.float32)  # FAISS要求float32
        self.clean_labels = np.load(clean_label_path)
        self.clean_file_paths = np.load(clean_file_paths_path, allow_pickle=True).tolist()

        if len(self.clean_file_paths) != len(self.clean_feats):
            raise ValueError(f"路径数量与特征数量不匹配！路径：{len(self.clean_file_paths)}，特征：{len(self.clean_feats)}")

    def _build_or_load_faiss_index(self):
        """
        构建/加载FAISS检索索引
        策略：若索引文件存在则直接加载，否则基于净化特征库构建并保存
        索引类型：IVF_FLAT + L2（归一化后L2等价于余弦相似度），支持GPU/CPU
        """
        # 特征归一化（关键：FAISS中L2距离在归一化后与余弦相似度等价，且检索速度更快）
        feats_normalized = self.clean_feats / (np.linalg.norm(self.clean_feats + 1e-8, axis=1, keepdims=True)).astype(
            np.float32)

        if os.path.exists(FAISS_INDEX_PATH):
            # 加载已有索引
            print(f"[FAISS] 加载预构建索引：{FAISS_INDEX_PATH}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            # 构建新索引
            print(f"[FAISS] 构建IVF_FLAT索引（nlist={FAISS_NLIST}）...")
            feature_dim = self.clean_feats.shape[1]  # 1024

            # 初始化索引：IVF_FLAT（适合中等/大规模特征库）
            cpu_index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(feature_dim),  # 基础索引：L2距离
                feature_dim,
                min(FAISS_NLIST, len(feats_normalized) // 10),  # 聚类数不超过样本数的1/10
                faiss.METRIC_L2
            )
            # 训练索引（IVF类索引必须先训练）
            cpu_index.train(feats_normalized)
            # 添加特征到索引
            cpu_index.add(feats_normalized)
            # 设置检索参数：nprobe=10（检索时遍历的聚类数，值越大精度越高、速度越慢，经验值10-50）
            cpu_index.nprobe = 10

            self.faiss_index = cpu_index
            # 保存索引到文件
            faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
            print(f"[FAISS] 索引构建完成并保存至：{FAISS_INDEX_PATH}")

        # 若有GPU，将索引移至GPU加速
        if FAISS_DEVICE:
            self.faiss_index = faiss.index_cpu_to_gpu(FAISS_DEVICE, 0, self.faiss_index)
            print(f"[FAISS] 索引已移至GPU加速")

    def extract_single_feature(self, step_file_path):
        """提取单个STEP文件的PointNet全局特征（与训练时一致，返回float32）"""
        point_cloud = step2point_cloud(step_file_path)
        if point_cloud is None:
            return None
        point_tensor = torch.from_numpy(point_cloud).transpose(0, 1).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            _, feat = self.pointnet_model(point_tensor)
        feat_np = feat.cpu().numpy().squeeze().astype(np.float32)  # FAISS要求float32
        return feat_np

    def _faiss_k_nearest_search(self, query_feat, top_k):
        """
        FAISS近邻检索：返回Top-K相似特征的索引和相似度
        :param query_feat: 待匹配特征 (1024,) float32
        :param top_k: 检索前K个近邻
        :return: (distances, indices) - 距离数组、特征库索引数组
        注意：归一化后L2距离越小，相似度越高，转换为余弦相似度范围[0,1]
        """
        # 特征归一化（与索引保持一致）
        query_normalized = query_feat / (np.linalg.norm(query_feat + 1e-8)).astype(np.float32)
        query_2d = query_normalized.reshape(1, -1)  # FAISS要求2D输入

        # FAISS检索：返回 (距离数组, 索引数组)，形状均为(1, top_k)
        distances, indices = self.faiss_index.search(query_2d, min(top_k, len(self.clean_feats)))

        # 转换L2距离为余弦相似度（归一化后：cos_sim = 1 - L2^2 / 2），范围[0,1]
        cos_sim = 1 - (distances ** 2) / 2

        return cos_sim.squeeze(), indices.squeeze()  # 降维为一维数组

    def match_single_file(self, step_file_path, top_k=5, save_result=False):
        """
        单STEP文件相似性匹配（FAISS优化版）：返回Top-K相似结果，完全兼容原结果格式
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
        query_feat_2d = query_feat.reshape(1, -1)

        # 步骤2：FAISS高效检索Top-K近邻（核心优化点，替换原线性计算）
        cos_sim, top_k_indices = self._faiss_k_nearest_search(query_feat, top_k)
        # 筛选特征库中对应的Top-K数据
        top_k_feats = self.clean_feats[top_k_indices]
        top_k_labels = self.clean_labels[top_k_indices]
        top_k_file_paths = [self.clean_file_paths[i] for i in top_k_indices]

        # 步骤3：计算LightGBM类别得分
        lgb_score = self.lgb_model.predict(query_feat_2d, num_iteration=self.lgb_model.best_iteration)[0]
        top_k_lgb_scores = [lgb_score[idx] for idx in top_k_labels]

        # 步骤4：计算综合得分（与原逻辑完全一致，归一化后加权）
        lgb_score_norm = (lgb_score - lgb_score.min()) / (lgb_score.max() - lgb_score.min() + 1e-8)
        cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        top_k_combined = 0.4 * lgb_score_norm[top_k_labels] + 0.6 * cos_sim_norm

        # 步骤5：构造结果数据（与原格式完全一致）
        result_data = {
            "匹配文件路径": top_k_file_paths,
            "相似类别": [self.idx2class[idx] for idx in top_k_labels],
            "余弦相似度": cos_sim.round(4),
            "LGBM类别得分": np.array(top_k_lgb_scores).round(4),
            "综合得分": top_k_combined.round(4)
        }
        result_df = pd.DataFrame(result_data).reset_index(drop=True)

        # 保存结果
        if save_result:
            save_name = f"FAISS相似性匹配结果_{os.path.basename(step_file_path)}.csv"
            save_path = os.path.join(VIS_SAVE_PATH, save_name)
            result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"[SAVE] 匹配结果已保存至：{save_path}")

        # 打印结果
        print(f"[RESULT] FAISS检索前{top_k}个相似结果（按综合得分降序）：")
        print(result_df.to_string(index=False))
        return result_df

    def match_batch_files(self, step_dir, top_k=5, save_result=True):
        """
        批量匹配文件夹下的所有STEP文件（FAISS优化版），完全兼容原调用和结果格式
        :param step_dir: STEP文件所在文件夹
        :param top_k: 每个文件返回前K个相似结果，默认5
        :param save_result: 是否保存批量结果到CSV，默认True
        :return: DataFrame(所有文件的匹配结果)
        """
        if not os.path.isdir(step_dir):
            raise NotADirectoryError(f"无效的文件夹路径：{step_dir}")

        step_files = [os.path.join(step_dir, f) for f in os.listdir(step_dir)
                      if f.lower().endswith((".step", ".stp"))]
        if len(step_files) == 0:
            raise FileNotFoundError(f"文件夹{step_dir}下未找到STEP/STP文件")

        print(f"\n[BATCH MATCH] FAISS批量匹配开始，共{len(step_files)}个STEP文件")
        batch_result = []
        for step_file in tqdm(step_files, desc="FAISS批量匹配进度"):
            try:
                single_result = self.match_single_file(step_file, top_k=top_k, save_result=False)
                single_result["待匹配文件"] = os.path.basename(step_file)
                batch_result.append(single_result)
            except Exception as e:
                print(f"[SKIP] 跳过文件{os.path.basename(step_file)}：{str(e)[:50]}...")
                continue

        if len(batch_result) == 0:
            raise RuntimeError("FAISS批量匹配无有效结果")

        final_batch_df = pd.concat(batch_result, ignore_index=True)
        col_order = ["待匹配文件", "匹配文件路径", "相似类别", "余弦相似度", "LGBM类别得分", "综合得分"]
        final_batch_df = final_batch_df[col_order]

        if save_result:
            save_path = os.path.join(VIS_SAVE_PATH, "FAISS批量相似性匹配结果.csv")
            final_batch_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"\n[SAVE] FAISS批量匹配结果已保存至：{save_path}")

        print(f"\n[BATCH DONE] FAISS批量匹配完成，有效匹配{len(batch_result)}个文件")
        return final_batch_df


# -------------------------- 测试示例（与原代码完全一致） --------------------------
if __name__ == "__main__":
    # 1. 初始化FAISS版相似性匹配器（自动加载/构建索引）
    matcher = STEPSimilarityMatcher()

    # 2. 单文件相似性匹配（替换为你的测试文件）
    test_step_file = r"D:\graduate\cad\pointnet\Opencascade\modesearch\testdata\0\国家标准GB_GB_T19066.2-2020A11508.step"
    matcher.match_single_file(test_step_file, top_k=10, save_result=True)

    # 3. 批量文件相似性匹配（替换为你的测试文件夹）
    # test_step_dir = r"D:\graduate\cad\pointnet\Opencascade\modesearch\testdata"
    # matcher.match_batch_files(test_step_dir, top_k=5, save_result=True)