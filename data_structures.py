# data_structures.py
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple # Import types here

@dataclass
class CellData:
    """存储单个细胞的数据结构"""
    cell_id: int
    channel1: np.ndarray  # 第一个通道的细胞区域数据
    channel2: np.ndarray  # 第二个通道的细胞区域数据
    file_path: str
    time_point: float  # 真实时间(秒),而非索引
    x_coords: np.ndarray  # 细胞在原图中的x坐标
    y_coords: np.ndarray  # 细胞在原图中的y坐标
    # Note: pearson_corr and p_value are calculated elsewhere now
    intensity1: np.ndarray = field(init=False)  # 展平的第一个通道强度
    intensity2: np.ndarray = field(init=False)  # 展平的第二个通道强度
    # pearson_corr: float = field(init=False, default=np.nan)  # Removed from __post_init__
    # p_value: float = field(init=False, default=np.nan)
    n_pixels: int = field(init=False)

    def __post_init__(self):
        """初始化后处理"""
        # 展平强度数据
        self.intensity1 = self.channel1.flatten()
        self.intensity2 = self.channel2.flatten()
        self.n_pixels = len(self.intensity1)
        # Note: Correlation calculation is now external

@dataclass
class FileData:
    """时间序列分析的数据结构"""
    file_path: str
    time_points: int
    cells: Dict[int, List[CellData]] = field(default_factory=dict)
    all_cells: List[CellData] = field(default_factory=list)  # 所有时间点的所有细胞
    skip_initial_frames: int = field(default=0)  # 跳过的初始帧数
    max_frames: Optional[int] = field(default=None)  # 最大分析帧数（None表示不限制）
    original_nd2_path: Optional[str] = field(default=None)  # 存储原始ND2路径
    position_index: Optional[int] = field(default=None)  # 拆分视野的P索引
    
    def is_frame_in_range(self, frame_idx: int) -> bool:
        """判断帧索引是否在分析范围内"""
        if frame_idx < self.skip_initial_frames:
            return False
        if self.max_frames is not None and frame_idx >= self.max_frames:
            return False
        return True

    def add_cell_data(self, cell_data: CellData):
        """添加细胞数据"""
        if cell_data.cell_id not in self.cells:
            self.cells[cell_data.cell_id] = []
        self.cells[cell_data.cell_id].append(cell_data)
        self.all_cells.append(cell_data)

