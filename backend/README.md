# Drug Screening System Backend API

基于 FastAPI 开发的药物筛选系统后端接口。该系统集成了深度学习模型（如 MLP）和化学信息学工具，提供药物分子属性预测和大规模虚拟筛选功能。

## 项目结构

```
backend/
├── main.py                 # 服务启动入口
└── app/
    ├── main.py             # FastAPI 应用定义与路由挂载
    ├── core/               
    │   └── config.py       # 全局配置、路径管理与日志
    ├── models/             
    │   └── schemas.py      # Pydantic 数据校验模型
    ├── services/           
    │   ├── ml_service.py   # 模型加载与推理服务 (Singleton)
    │   └── chemistry.py    # 化学信息学工具 (Lipinski's Rule)
    └── api/                
        └── routers/        # API 端点定义 (Health, Predict, Screen)
```

## 核心功能

1.  **单分子预测 (`/predict`)**：
    - 输入：SMILES 字符串。
    - 输出：分子活性概率、物理化学属性（分子量、LogP、TPSA等）以及是否符合 Lipinski 五规则。
2.  **批量筛选 (`/screen`)**：
    - 支持大规模 SMILES 列表输入。
    - 结合深度学习评分与化学规则过滤，返回 Top-K 候选药物分子。
3.  **健康检查 (`/health`)**：
    - 监控模型加载状态与计算设备 (CPU/CUDA)。

## 快速开始

### 运行环境
确保已安装 `requirements.txt` 中的依赖，特别是 `fastapi`, `uvicorn`, `torch`, `rdkit` 和 `pandas`。

### 启动服务
在项目根目录下运行：
```bash
python backend/main.py
```
服务默认启动在 `http://0.0.0.0:8000`。

### 接口文档
启动后可访问以下地址查看交互式 API 文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 技术栈
- **框架**: FastAPI
- **推理**: PyTorch
- **化学信息学**: RDKit
- **数据处理**: Pandas
- **服务器**: Uvicorn
