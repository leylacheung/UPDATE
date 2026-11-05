# PolymerBenchmark2026 — 四大架构图

> 基于上传的 PolymerBenchmark2026-main.zip 静态源码。实线表示经源码文件/函数定位的计算路径；示意节点不等于动态 profiler 调用栈。外部 Chemprop 代码不在 ZIP 内。

## 1. 源码模块及关键函数调用图

```mermaid
flowchart TD
  A[Polymer_Model/train.py] --> B[chemprop.train.cross_validate]
  B --> C[run_training]
  C --> D[get_data / split_data]
  D --> E[MoleculeDataset / MoleculeDataLoader]
  E --> F[BatchMolGraph]
  F --> G[MPNEncoder.forward]
  C --> H[MoleculeModel]
  H --> G
  G --> I[Readout]
  I --> J[FFN]
  C --> K[train.train]
  K --> H
  C --> L[evaluate]
  L --> M[predict]
  M --> H
  C --> N[Checkpoint]
```

## 2. GNN 张量形状及消息传递图

```mermaid
flowchart TD
  A[SMILES] --> B[MolGraph]
  B --> C[BatchMolGraph]
  C --> D[Atom f_atoms: Na × Fa]
  C --> E[Directed bond f_bonds: Nb × Fb]
  C --> F[a2b / b2a / b2revb]
  E --> G[W_i: Nb × H]
  F --> H[Gather neighbor bond messages]
  G --> H
  H --> I[Sum incoming - reverse message]
  I --> J[W_h + activation + dropout]
  J -->|depth-1 iterations| H
  J --> K[Atom aggregation: Na × H]
  D --> L[Concat: Na × Fa+H]
  K --> L
  L --> M[W_o: Na × H]
  M --> N[Mean / Sum / Norm readout]
  N --> O[Graph embedding: B × H]
  O --> P[FFN]
  P --> Q[Prediction: B × T]
```

*符号：Na/Nb 是含 padding 的批图原子/有向键数；Fa/Fb 为运行时特征维度；H=hidden_size；B=batch 图数量；T=num_tasks。若启用 atom_messages 或额外 descriptors，路径和维度会发生变化。*

## 3. 数据处理—训练—预测全流程

```mermaid
flowchart TD
  A[CSV / JSONL] --> B[Dataset construction scripts]
  B --> C[PSMILES / wPSMILES]
  C --> D[Feature conversion / featurization]
  D --> E[Dataset]
  E --> F[Batch / BatchMolGraph]
  F --> G[Forward]
  G --> H[Loss]
  H --> I[Backward + optimizer]
  I --> G
  G --> J[Validation]
  J --> K[Best checkpoint]
  K --> L[Load checkpoint]
  L --> M[Test predictions]
  M --> N[RMSE / scaling summary]
```

## 4. 与3D多层级聚合物表征的接口映射（建议方案，非仓库现有实现）

```mermaid
flowchart TD
  A[Sample ID + PSMILES] --> B[3D conformer + DP + sequence]
  B --> C[Atom 3D GNN + long-range MP]
  B --> D[Functional-group encoder + FG pairs]
  B --> E[Repeat-unit encoder + unit pairs]
  C --> F[Atom↔Unit enhancement]
  E --> F
  F --> G[Enhanced Atom and Unit]
  G --> H[Multi-level fusion]
  D --> H
  H --> I[Chain readout + prediction head]
  I --> J[Same benchmark splits + RMSE]
```

详尽的72个Python文件、468个函数/类/方法及源码行号和静态调用索引，见配套Word/PDF文档。
