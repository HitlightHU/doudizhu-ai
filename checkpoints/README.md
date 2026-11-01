# 模型卡

[简体中文](README.md) | [English](README_EN.md)

发布版本：**v1.0.0**（2026-11-01）  
权重只保留推理必需的最小 state_dict，不含 optimizer；两个大权重作为 **GitHub Release v1.0.0 附件**，运行 `python scripts/download_weights.py` 下载（支持断点续传 + SHA256 校验）。

## 文件

| 文件 | 用途 | 大小 |
|---|---|---:|
| `best_model.pt` | 常规模式默认模型（FP16 存储，加载为 FP32） | 11.2 MB |
| `best_model_noshuffle.pt` | 不洗牌分析模式专家（FP16 存储，加载为 FP32） | 11.2 MB |
| `bid_net.pt` | 常规模式叫地主先验 | 84 KB |

SHA256：

```text
best_model.pt          C7DDCE6900E7E2243704F9362727E9BDA225C0F316C72D2588308C2B71B7E5E0
best_model_noshuffle.pt 5F702526994AD6A3D61599D45421D2A1EB9EAF02BB9CB995834E37D5D18361ED
bid_net.pt             64B8CA85A625D5F094B01F61874D03059915E6D34775EAF9E8A6CFFB226EFC6F
```

## 常规模型

- 5.61M 参数：4.56M v2 + 1.05M residual adapter，3 检查点 soup；
- 26,064 + PASS 动作；
- 1024 副固定发牌：

| 对手 | 总胜率 | 地主 | 农民 |
|---|---:|---:|---:|
| 随机 | 0.958 | 0.973 | 0.951 |
| 贪心 | 0.814 | 0.814 | 0.813 |
| 启发式（单座） | 0.657 | 0.768 | 0.602 |
| 启发式（整方） | 0.770 | 0.767 | 0.772 |
| 启发式 + oracle 地主 | 0.578 | 0.908 | 0.413 |

## 不洗牌专家

- 从常规 5.61M 继续 80 iter / 1,313,483 局；
- 256 个不洗牌牌局 / 768 局：

| 模型 | 总胜率 | 地主 | 农民 |
|---|---:|---:|---:|
| 常规 adapter | 0.5625 | 0.4375 | 0.6250 |
| **专家** | **0.6549** | **0.5898** | **0.6875** |

整方头对头 512 局：常规 0.3613 vs 专家 **0.6387**。

不洗牌模式使用专家自己的参数做初始叫地主、实时叫地主、前三选、双方胜率、加倍；
**不使用 `bid_net.pt`**。叫地主规则为 **p_landlord > 0.75**。

## 使用

GUI 默认加载：

```text
checkpoints/best_model.pt
checkpoints/best_model_noshuffle.pt
checkpoints/bid_net.pt
```

可用环境变量覆盖：

```powershell
$env:DOUDIZHU_MODEL = "checkpoints/best_model.pt"
$env:DOUDIZHU_MODEL_NOSHUFFLE = "checkpoints/best_model_noshuffle.pt"
$env:DOUDIZHU_BID = "checkpoints/bid_net.pt"
```

## 指标文件

- `model_info.json`：模型卡结构化数据；
- `final_eval.json`：常规评估；
- `noshuffle_eval.json`：不洗牌对照；
- `head_noshuffle.json`：不洗牌整方头对头；
- `latency.json`：anytime 延迟；
- `eval_oracle.txt`：oracle 地主评估输出。

## 限制

没有人类评测；所有胜率都是对 bot/模型。  
不洗牌规则是本地可复现近似，不是官方服务端逐字节复刻。  
模型权重以 MIT License 发布；使用需遵守仓库免责声明。
