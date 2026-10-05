# 游戏内体素引擎与Marching Cubes系统

## 项目描述

游戏内体素引擎与Marching Cubes系统的bug修复项目。核心模块 `voxel/engine.py` 存在多个已知bug，需要修复后使测试全部通过。

## 已知问题

- 体素存储完全没做只能用方块表示世界
- Marching Cubes完全没做不能生成平滑表面
- 体素渲染完全没做不能高效渲染体素
- 体素编辑完全没做不能修改体素世界
- 无限地形完全没做地形是有限的

## 运行测试

```bash
python -m pytest tests/test_voxel_engine.py -q
```

修复目标：所有 16 个测试全部通过。

## 技术要求

- 纯Python标准库，不引入第三方依赖
- 不修改测试文件
- 保持现有API接口不变
