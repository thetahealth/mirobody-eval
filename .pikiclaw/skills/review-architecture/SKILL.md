---
name: review-architecture
description: 审查项目架构健康度 — 检查 GitOps 合规、Plugin 可插拔性、CLI/Web 复用。当用户要求架构审查、代码重构后验证、或新增模块后检查依赖关系时使用。
argument-hint: [focus-area] # 可选: gitops | plugin | reuse | all(默认)
---

# Review Architecture

对 HolyEval 项目进行结构化架构审查，输出检查报告并标注违规项。

## 审查范围

根据 $ARGUMENTS 决定审查范围（默认 `all`）：

| 参数 | 审查内容 |
|------|---------|
| `gitops` | 仅检查 GitOps 合规 |
| `plugin` | 仅检查 Plugin 可插拔性 |
| `reuse` | 仅检查 CLI/Web 复用 |
| `all` | 全部检查 |

## Check 1: GitOps 合规

**原则**: 所有代码、benchmark 数据、评测报告都在 Git 仓库中管理，便于版本追踪和协作维护。

### 检查项

1. **benchmark 数据目录结构**
   - `benchmark/data/<name>/` 每个 benchmark 包含 `metadata.json` + `*.jsonl`
   - 运行: `ls benchmark/data/*/`，确认无游离文件

2. **report 输出在仓库内**
   - `benchmark/report/` 目录存在且镜像 `data/` 子目录结构
   - 报告文件名遵循 `{dataset}_{target_label}_{YYYYMMDD_HHmmss}.json` 格式
   - 运行: `ls benchmark/report/*/` 检查

3. **.gitignore 合理性**
   - `.env` 被忽略（含 API Key）
   - `__pycache__`、`.venv` 被忽略
   - `benchmark/report/` **不被**忽略（报告需入库）
   - 运行: `cat .gitignore`，检查以上规则

4. **无硬编码路径或环境依赖**
   - 搜索 `evaluator/` 和 `web/` 中是否有硬编码绝对路径
   - 运行: `rg '/Users/|/home/|C:\\\\' evaluator/ web/ benchmark/`
   - 配置通过 `.env` + `evaluator/utils/config.py` 读取，不散落在代码中

5. **依赖管理**
   - `pyproject.toml` 存在且包含完整依赖声明
   - `uv.lock` 存在且已提交（锁定版本）
   - 运行: `git ls-files pyproject.toml uv.lock`

### 输出模板

```
## GitOps 合规检查
- [x] benchmark 数据目录结构规范
- [x] report 输出在仓库内
- [ ] ⚠️ .gitignore 缺少 xxx 规则
- [x] 无硬编码路径
- [x] 依赖管理完整
```

## Check 2: Plugin 可插拔性

**原则**: Plugin 通过 `__init_subclass__` 注册，core/ 层仅依赖抽象接口，新增/删除 plugin 不影响框架核心。

### 检查项

1. **core/ 不导入具体 plugin 类**
   - 在 `evaluator/core/` 中搜索是否直接 import 了 plugin 实现类名
   - 运行: `rg 'from evaluator\.plugin\.' evaluator/core/`
   - **允许**: `import evaluator.plugin.xxx` （触发注册的 side-effect import）
   - **禁止**: `from evaluator.plugin.xxx import XxxAgent` （直接引用具体类）

2. **web/ 不直接导入 plugin 实现**
   - 运行: `rg 'from evaluator\.plugin\.\w+\.\w+ import' web/`
   - **允许**: `import evaluator.plugin.xxx` （注册触发）
   - **禁止**: 直接导入具体实现类

3. **Schema 集中定义**
   - `EvalInfo` / `TargetInfo` 在 `evaluator/core/schema.py` 里是
     `Annotated[Any, BeforeValidator(...)]`，按注册表动态分发
   - 每个插件的 `*Info` 配置类与实现**同文件**，用 `params_model=` 注册
   - **EvalAgent / TargetAgent** 必须传 `params_model=`:
     `rg 'Abstract(Eval|Target)Agent,' -A3 evaluator/plugin/ | rg -c params_model`
   - **TestAgent 不收 `params_model`**（`abstract_test_agent.py:43` 的 `__init_subclass__`
     只接受 `name`），它用类属性 `_config_model` 的字符串名。给 TestAgent 传
     `params_model=` 会直接 `TypeError`

4. **Registry 反射使用**
   - `evaluator/utils/agent_inspector.py` 通过 `get_all()` 反射发现 plugin
   - 不硬编码 plugin 名称列表
   - 运行: `rg 'get_all\(\)|get\(' evaluator/utils/agent_inspector.py`

5. **Plugin 间无交叉依赖**
   - 一个 plugin 不 import 另一个 plugin
   - 运行: `rg 'from evaluator\.plugin\.' evaluator/plugin/ --glob '*.py'`
   - 每个匹配应只在自己的 `__init__.py` 中（注册用）

6. **自动派生检查**
   - `bench_schema.py` 的 `_get_target_type_map()` 从 `AbstractTargetAgent._params_registry` 派生
   - `agent_inspector.py` 的 config map 同样从各插件的 `params_model` 派生
   - 不存在手动维护的 type→class 映射表

### 输出模板

```
## Plugin 可插拔性检查
- [x] core/ 不直接导入 plugin 实现
- [x] web/ 不直接导入 plugin 实现
- [x] Schema 集中定义在 schema.py
- [x] Registry 反射使用
- [ ] ⚠️ plugin A 导入了 plugin B: xxx
- [x] 自动派生无硬编码映射
```

## Check 3: CLI/Web 复用

**原则**: CLI 和 Web 共享同一套底层能力（evaluator/utils/ + evaluator/core/），不重复实现。

### 检查项

1. **共享层位置正确**
   - 以下功能在 `evaluator/utils/` 或 `evaluator/core/` 中:

   | 能力 | 期望位置 |
   |------|---------|
   | benchmark 读取 (list/load/detail/filter) | `evaluator/utils/benchmark_reader.py` |
   | report 读写 | `evaluator/utils/report_reader.py` |
   | agent 元数据 | `evaluator/utils/agent_inspector.py` |
   | target 解析 | `evaluator/core/bench_schema.py` |
   | 批量执行 | `evaluator/core/orchestrator.py` |
   | BenchItem→TestCase | `evaluator/core/bench_schema.py` |

2. **Web services 层是纯 re-export**
   - `web/app/services/benchmark_reader.py` 应为 `from evaluator.utils.benchmark_reader import *`
   - `web/app/services/report_reader.py` 应为 `from evaluator.utils.report_reader import *`
   - `web/app/services/agent_inspector.py` 应为 `from evaluator.utils.agent_inspector import *`
   - 不应包含额外业务逻辑
   - 运行: `wc -l web/app/services/benchmark_reader.py web/app/services/report_reader.py web/app/services/agent_inspector.py`

3. **无跨包逆向依赖**
   - `web/` 不导入 `benchmark/`（benchmark 是 CLI runner，不是共享层）
   - `evaluator/` 不导入 `web/` 或 `benchmark/`
   - 运行:
     ```
     rg 'from benchmark\.' web/
     rg 'from web\.' evaluator/
     rg 'from benchmark\.' evaluator/
     ```

4. **CLI 和 Web 调用同一入口**
   - `benchmark/basic_runner.py` 和 `web/app/services/task_manager.py` 都调用:
     - `bench_item_to_test_case()`
     - `resolve_runtime_target()`
     - `build_bench_report()`
     - `BatchSession` / `do_batch_test()`
   - 不应各自实现解析/转换逻辑

### 输出模板

```
## CLI/Web 复用检查
- [x] 共享层位置正确
- [x] Web services 层纯 re-export
- [ ] ⚠️ web/ 导入了 benchmark/: xxx
- [x] CLI 和 Web 调用同一入口
```

## 依赖方向图（合规基准）

```
evaluator/core/        ← 核心接口 + schema（不依赖 plugin/utils 以外的任何包）
evaluator/utils/       ← 共享工具层（仅依赖 core/）
evaluator/plugin/      ← 可插拔实现（依赖 core/ + utils/）
benchmark/             ← CLI runner（依赖 core/ + utils/，不被 web/ 依赖）
web/                   ← Web UI（依赖 core/ + utils/，不依赖 benchmark/）
generator/             ← 数据转换工具（独立，仅依赖 core/schema）
```

违反此图的导入方向即为架构问题。

## 执行流程

1. 根据 $ARGUMENTS 确定审查范围
2. 对每个 Check，逐项执行检查命令
3. 收集结果，输出完整报告
4. 对每个 ⚠️ 项给出修复建议（文件路径 + 具体改法）
5. 如无违规，输出 "✅ 架构审查通过"
