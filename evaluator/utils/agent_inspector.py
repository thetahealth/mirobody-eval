"""反射 plugin registry，提取 agent 元数据（docstring + schema + examples + 展示/费用元数据）

自动发现机制（inspector 完全不感知具体 plugin）:
- EvalAgent / TargetAgent 的 config map 从各 plugin 类注册的 params_model 自动派生
- TestAgent 的 config map 从 plugin 类的 _config_model 属性自动发现
- 展示元数据（icon / color / features）从 plugin 类的 _display_meta 属性读取
- 费用预估元数据从 plugin 类的 _cost_meta 属性读取
- 未声明的 plugin 使用默认值，不影响功能
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Type, get_args

# 触发插件注册
import evaluator.plugin.eval_agent  # noqa: F401
import evaluator.plugin.target_agent  # noqa: F401
import evaluator.plugin.test_agent  # noqa: F401
from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.interfaces.abstract_test_agent import AbstractTestAgent
from evaluator.core.schema import AgentInfo


# ==================== Config Map 自动派生 ====================


def _build_config_map(union_type: type, discriminator_field: str) -> Dict[str, type]:
    """从 Discriminated Union 自动构建 name → config model 映射

    开发者只需在 schema.py 中定义配置模型并加入 Union，此处自动发现。

    原理:
        EvalInfo = Annotated[A | B | C, Discriminator("evaluator")]
        → get_args → 遍历 Union 成员 → 读取 discriminator 字段的 Literal 值
    """
    args = get_args(union_type)
    if not args:
        return {}

    # Annotated[Union[A, B, C], Discriminator("field")] → 取 union 成员
    union_members = get_args(args[0])
    if not union_members:
        return {}

    config_map: Dict[str, type] = {}
    for member in union_members:
        field_info = member.model_fields.get(discriminator_field)
        if field_info and field_info.annotation:
            literal_args = get_args(field_info.annotation)
            if literal_args:
                config_map[literal_args[0]] = member

    return config_map


# EvalAgent / TargetAgent: 从 plugin _params_registry 自动派生
_EVAL_CONFIG_MAP: Dict[str, type] = dict(AbstractEvalAgent._params_registry)
_TARGET_CONFIG_MAP: Dict[str, type] = dict(AbstractTargetAgent._params_registry)


def _build_test_config_map() -> Dict[str, type]:
    """从 TestAgent plugin 类的 _config_model 属性自动构建 config map

    plugin 类声明方式:
        class AutoTestAgent(AbstractTestAgent, name="auto"):
            _config_model = "AutoUserInfo"  # schema.py 中的类名
    """
    import evaluator.core.schema as schema_module

    config_map: Dict[str, type] = {}
    for name, cls in AbstractTestAgent.get_all().items():
        model_name = getattr(cls, "_config_model", None)
        if model_name:
            model_cls = getattr(schema_module, model_name, None)
            if model_cls is not None:
                config_map[name] = model_cls
    return config_map


# TestAgent: 从 plugin 类的 _config_model 属性自动发现
_TEST_CONFIG_MAP: Dict[str, type] = _build_test_config_map()


# ==================== 展示元数据 ====================


@dataclass
class _AgentMeta:
    """Agent 展示元数据，纯前端 presentation 层"""

    icon: str  # SVG path data (heroicons outline, 24x24 viewBox)
    color: str  # CSS 颜色值
    features: list[str] = field(default_factory=list)
    icon_url: str = ""  # 产品 logo 图片路径（优先于 SVG icon 展示）


# 未声明 _display_meta 的 plugin 使用此默认值
_DEFAULT_META = _AgentMeta(
    icon="M11.42 15.17l-5.657-5.657a8 8 0 1111.314 0l-5.657 5.657z",
    color="#71717a",
)


# ==================== 核心逻辑 ====================


def _get_module_doc(cls: type) -> str:
    """从 class 所在模块获取 module docstring"""
    module = sys.modules.get(cls.__module__)
    if module and module.__doc__:
        return module.__doc__.strip()
    return cls.__doc__ or ""


def _parse_short_desc(module_doc: str) -> str:
    """从 module_doc 中提取一句话简介（标题 — 后面的部分）"""
    lines = module_doc.strip().split("\n")
    if lines:
        title = lines[0]
        for sep in ("—", "—"):
            if sep in title:
                return title.split(sep, 1)[1].strip()
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_config_info(config_model: type | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """提取 Pydantic model 的 JSON schema 和 examples"""
    if config_model is None:
        return {}, []
    schema = config_model.model_json_schema()
    examples = []
    extra = config_model.model_config.get("json_schema_extra", {})
    if isinstance(extra, dict):
        examples = extra.get("examples", [])
    return schema, examples


def _resolve_meta(cls: type) -> _AgentMeta:
    """从 plugin 类的 _display_meta 属性读取展示元数据，未声明则使用默认值

    plugin 类声明方式:
        class MyAgent(AbstractEvalAgent, name="my"):
            _display_meta = {
                "icon": "M9 12.75L...",           # heroicons SVG path (可选)
                "color": "#8b5cf6",                # CSS 颜色值 (可选)
                "features": ["LLM 驱动", "..."],   # 特性标签 (可选)
            }
    """
    plugin_meta = getattr(cls, "_display_meta", None)
    if isinstance(plugin_meta, dict):
        return _AgentMeta(
            icon=plugin_meta.get("icon", _DEFAULT_META.icon),
            color=plugin_meta.get("color", _DEFAULT_META.color),
            features=plugin_meta.get("features", []),
            icon_url=plugin_meta.get("icon_url", ""),
        )
    return _DEFAULT_META


def _resolve_cost_meta(cls: type) -> Dict[str, Any]:
    """从 plugin 类的 _cost_meta 属性读取费用预估元数据

    plugin 类声明方式:
        # EvalAgent: 声明单 case 评估费用 (USD)
        class MyEvalAgent(AbstractEvalAgent, name="my"):
            _cost_meta = {"est_cost_per_case": 0.035}

        # TargetAgent: 声明单次调用 token 估算（实际费用 = tokens × 模型定价）
        class MyTargetAgent(AbstractTargetAgent, name="my"):
            _cost_meta = {"est_input_tokens": 200, "est_output_tokens": 600}
    """
    cost_meta = getattr(cls, "_cost_meta", None)
    if isinstance(cost_meta, dict):
        return dict(cost_meta)  # 返回副本
    return {}


def _inspect_registry(registry: Dict[str, Type], config_map: Dict[str, type]) -> list[AgentInfo]:
    """通用的 registry 反射函数"""
    result: list[AgentInfo] = []
    for name, cls in sorted(registry.items()):
        config_model = config_map.get(name)
        schema, examples = _extract_config_info(config_model)
        module_doc = _get_module_doc(cls)
        meta = _resolve_meta(cls)
        result.append(
            AgentInfo(
                name=name,
                class_name=cls.__name__,
                module_doc=module_doc,
                short_desc=_parse_short_desc(module_doc),
                features=meta.features,
                icon=meta.icon,
                color=meta.color,
                icon_url=meta.icon_url,
                config_schema=schema,
                config_examples=examples,
                cost_meta=_resolve_cost_meta(cls),
            )
        )
    return result


def list_eval_agents() -> list[AgentInfo]:
    return _inspect_registry(AbstractEvalAgent.get_all(), _EVAL_CONFIG_MAP)


def list_target_agents() -> list[AgentInfo]:
    return _inspect_registry(AbstractTargetAgent.get_all(), _TARGET_CONFIG_MAP)


def list_test_agents() -> list[AgentInfo]:
    return _inspect_registry(AbstractTestAgent.get_all(), _TEST_CONFIG_MAP)
