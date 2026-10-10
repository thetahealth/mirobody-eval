"""
核心数据结构定义

测试用例:
- UserInfo                虚拟用户配置
- EvalInfo                评估配置（动态分发 — 配置模型由各 EvalAgent 插件自注册）
- TargetInfo              目标配置（动态分发 — 配置模型由各 TargetAgent 插件自注册）
- TestCase                测试用例聚合根

测试结果:
- TestCost          成本统计（基于 langchain UsageMetadata）
- EvalTrace         评估追踪信息（运维排查用）
- EvalResult        评估结果
- TestResult        单测试执行结果（组合 EvalResult）
- TestReport        完整评测报告

交互协议:
- TestAgentAction       测试代理动作
- TestAgentReaction     测试代理反应
- TargetAgentReaction   被测目标反应

记忆:
- TestAgentMemory       测试代理单轮记忆
- TargetAgentMemory     被测目标代理单轮记忆
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field, model_validator


# ============================================================
# 统一消息类型
# ============================================================

_ROLE_TO_CLASS = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage, "tool": ToolMessage}
_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _to_base_message(v: Any) -> BaseMessage:
    """将 OpenAI 格式 dict 或 langchain BaseMessage 统一转为 BaseMessage

    支持的输入格式:
      - langchain BaseMessage 实例: 直接透传
      - OpenAI 格式 dict: {"role": "user"/"assistant"/"system"/"tool", "content": "..."}
      - langchain 格式 dict: {"type": "human"/"ai"/"system", "content": "..."}（兼容旧数据反序列化）
    """
    if isinstance(v, BaseMessage):
        return v
    if isinstance(v, dict):
        content = v.get("content", "")
        # OpenAI 格式: {role, content}
        if "role" in v:
            cls = _ROLE_TO_CLASS.get(v["role"])
            if cls is None:
                raise ValueError(f"不支持的 role: {v['role']!r}，支持: {list(_ROLE_TO_CLASS.keys())}")
            if cls is ToolMessage:
                return cls(content=content, tool_call_id=v.get("tool_call_id", ""))
            return cls(content=content)
        # langchain 格式: {type, content} — 兼容旧报告反序列化
        if "type" in v:
            role = _TYPE_TO_ROLE.get(v["type"])
            if role is None:
                raise ValueError(f"不支持的 message type: {v['type']!r}，支持: {list(_TYPE_TO_ROLE.keys())}")
            cls = _ROLE_TO_CLASS[role]
            if cls is ToolMessage:
                return cls(content=content, tool_call_id=v.get("tool_call_id", ""))
            return cls(content=content)
        raise ValueError(f"消息 dict 必须包含 'role' 或 'type' 字段，实际: {list(v.keys())}")
    raise ValueError(f"消息格式不支持: {type(v).__name__}，需要 BaseMessage 或 {{role, content}} dict")


ChatMessage = Annotated[BaseMessage, BeforeValidator(_to_base_message)]
"""统一消息类型 — 兼容 OpenAI {role, content} 和 langchain BaseMessage

输入时自动转换为 BaseMessage（HumanMessage / AIMessage / SystemMessage），
下游可直接作为 BaseMessage 使用，对 langchain 完全透明。

接受的输入格式::

    # OpenAI 格式
    {"role": "user", "content": "你好"}
    {"role": "assistant", "content": "你好，有什么可以帮您？"}

    # langchain 格式（兼容旧数据）
    {"type": "human", "content": "你好"}

    # langchain 对象
    HumanMessage(content="你好")
"""


# ============================================================
# 虚拟用户配置
# ============================================================


# ============================================================
# 虚拟用户配置（Discriminated Union）
# ============================================================


class AttachmentRef(BaseModel):
    """单个附件引用 — 公开可访问的 URL + metadata

    用例 JSONL 里直接写链接和基本信息，target agent 原样塞进
    backend `file_list` 字段即可。不涉及签名、本地文件、manifest —
    所以任何数据集、任何 target 都能复用。

    ``turn`` 用于指定附件挂在第几轮用户消息上（从 1 开始）。默认第 1
    轮，也就是首条用户输入。
    """

    model_config = ConfigDict(extra="forbid")

    file_url: str = Field(description="公开可匿名访问的 URL")
    file_name: str = Field(description="文件原始名，展示用")
    file_type: str = Field(description="MIME，如 image/png / application/pdf")
    file_size: int = Field(description="字节数")
    file_key: Optional[str] = Field(default=None, description="可选的存储 key；纯记账用")
    turn: int = Field(default=1, ge=1, description="挂载到第几轮用户消息（1-based）")


class PersonaConfig(BaseModel):
    """虚拟用户行为特征配置 — 五维三档对抗画像

    | 维度        | 档1 (baseline)          | 档2 (中间)                  | 档3 (极端)        |
    |-------------|------------------------|----------------------------|--------------------|
    | disclosure  | responsive             | reluctant                  | withholding        |
    | attitude    | compliant              | impatient                  | dominant           |
    | cognition   | accurate               | partial_understanding      | complete_denial    |
    | logic       | consistent             | occasional_contradiction   | fabricating        |
    | expression  | normal                 | vague                      | incoherent         |
    """

    model_config = ConfigDict(extra="forbid")

    disclosure: Literal["responsive", "reluctant", "withholding"] = "responsive"
    attitude: Literal["compliant", "impatient", "dominant"] = "compliant"
    cognition: Literal["accurate", "partial_understanding", "complete_denial"] = "accurate"
    logic: Literal["consistent", "occasional_contradiction", "fabricating"] = "consistent"
    expression: Literal["normal", "vague", "incoherent"] = "normal"


class AutoUserInfo(BaseModel):
    """自动模式虚拟用户配置（LLM 驱动）

    LLM 根据 goal / context / finish_condition 和对话历史来决定说什么、是否结束。
    strict_inputs 用于控制前 N 轮强制输入，消费完毕后切换为 LLM 自主生成。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "type": "auto",
                    "goal": "咨询最近反复出现的偏头痛的可能原因和缓解方法",
                    "context": (
                        "28岁女性，互联网产品经理，长期伏案工作。"
                        "近一个月偏头痛发作 3 次，每次持续 2-4 小时，集中在右侧太阳穴。"
                        "有轻度近视（400度），无高血压或家族遗传病史。"
                        "性格偏急躁，期望快速得到明确答复。"
                    ),
                    "strict_inputs": ["我最近老是头疼，右边太阳穴那里，疼起来大概两三个小时"],
                    "max_turns": 8,
                    "finish_condition": "AI 给出了可能的病因分析并建议了具体的缓解措施或就医建议",
                },
            ]
        },
    )

    type: Literal["auto"] = Field(default="auto", description="用户类型")
    goal: str = Field(
        description=(
            "用户目标 — 用户希望通过与 AI 助手的对话达成什么结果。"
            "应当描述具体的健康诉求，LLM 会据此模拟有针对性的用户行为。"
            "好的示例：'了解反复偏头痛的可能原因，以及日常可以做哪些缓解'。"
        ),
    )
    context: Optional[str] = Field(
        None,
        description=(
            "用户上下文 — 一段自然语言描述，为 LLM 提供用户画像和背景信息。"
            "建议包含：基本信息、健康状况、生活习惯、性格特征等。"
            "LLM 会据此模拟出更逼真的用户行为。"
        ),
    )
    strict_inputs: List[str] = Field(
        default_factory=list,
        description=(
            "前 N 轮强制发送的对话内容，按顺序逐轮消费，消费完毕后切换为 LLM 自主生成。"
            "适用于需要精确控制开场白或复现固定对话路径的场景。"
        ),
    )
    max_turns: Optional[int] = Field(
        5,
        description=(
            "最大对话回合数 — 建议根据场景复杂度设置：\n"
            "  - 简单问答：3~5 轮\n"
            "  - 症状咨询与分析：5~10 轮\n"
            "  - 复杂多症状/慢性病管理：10~15 轮"
        ),
    )
    finish_condition: Optional[str] = Field(
        None,
        description=("终止条件 — 对话终止条件的语义化描述，LLM 据此判断何时结束。\n不填时默认为「goal 已达成」。"),
    )
    persona: Optional[PersonaConfig] = Field(
        None,
        description=(
            "虚拟用户行为特征 — 五维三档对抗画像配置。\n"
            "维度: disclosure (信息披露), attitude (态度), cognition (认知), logic (逻辑), expression (表达)。\n"
            "每个维度三档: 档1(baseline) → 档2(中间) → 档3(极端)。\n"
            "auto TestAgent 会据此在 system prompt 中注入对应的行为描述。"
        ),
    )
    attachments: List[AttachmentRef] = Field(
        default_factory=list,
        description=(
            "附件列表 — 每项含 file_url / file_name / file_type / file_size 和"
            " turn（挂到第几轮消息上，1-based，默认 1）。target agent 决定如何消费（通常"
            "透传到被测后端的 file_list 字段）。"
        ),
    )
    model: Optional[str] = Field(
        None,
        description=(
            "扮演这个虚拟用户的 LLM。留空则用 auto TestAgent 的默认值。"
            "CLI 用 --user-model 覆盖。\n"
            "这个字段存在的原因：它原先只是 AutoTestAgent 构造函数的一个默认参数，"
            "而 orchestrator 建实例时从不传它 —— 于是所有 auto 模式的数据集都被钉在"
            "那个默认模型上，没有 OPENAI_API_KEY 的环境里一条都跑不了，且无处可改。"
        ),
    )


class ManualUserInfo(BaseModel):
    """手动模式虚拟用户配置（脚本驱动）

    按顺序发送 strict_inputs 中的预设输入，用完即结束。
    零 LLM 调用、完全确定性、零成本。最大轮次自动设为 len(strict_inputs) + 1。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "type": "manual",
                    "strict_inputs": ["我最近总觉得心慌，有时候还会出冷汗", "大概持续半个月了，每次几分钟"],
                },
            ]
        },
    )

    type: Literal["manual"] = Field(default="manual", description="用户类型")
    strict_inputs: List[str] = Field(
        description=(
            "按序发送的预设对话内容（必填，至少一条）。"
            "按顺序逐轮发送，全部消费完毕后自动结束对话。"
            "适用场景：数据提取验证、预设问答、回归测试等单轮/固定轮次测试。"
        ),
    )
    attachments: List[AttachmentRef] = Field(
        default_factory=list,
        description=(
            "附件列表 — 每项含 file_url / file_name / file_type / file_size 和"
            " turn（挂到第几轮消息上，1-based，默认 1）。target agent 决定如何消费（通常"
            "透传到被测后端的 file_list 字段）。"
        ),
    )


# Discriminated Union — Pydantic 根据 type 字段自动路由到对应的 Config 类型
UserInfo = Annotated[
    AutoUserInfo | ManualUserInfo,
    Discriminator("type"),
]


# ============================================================
# 评估配置（动态分发 — 配置模型由各 EvalAgent 插件自注册）
# ============================================================


def _validate_eval_info(v: Any) -> Any:
    """从 EvalAgent plugin registry 动态分发 EvalInfo 验证

    每个 EvalAgent 插件通过 __init_subclass__(params_model=XxxEvalInfo) 注册配置模型，
    此验证器在 Pydantic 解析时根据 evaluator 字段查找对应模型并验证。
    """
    if isinstance(v, BaseModel):
        return v
    if not isinstance(v, dict):
        raise ValueError(f"EvalInfo 需要 dict 或 BaseModel，收到 {type(v).__name__}")
    evaluator = v.get("evaluator")
    if not evaluator:
        raise ValueError("EvalInfo 缺少 'evaluator' 字段")
    from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent

    model_cls = AbstractEvalAgent._params_registry.get(evaluator)
    if model_cls is None:
        registered = sorted(AbstractEvalAgent._params_registry.keys())
        raise ValueError(f"未注册的 evaluator: {evaluator!r}，已注册: {registered}")
    return model_cls.model_validate(v)


EvalInfo = Annotated[Any, BeforeValidator(_validate_eval_info)]


# ============================================================
# 被测目标配置（动态分发 — 配置模型由各 TargetAgent 插件自注册）
# ============================================================


def _validate_target_info(v: Any) -> Any:
    """从 TargetAgent plugin registry 动态分发 TargetInfo 验证"""
    if isinstance(v, BaseModel):
        return v
    if not isinstance(v, dict):
        raise ValueError(f"TargetInfo 需要 dict 或 BaseModel，收到 {type(v).__name__}")
    target_type = v.get("type")
    if not target_type:
        raise ValueError("TargetInfo 缺少 'type' 字段")
    from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent

    model_cls = AbstractTargetAgent._params_registry.get(target_type)
    if model_cls is None:
        registered = sorted(AbstractTargetAgent._params_registry.keys())
        raise ValueError(f"未注册的 target type: {target_type!r}，已注册: {registered}")
    return model_cls.model_validate(v)


TargetInfo = Annotated[Any, BeforeValidator(_validate_target_info)]


# ============================================================
# Target 配置规格（metadata.json 用）
# ============================================================


class TargetFieldSpec(BaseModel):
    """target 字段的配置规格 — 控制默认值、可编辑性、是否必填"""

    default: Any = Field(default=None, description="默认值")
    editable: bool = Field(default=True, description="是否允许 CLI/UI 修改（false 时只读）")
    required: bool = Field(default=False, description="是否为必填字段（无默认值时需用户提供）")


class TargetSpec(BaseModel):
    """metadata.json 中的 target 配置规格

    通过 type 确定目标类型，fields 逐字段定义默认值和可编辑性。
    CLI/UI 只能修改 editable=true 的字段；target_overrides（JSONL per-case）可修改任意字段。

    示例::

        {
            "type": "llm_api",
            "fields": {
                "model": {"default": "gpt-4.1", "editable": true, "required": true},
                "system_prompt": {"default": null, "editable": true}
            }
        }
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="目标类型（如 llm_api / hermes）")
    fields: Dict[str, TargetFieldSpec] = Field(
        default_factory=dict,
        description="各字段的配置规格（字段名 → TargetFieldSpec）",
    )

    def build_defaults(self) -> Dict[str, Any]:
        """从 spec 构建默认值 dict（含 type）"""
        base: Dict[str, Any] = {"type": self.type}
        for name, field in self.fields.items():
            if field.default is not None:
                base[name] = field.default
        return base

    def editable_field_names(self) -> set[str]:
        """获取所有可编辑字段名"""
        return {name for name, f in self.fields.items() if f.editable}


# ============================================================
# 测试用例聚合根
# ============================================================


class TestCase(BaseModel):
    """测试用例聚合根

    一个完整的测试用例，描述「谁（user）在哪个系统（target）上做什么测试，以及如何评估（eval）」。
    用于驱动 do_single_test 执行一次完整的虚拟用户对话 + 评估流程。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "id": "headache_basic_001",
                    "title": "偏头痛症状咨询 — 青年女性上班族",
                    "description": (
                        "模拟一个年轻女性用户咨询反复偏头痛问题。"
                        "验证 AI 能否通过追问收集关键信息（频率、持续时间、伴随症状），"
                        "并给出合理的病因分析和缓解建议。"
                    ),
                    "user": {
                        "type": "auto",
                        "goal": "咨询最近反复出现的偏头痛的可能原因和缓解方法",
                        "context": (
                            "28岁女性，互联网产品经理，长期伏案工作。"
                            "近一个月偏头痛发作 3 次，每次持续 2-4 小时，集中在右侧太阳穴。"
                            "有轻度近视（400度），无高血压或家族遗传病史。"
                            "性格偏急躁，期望快速得到明确答复。"
                        ),
                        "strict_inputs": ["我最近老是头疼，右边太阳穴那里，疼起来大概两三个小时"],
                        "max_turns": 8,
                        "finish_condition": "AI 给出了可能的病因分析并建议了具体的缓解措施或就医建议",
                    },
                    "target": {
                        "type": "llm_api",
                        "email": "demo1@symptom_entry_evaluation.com",
                    },
                    "eval": {
                        "evaluator": "semantic",
                        "criteria": [
                            {
                                "name": "diagnosis_quality",
                                "display_name": "病因分析质量",
                                "description": "AI 是否给出了合理的偏头痛病因分析",
                                "weight": 40,
                                "evaluation_points": ["是否考虑了紧张性头痛的可能", "是否关联了用户的工作习惯"],
                            },
                            {
                                "name": "advice_actionability",
                                "display_name": "建议可操作性",
                                "description": "给出的缓解建议是否具体、可执行",
                                "weight": 35,
                            },
                            {
                                "name": "communication",
                                "display_name": "沟通体验",
                                "description": "是否自然、共情、避免过度医学术语",
                                "weight": 25,
                            },
                        ],
                        "threshold": 0.7,
                    },
                    "tags": ["症状咨询", "头痛", "青年女性", "基础场景"],
                },
            ]
        },
    )

    id: str = Field(
        description=(
            "测试用例唯一标识。建议格式：{场景}_{难度}_{序号}，如 headache_basic_001、"
            "diabetes_edge_003。便于按场景和难度筛选、聚合报告。"
        ),
    )
    title: str = Field(
        description=(
            "测试用例标题 — 一句话概括测试场景，格式建议：'{症状/场景} — {用户画像关键特征}'。"
            "示例：'偏头痛症状咨询 — 青年女性上班族'、'儿童反复咳嗽 — 焦虑型家长'、"
            "'降压药副作用咨询 — 老年慢病患者'"
        ),
    )
    description: Optional[str] = Field(
        None,
        description=(
            "测试用例描述 — 2~3 句话说明本用例的测试意图和验证重点。"
            "应回答：这个用例想测什么？AI 应该表现出什么能力？什么情况算失败？"
            "示例：'模拟一个焦虑的家长咨询孩子反复咳嗽问题。"
            "验证 AI 能否有效安抚情绪并给出分级处理建议（观察 vs 就医）。'"
        ),
    )
    # 核心配置
    user: UserInfo = Field(description="虚拟用户配置 — 定义用户画像、目标和行为模式")
    target: Optional[TargetInfo] = Field(
        None,
        description=(
            "被测目标配置 — 定义被测系统实例和接入方式。"
            "可选字段：若不填，则使用运行时全局配置（如 benchmark runner 的 --target-xxx 参数）。"
            "优先级：用例级 target > 全局配置 > 错误"
        ),
    )
    eval: EvalInfo = Field(
        default_factory=lambda: __import__(
            "evaluator.plugin.eval_agent.semantic_eval_agent", fromlist=["SemanticEvalInfo"]
        ).SemanticEvalInfo(),
        description="评估配置 — 定义评估维度和通过标准，不填时使用默认语义评估",
    )

    # 评测前的历史对话
    history: List[ChatMessage] = Field(
        default_factory=list,
        description=(
            "评测前的历史对话记录 — 使用 ChatMessage 统一类型。\n"
            "支持多种输入格式（内部统一存储为 BaseMessage）：\n"
            "  - OpenAI 格式: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]\n"
            "  - langchain 对象: [HumanMessage(content='...'), AIMessage(content='...')]\n"
            "示例场景：用户与 AI 已进行了几轮对话后，从某个节点继续评测。"
        ),
    )

    # 元数据
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "测试用例标签 — 用于分类、筛选和聚合报告。建议从以下维度选取：\n"
            "  - 场景类型：症状咨询、用药指导、健康管理、急症处理、心理健康、营养饮食\n"
            "  - 用户特征：老年人、儿童家长、孕妇、慢病患者、青年白领\n"
            "  - 难度级别：基础场景、进阶场景、边界场景\n"
            "  - 能力维度：信息收集、病因分析、安全兜底、情绪安抚、多轮追问"
        ),
    )


# ============================================================
# 测试结果相关定义
# ============================================================


class TestCost(BaseModel):
    """测试成本

    各字段均为 dict[model_name, UsageMetadata]，与 do_execute 返回的 ExecuteResult.usage 格式一致。
    key 为模型名（如 "gpt-5.2"），value 为 langchain UsageMetadata（input_tokens / output_tokens / total_tokens）。
    """

    test: Dict[str, UsageMetadata] = Field(default_factory=dict, description="测试代理成本")
    eval: Dict[str, UsageMetadata] = Field(default_factory=dict, description="评测器成本")
    target: Optional[Dict[str, UsageMetadata]] = Field(default=None, description="被测目标成本（可选）")
    target_detail: Optional[Dict[str, Any]] = Field(
        default=None, description="被测目标成本明细（含 breakdown、cache、cost 等原始数据）"
    )


class EvalResult(BaseModel):
    """评估结果

    result 四态，按「有没有判成分」划分:

    判成了 ——
    - "pass"   — 得分 >= threshold，通过
    - "fail"   — 得分 < threshold，未通过
    - "scored" — 仅评分，不判定通过/失败（threshold 未设置时使用）

    没判成 ——
    - "error"  — 判分没发生:基础设施故障（API 502、超时、连接失败）、判分器
                 不可用（`JudgeUnavailable`）、用例被取消，以及本框架自身抛出
                 的任何异常。不计入 pass/fail，也不计入 avg_score。

    这条界线是硬的:`score` 字段在 error 下恒为 0.0，但那个 0 不是成绩，读它
    的代码必须先看 result。把没判成的算作 fail 0.00 会让「没跑完」和「答错了」
    无法区分 —— 而同一道题在预算不足时 0.00、预算够时 0.89，一个随评测方配置
    变化的数不能当成被测系统的属性。
    """

    result: Literal["pass", "fail", "scored", "error"] = Field(description="测试结果（pass/fail/scored/error）")
    score: float = Field(ge=0.0, le=1.0, description="测试得分（0.0 ~ 1.0）")
    feedback: Optional[str] = Field(None, description="测试结果的语义化描述")
    trace: Optional["EvalTrace"] = Field(None, description="评估追踪信息（两方记忆 + 评估详情）")


class TestResult(BaseModel):
    """单个测试用例执行结果"""

    id: str = Field(description="测试用例 ID")
    title: str = Field("", description="用例标题（来自 TestCase.title）")
    user_type: str = Field("", description="TestAgent 类型（如 auto / manual）")
    target_type: str = Field("", description="TargetAgent 类型（如 llm_api / hermes）")
    eval_type: str = Field("", description="EvalAgent 类型（如 semantic / healthbench）")
    eval: EvalResult = Field(description="评估结果")
    cost: TestCost = Field(default_factory=TestCost, description="测试成本")
    start: datetime = Field(description="测试开始时间")
    end: datetime = Field(description="测试结束时间")
    tags: List[str] = Field(default_factory=list, description="用例标签（来自 TestCase.tags）")
    target_config: Optional[Dict[str, Any]] = Field(
        default=None, description="被测目标配置参数（序列化的 TargetInfo，不含 type）"
    )
    eval_config: Optional[Dict[str, Any]] = Field(
        default=None, description="评估器配置参数（序列化的 EvalInfo，不含 evaluator）"
    )


class TestReport(BaseModel):
    """完整的评测报告"""

    cases: List[TestResult] = Field(default_factory=list, description="测试用例结果列表")
    total_cost: Dict[str, UsageMetadata] = Field(default_factory=dict, description="测试总成本")

    @property
    def pass_count(self) -> int:
        """通过的用例数"""
        return sum(1 for c in self.cases if c.eval.result == "pass")

    @property
    def fail_count(self) -> int:
        """失败的用例数"""
        return sum(1 for c in self.cases if c.eval.result == "fail")

    @property
    def error_count(self) -> int:
        """没判成分的用例数 — 超时/连接失败/判分器不可用/被取消

        读 `avg_score` 必须同时读这个:它们被排除在平均分之外，所以一个大面积
        失败的跑批会显示成漂亮的高分。
        """
        return sum(1 for c in self.cases if c.eval.result == "error")

    @property
    def pass_rate(self) -> float:
        """通过率 — 仅在有 pass/fail 判定的用例中计算，全部为 scored 时返回 0.0"""
        judged = self.pass_count + self.fail_count
        return self.pass_count / judged if judged else 0.0

    @property
    def avg_score(self) -> float:
        """平均得分 — 分母只含判成了分的用例（不含 error），无此类用例时返回 0.0

        error 的 score 恒为 0.0，但那个 0 不是成绩。见 EvalResult 的四态说明。
        """
        scored = [c for c in self.cases if c.eval.result != "error"]
        return sum(c.eval.score for c in scored) / len(scored) if scored else 0.0

    @property
    def total_duration_seconds(self) -> float:
        """所有用例的总耗时（秒）"""
        return sum((c.end - c.start).total_seconds() for c in self.cases)


# ============================================================
# 测试代理执行参数定义
# ============================================================


class TestAgentAction(BaseModel):
    """测试代理动作"""

    type: Literal["semantic", "message", "custom"] = Field(
        description="动作类型 — semantic: 语义化行为, message: 大模型消息, custom: 自定义行为"
    )
    semantic_content: Optional[str] = Field(None, description="语义化内容")
    message_content: Optional[Dict[str, Any]] = Field(None, description="大模型消息内容")
    custom_content: Optional[Dict[str, Any]] = Field(None, description="自定义内容（仅当 type 为 custom 时有效）")
    attachments: List[AttachmentRef] = Field(
        default_factory=list,
        description=(
            "随本次动作一同发送的附件列表。test agent 在消费"
            " user_info.attachments 时，按 turn 命中的那部分注入进来，target agent 原样透传。"
        ),
    )

    @model_validator(mode="after")
    def _check_type_content_consistency(self) -> "TestAgentAction":
        _FIELD_MAP = {"semantic": "semantic_content", "message": "message_content", "custom": "custom_content"}
        for action_type, field_name in _FIELD_MAP.items():
            if action_type != self.type and getattr(self, field_name) is not None:
                raise ValueError(f"type={self.type!r} 时 {field_name} 不应被设置")
        return self


class TestAgentReaction(BaseModel):
    """测试代理反应"""

    action: TestAgentAction = Field(description="当前测试代理期望行为")
    next_fuzzy_action: Optional[str] = Field(None, description="下一个模糊行为")
    reason: Optional[str] = Field(None, description="动作原因")
    is_finished: bool = Field(default=False, description="是否已完成目标，用于终止对话循环")
    usage: Optional[Dict[str, UsageMetadata]] = Field(
        None, description="本轮 LLM token 用量（dict[model_name, UsageMetadata]），非 LLM 路径为 None"
    )


# ============================================================
# 被测目标执行参数定义
# ============================================================


class TargetAgentReaction(BaseModel):
    """被测目标反应"""

    type: Literal["gui", "message", "custom"] = Field(
        description="被测目标动作类型 — gui: GUI 截图, message: 大模型消息, custom: 自定义行为"
    )
    gui_snapshots: Optional[List[str]] = Field(None, description="GUI 截图列表")
    message_list: Optional[List[Dict[str, Any]]] = Field(
        None, description="被测目标消息返回列表（包含 content / thinking / tools 等）"
    )
    custom_content: Optional[Dict[str, Any]] = Field(None, description="自定义内容（仅当 type 为 custom 时有效）")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="工具调用记录列表（ReAct 循环中的所有 tool call，每条含 name/args/result/call_id）"
    )
    usage: Optional[Dict[str, UsageMetadata]] = Field(
        None, description="本轮 LLM token 用量（dict[model_name, UsageMetadata]），非 LLM 路径为 None"
    )

    @model_validator(mode="after")
    def _check_type_content_consistency(self) -> "TargetAgentReaction":
        _FIELD_MAP = {"gui": "gui_snapshots", "message": "message_list", "custom": "custom_content"}
        for action_type, field_name in _FIELD_MAP.items():
            if action_type != self.type and getattr(self, field_name) is not None:
                raise ValueError(f"type={self.type!r} 时 {field_name} 不应被设置")
        return self

    def extract_text(self) -> str:
        """从 TargetAgentReaction 中提取可读文本（仅 reply 类型）"""
        if self.type == "message" and self.message_list:
            parts: list[str] = []
            for msg in self.message_list:
                if not isinstance(msg, dict):
                    continue
                # 有 type 字段时只取 reply；无 type（旧格式）视为 reply
                if msg.get("type", "reply") != "reply":
                    continue
                content = msg.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
            return "\n".join(parts)

        if self.type == "gui" and self.gui_snapshots:
            return "[系统展示了页面截图]"

        if self.type == "custom" and self.custom_content:
            return str(self.custom_content)

        return ""


# ============================================================
# 记忆模块定义
# ============================================================


class TestAgentMemory(BaseModel):
    """测试代理单轮记忆

    NOTE: 可变模型。创建时仅设置 test_reaction / test_reaction_time，
    target_response / target_response_time 在下一轮 do_generate 开始时由框架补全。
    最后一轮对话结束后 target_response 可能仍为 None。
    See: AbstractTestAgent.do_generate()
    """

    test_reaction: TestAgentReaction = Field(description="测试代理反应")
    test_reaction_time: datetime = Field(description="测试代理反应时间")
    target_response: Optional[TargetAgentReaction] = Field(None, description="被测系统响应")
    target_response_time: Optional[datetime] = Field(None, description="被测系统响应时间")


class TargetAgentMemory(BaseModel):
    """被测目标代理单轮记忆

    NOTE: 可变模型。创建时仅设置 target_reaction / target_reaction_time，
    test_response / test_response_time 在下一轮 do_generate 开始时由框架补全。
    See: AbstractTargetAgent.do_generate()
    """

    target_reaction: TargetAgentReaction = Field(description="被测系统反应")
    target_reaction_time: datetime = Field(description="被测系统反应时间")
    test_response: Optional[TestAgentAction] = Field(None, description="测试代理行为")
    test_response_time: Optional[datetime] = Field(None, description="测试代理行为时间")


class SessionInfo(BaseModel):
    """目标系统会话信息，由 TargetAgent.get_session_info() 提供

    用于向 EvalAgent 暴露目标系统的认证/身份数据（如 IndicatorEvalAgent 需要
    token + user_id 来查询健康指标 API），避免 eval 插件直接 getattr agent 对象。
    """

    user_id: str = ""
    user_token: str = ""
    has_user_data: bool = False  # True 表示目标系统已获得用户健康档案（如 llm_api 注入了 system_prompt）


# ============================================================
# 评估追踪信息
# ============================================================


class EvalTrace(BaseModel):
    """评估追踪信息（运维排查用）

    完整保留测试代理和被测目标两方的交互记忆，以及评估器的详细输出，
    方便后续回溯对话过程、定位问题。
    """

    # 评测前历史对话
    history: List[ChatMessage] = Field(default_factory=list, description="评测前的历史对话记录")

    # 两方完整交互记忆
    test_memory: List[TestAgentMemory] = Field(default_factory=list, description="测试代理完整对话记忆")
    target_memory: List[TargetAgentMemory] = Field(default_factory=list, description="被测目标完整对话记忆")

    # 评估器详细输出（各评估器自行定义内容）
    eval_detail: Optional[Dict[str, Any]] = Field(None, description="评估器输出的详细信息")


# 解析前向引用
EvalResult.model_rebuild()


# ============================================================
# 共享数据模型（CLI + Web API 通用）
# ============================================================


class DatasetInfo(BaseModel):
    """数据集基本信息"""

    name: str
    case_count: int
    file_size_kb: float
    evaluator: str = ""  # 数据集使用的评估器类型（首条 case 的 eval.evaluator）
    # 能不能跑。发布批次里最新一期是闭卷的 —— 题目发了、答案没发，所以没有
    # evaluator 字段、加载即失败。列出来但标明原因，比藏起来有用：它确实存在、
    # 也确实是榜单在用的那个，只是还不能自己跑。
    unavailable_reason: str = ""


class BenchmarkSummary(BaseModel):
    """Benchmark 摘要"""

    name: str
    description: str
    datasets: List[DatasetInfo]
    target: List[TargetSpec] = Field(default_factory=list, description="Target 配置规格列表（支持多 target 类型）")
    prepare_status: str | None = None  # None | "running" | "completed" | "error"
    prepare_error: str | None = None  # 错误信息（仅 status="error" 时）


class CaseSummary(BaseModel):
    """用例轻量摘要"""

    id: str
    title: str
    user_type: str = ""
    target_type: str = ""
    evaluator: str = ""


class DatasetDetail(BaseModel):
    """数据集详情"""

    benchmark: str
    dataset: str
    case_count: int
    cases_preview: List[Dict[str, Any]]
    case_summaries: List[CaseSummary]
    tag_distribution: Dict[str, int]
    description: str = ""
    target: List[TargetSpec] = Field(default_factory=list, description="Target 配置规格列表（支持多 target 类型）")
    evaluator: str = ""  # 数据集使用的评估器类型


class AgentInfo(BaseModel):
    """Agent 元信息"""

    name: str
    class_name: str
    module_doc: str
    short_desc: str
    features: List[str]
    icon: str
    color: str
    icon_url: str = ""  # 产品 logo 图片路径（优先于 SVG icon 展示）
    config_schema: Dict[str, Any]
    config_examples: List[Dict[str, Any]]
    cost_meta: Dict[str, Any] = {}  # 费用预估元数据（由 plugin 声明 _cost_meta）


class ReportEntry(BaseModel):
    """报告索引条目"""

    benchmark: str
    dataset: str
    date: str
    filename: str
    target_label: str = ""
