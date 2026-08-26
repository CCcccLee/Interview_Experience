# LangChain

#### 如何理解langchain

- 在现在的 LangChain 里，Chain 最重要的技术基础是 Runnable。每个步骤都尽量遵守统一的输入输出和执行接口，再通过 **LCEL** 的 管道操作符| 做串行组合（RunnableSequence），或者通过字典、RunnableParallel 做并行组合。

- 组合后的整条 Chain 本身仍然是 Runnable，所以可以继续嵌套，嵌套之后的结果仍然是Runnable，也能统一使用 invoke、ainvoke、batch 和 stream 等能力。Runnable 还暴露输入、输出和配置的 schema，并允许通过 `config` 携带标签、元数据等信息。

- 核心设计理念是「组合优于堆积封装」。开发者只关注每一步做什么、数据怎么流动，框架负责把执行方式、配置、重试、回退和追踪等通用能力接到整条流程上。

- 需要注意版本边界。LLMChain、SequentialChain 属于旧式 Chain API，LangChain v1 已把这类能力移入 langchain-classic，适合维护旧项目，不应再作为新项目的首选写法。

- 确定性的线性或分支流程可以用 Runnable 和 LCEL，Agent 让模型在运行时动态决定下一步；带循环、持久状态和人工审批的复杂工作流，则更适合用 LangGraph。

  - ##### 怎么选用langchain，agent，langgraph

    如果步骤和数据流在编码时就能确定，比如文本清洗、检索增强问答、分类后解析，Runnable 和 LCEL 通常很合适。它们结构直接，调用方式统一，也容易追踪。
    如果下一步取决于模型的动态判断，例如模型要自己选择搜索、计算器还是数据库工具，并可能重复多轮，问题就从「固定数据流」变成了「Agent 循环」。在 LangChain v1 中，官方推荐用 create_agent 构建标准 Agent，这套 Agent 架构运行在 LangGraph 之上。
    如果我们还要精确控制循环、分支、状态持久化、失败恢复和人工介入，那就进一步使用 LangGraph 的底层图编排能力。LangGraph 并不是为了取代每一条简单 Chain，而是处理 Chain 难以清楚表达的长时、有状态工作流。

#### LangChain v1底层架构与实现原理

 LangChain v1 更像一套面向 Agent 的分层开发框架，而不只是将 Prompt 串起来的 Chain 工具。

底层的 `langchain-core` 定义 Message、Model、Tool 和 Runnable 等标准协议；不同模型厂商的独立集成包负责把自己的请求与响应适配到这些协议，因此应用层可以使用相对统一的方式切换模型和工具。

在执行层，Runnable 统一了组件的同步、异步、批处理和流式调用方式。对于步骤固定的流程，可以使用 LCEL 组合 Prompt、Model 和 Parser；对于需要模型自主选择工具的任务，则使用 `create_agent` 创建 Agent。

`create_agent` 会把模型节点和工具节点编译成 **LangGraph 状态图**。模型读取消息后生成 `AIMessage`；如果其中包含 `tool_calls`，运行时执行对应工具，并把结果包装成带相同调用 ID 的 `ToolMessage` 写回状态；模型再次读取工具结果并继续判断，直到生成最终回答。

在这套架构中，State 保存会变化的消息与业务状态，Context 提供可信依赖，Store 保存跨线程数据；Middleware 负责在模型或工具调用前后加入权限、重试、摘要与人工审批；LangGraph 则负责路由、检查点、暂停恢复和长时间运行。

#### LangChain的Deep Research

核心流程是：先澄清用户目标并生成 Research Brief，再由 Supervisor 把问题拆成相对独立的子课题。多个 Researcher 在隔离上下文中并行检索，并对来源进行筛选和压缩。

Supervisor 会检查证据是否覆盖研究目标，发现空白就继续补搜，最后再由统一的写作阶段综合证据并生成带引用的报告。

这种架构的价值不只是并行加速。子 Agent 可以隔离不同主题的上下文，避免大量搜索结果互相干扰；Supervisor 可以根据中间证据动态调整方向；统一写作则能减少章节重复和口径冲突。

它适合竞品分析、技术调研、文献综述和供应商尽调等开放式、多来源、可拆分任务，不适合一次搜索就能回答的简单事实，也不适合子任务高度依赖的强耦合工作。

生产环境必须限制并发、迭代、Token 和搜索预算，并对网页提示词注入、来源可信度和高风险结论进行人工复核。