# LangChain

#### 如何理解langchain

- 在现在的 LangChain 里，Chain 最重要的技术基础是 Runnable。每个步骤都尽量遵守统一的输入输出和执行接口，再通过 **LCEL** 的 管道操作符| 做串行组合（RunnableSequence），或者通过字典、RunnableParallel 做并行组合。

- 组合后的整条 Chain 本身仍然是 Runnable，所以可以继续嵌套，嵌套之后的结果仍然是Runnable，也能统一使用 invoke、ainvoke、batch 和 stream 等能力。Runnable 还暴露输入、输出和配置的 schema，并允许通过 `config` 携带标签、元数据等信息。

- 核心设计理念是「组合优于堆积封装」。开发者只关注每一步做什么、数据怎么流动，框架负责把执行方式、配置、重试、回退和追踪等通用能力接到整条流程上。

- 需要注意版本边界。LLMChain、SequentialChain 属于旧式 Chain API，LangChain v1 已把这类能力移入 langchain-classic，适合维护旧项目，不应再作为新项目的首选写法。

- 确定性的线性或分支流程可以用 Runnable 和 LCEL，Agent 让模型在运行时动态决定下一步；带循环、持久状态和人工审批的复杂工作流，则更适合用 LangGraph。

  - ##### 怎么选用langchain，agent，langgraph

    - 如果步骤和数据流在编码时就能确定，比如文本清洗、检索增强问答、分类后解析，Runnable 和 LCEL 通常很合适。它们结构直接，调用方式统一，也容易追踪。
    - 如果下一步取决于模型的动态判断，例如模型要自己选择搜索、计算器还是数据库工具，并可能重复多轮，问题就从「固定数据流」变成了「Agent 循环」。在 LangChain v1 中，官方推荐用 create_agent 构建标准 Agent，这套 Agent 架构运行在 LangGraph 之上。
    - 如果我们还要精确控制循环、分支、状态持久化、失败恢复和人工介入，那就进一步使用 LangGraph 的底层图编排能力。LangGraph 并不是为了取代每一条简单 Chain，而是处理 Chain 难以清楚表达的长时、有状态工作流。

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

# dataFlyWheel LangGraph 项目技术面试问答

> 本文与 `INTERVIEW_QUESTIONS.md` 配套，按“项目从一开始就采用 LangGraph”这一口径回答。  
> 回答分为“当前实现”和“生产化演进”两层；面试时应结合真实职责替换涉及个人贡献和业务收益的示例。

## 1. 项目介绍与架构判断

### 问题 1：三分钟介绍项目

**参考回答：**

这是一个面向网络故障排障场景的 API 仿真数据生成系统。输入包括故障类型、网络拓扑、排障步骤、API 定义和会话信息；输出是按排障逻辑组织的模拟 API 调用结果、每轮校验结果、反思内容和完整执行轨迹。大模型负责处理规则难以穷举的语义推理和结构化生成，确定性规则负责兜底约束。

系统用 LangGraph 编排 `prepare_context → generate → advise → 三路并行验证 → merge_validation`。通过则结束，超时则终止，其他失败进入 `reflect → persist_reflection → advance_round` 再生成。LangGraph 主要解决显式状态、条件分支、循环、并行汇合、Checkpoint 和失败恢复；RAG、模型调用、规则校验和 MCP 仍由原有领域模块承担。

它与普通 RAG 问答的区别是：目标不是生成一段读起来合理的文本，而是生成可执行、顺序正确、参数合法且符合网络故障因果关系的结构化 API 结果。评价核心是结果正确性，语言质量只是辅指标。纯规则系统可以覆盖 Schema 和少量固定故障，但很难覆盖组合拓扑、长尾故障和开放式推理。

### 问题 2：画出完整 LangGraph 流程

**参考回答：**

```text
START
  → prepare_context       # 取领域知识、历史反思和根因提示
  → generate              # Worker 生成本轮 mocked_apis
  → advise                # Advisor 修正跨 API 一致性
  → ┬ llm_validate        # 语义质量，调用 LLM
    ├ rule_validate       # 顺序/字段等硬规则，确定性
    └ mcp_validate        # 外部工具或服务验证
  → merge_validation      # 汇总证据并形成统一分数/结论
  → passed  → finish_passed → END
  → timeout → finish_timeout → END
  → reflect → persist_reflection
             → max_rounds → finish_max_rounds → END
             → advance    → advance_round → generate
```

节点只返回自己负责的 Partial State。`prepare_context`、规则验证、合并、计数和路由原则上是确定性的；Worker、Advisor、LLM Validator、Tutor 会调用模型；MCP 和持久化节点会访问外部系统。纯计算节点可安全重试，MCP 查询通常也可重试，但写文件、写知识库等副作用必须以 `source_id` 做幂等。Worker 负责创造候选，Advisor 负责跨 API 一致性，拆开后提示词、模型、评测和失败定位都更清晰。

### 问题 3：为什么选择 LangGraph？

**参考回答：**

本项目不是一次性的线性 Chain，而是“生成—并行校验—条件路由—反思—再生成”的有状态循环。LangGraph 把节点、边、State、并行汇合和 Checkpoint 变成统一执行语义，使每轮状态可观察、可持久化、可恢复，也能自然加入人工中断。

普通 Python 循环可以实现功能，但并行分支、状态快照、断点恢复、流式节点事件和人工恢复都要自行建设。LangChain 更适合模型、Prompt、Retriever、Tool 等组件抽象；LangGraph更适合控制流程，二者不是互斥关系，本项目可以在节点内部使用 LangChain 组件。完全自主的 LangChain Agent 会把关键路由交给模型，不利于稳定性、审计和成本控制。

传统状态机能做确定性流转，却缺少面向 LLM 工作流的状态通道、并行节点、持久化和工具生态。Celery负责分布式任务，Temporal负责更强的分布式持久执行，Airflow偏批处理调度；它们可以成为生产外围基础设施，但不替代图内的 Agent 编排。若流程只有一次检索和一次生成、无循环恢复，使用 LangGraph就是过度设计。

### 问题 4：为什么不是所有模块都做成 Agent？

**参考回答：**

Agent 适合目标明确但步骤不固定、需要根据语义动态选工具的环节；确定性路由适合安全边界、硬规则、成本控制和可审计流程。Rule Validator 不需要 Agent，因为同样输入必须得到同样输出。MCP 是否调用由图决定，模型只提供必要参数，避免模型跳过关键校验或调用越权工具。

多 Agent 不必然更强，它会增加提示词漂移、调用成本、通信协议和调试难度。本项目用一个显式状态图协调若干职责清晰的模型节点，只有在故障类型差异大、工具集合动态或某子任务确实需要自主规划时，才考虑独立 Agent 或子图。

## 2. StateGraph 与状态设计

**性能建议**：State 越大，每次 Checkpoint 写入的开销就越大。实践中，对于长文本，应该只存其**哈希值、文档 ID**，在需要时再通过检索器获取，而不是把整个文本在节点间传来传去。

### 问题 5：Graph State 保存了哪些信息？

**参考回答：**

`sample` 是原始业务输入；`session_id` 关联业务会话；`iteration_id` 是当前轮次；`current_reflection` 驱动下一轮；`domain_knowledge`、`root_cause_prompt` 是检索上下文；`mocked_apis` 是本轮候选；三个验证字段保存并行分支结果；`result` 聚合所有轮次；`success` 和 `termination_reason` 表示终态。运行耗时还由 `started_at` 支撑。

业务状态是样本、知识、候选、验证和结果，运行状态是轮次、起始时间和终止原因。LLM Client、数据库连接、锁、文件句柄不能放进 State，因为它们不可稳定序列化，也不应成为业务快照的一部分。State 越大，Checkpoint 的写入、网络、存储和恢复成本越高；生产中长文本宜保存内容哈希、文档 ID 和版本，只有恢复必须的短上下文才内嵌。

### 问题 6：为什么节点返回 Partial State？

**参考回答：**

Partial State 明确声明节点写了哪些通道，便于 LangGraph 合并并行写入、生成 Checkpoint 和追踪差异。原地修改共享对象会导致框架无法判断变更边界，并行时产生数据竞争，重放时也可能在旧对象上重复追加。

节点应尽量设计成“State 输入 → Partial State 输出”的纯函数。模型和外部服务调用不可避免地有外部依赖，但结果仍通过返回值进入 State；真正的持久化副作用放到单独节点，并配幂等键、超时和审计。

### 问题 7：并行节点为什么不能写同一个 State Key？

**参考回答：**

同一个 superstep 内若多个节点写 `result`，框架无法确定覆盖顺序，可能抛并发更新错误，也可能产生非确定结果。因此三个验证器分别写 `llm_validation_result`、`rule_validation_result` 和 `parallel_mcp_validation`，再由唯一的 `merge_validation` 汇总。

若业务确实需要并行追加列表，可以为字段声明 Reducer。Reducer最好满足结合律；当调度顺序不保证时还应满足交换律，这样重放、分批合并和并行顺序变化不会改变结果。若顺序有业务含义，应写入稳定排序键并在汇合后排序，而不是依赖执行先后。

### 问题 8：什么是 LangGraph 的 Superstep？

**参考回答：**

Superstep 是一批可同时执行的节点：它们读取上一稳定状态，完成后统一提交更新，再进入下一步。`advise` 后的三个 Validator 属于同一 superstep，`merge_validation` 是下一步的 fan-in。

某分支异常时是否保留其他分支结果取决于 Checkpointer 和执行阶段；支持 pending writes 的持久化实现可以记录已完成写入，恢复时减少重复执行。不能把所有 Saver 都假设为相同语义，必须用故障注入测试验证。fan-out 由一个节点指向多个节点实现，fan-in 使用多起点边 `add_edge([a,b,c], merge)` 等待全部完成。

### 问题 9：如何控制 State Schema 演进？

**参考回答：**

State 中增加带默认值的可选字段通常最容易兼容；删除和重命名必须在恢复入口做迁移。建议保存 `state_version` 和 `graph_version`，在 `_normalize_state` 前按版本逐级迁移，再用 Pydantic 或显式构造恢复领域对象。

迁移脚本应可重复运行、保留原快照并统计失败记录。若节点名称或拓扑发生不兼容变化，运行中的旧任务继续由旧图版本恢复，新任务进入新图；通过蓝绿部署和版本路由完成过渡，而不是让新代码直接读取全部旧 Checkpoint。

## 3. 生成、校验与反思闭环

### 问题 10：为什么拆分 Worker、Validator 和 Tutor？

**参考回答：**

Worker优化召回率和候选生成，Validator优化错误发现，Tutor把证据转成下一轮可执行建议。职责分离使每个节点可使用不同模型、Prompt、温度和评测集，也避免一个模型同时“出题、答题、判卷”造成自洽偏差。

Worker和LLM Validator若使用同一模型，容易共享盲点；生产中应使用不同 Prompt，必要时选择异构或更稳定的评审模型，并用规则/MCP提供独立证据。Validator不一定总用更大模型，要根据误判成本评估。Tutor不直接改答案，是为了保留“诊断”和“生成”的边界，让反思能够沉淀、审核和复用。

### 问题 11：反思如何驱动下一轮生成？

**参考回答：**

`merge_validation`失败后进入 Tutor，Tutor读取样本、候选以及三路验证证据，产生结构化改进建议并更新 `current_reflection`。`advance_round`增加轮次后回到 Worker，Worker在原始任务上下文中加入这条反思重新生成。

当前图保留最新反思，同时完整历史在 `result.iterations_info` 中，避免把所有历史重复塞入 Prompt。若需要多轮信息，应先压缩为“仍未解决约束”。Tutor也可能错误，所以反思必须引用验证证据、经过质量门禁；新一轮仍由独立验证器裁决。可以保留上一轮最佳结果，若分数下降则回退并停止无效迭代。

### 问题 12：如何防止无限反思？

**参考回答：**

系统同时设置最大轮次和总体超时：轮次控制模型调用成本，墙钟超时覆盖外部服务卡顿。超时是失败终态，不能伪装成成功。还可以增加“连续两轮提升低于阈值”“重复错误指纹”和 Token 预算等提前终止条件。

最后一轮失败仍执行 Tutor，是为了给人工诊断和数据飞轮留下失败原因，然后由 `persist_reflection` 路由到 `finish_max_rounds`。但若失败来自 MCP 不可用、限流等基础设施问题，应标记为系统错误，不生成业务反思，避免污染知识库。

### 问题 13：为什么通过阈值是 80？

**参考回答：**

80 是当前配置中的质量门槛，不应声称它天然正确。正确做法是在标注集上比较分数和人工“可用/不可用”标签，根据业务更关注漏放还是误杀，选择满足目标 Precision、Recall 或成本函数的阈值，并按故障类型校准。

硬规则和 Schema 错误不能被高 LLM 分数抵消。人工评价与模型分数冲突时，先分析标注一致性，再做分数校准、Prompt调整或增加独立评审；线上还应监控各分段真实通过率和人工退回率。

### 问题 14：多种校验结果冲突时如何处理？

**参考回答：**

示例中最终不通过。API调用顺序和参数 Schema 是硬约束，任一个失败即可否决；LLM的90分只是软语义证据，不能简单平均。当前实现中 Rule失败会把统一分数置为0，MCP结果再经 `merge_mcp_validation` 合入。

Tutor应同时看到统一结论和三个原始证据，才能给出具体修复建议。结果中应保存验证器版本、原始输出、规则 ID、失败字段、MCP响应摘要、时间戳和 trace ID，以便审计与复现。软指标可加权，硬约束应采用 veto 或分层门禁。

### 问题 15：为什么最后一轮失败仍然生成反思？

**参考回答：**

它不能挽救当前任务，但能解释最终失败、帮助人工修订，并作为候选经验进入数据飞轮，使相似任务减少重犯。是否入库仍要经过质量评分、去重和人工审核。

只有业务质量失败才适合反思；超时、网络错误、鉴权失败或模型限流应进入运维事件和重试队列，不能转成领域知识。

## 4. 并行、并发与资源治理

### 问题 16：为什么三个 Validator 可以并行？

**参考回答：**

三者都只依赖 Advisor 输出和同一份只读上下文，彼此没有数据依赖，所以适合 fan-out。串行理论耗时约为 `5 + 2 + 0.01 = 7.01s`，并行关键路径约为 `max(5,2,0.01) + 调度/汇合开销 ≈ 5s`。

风险包括并发 State 写冲突、外部限流、连接池耗尽、部分分支失败、Trace乱序和恢复语义复杂化，因此每个分支使用独立字段、独立超时和统一汇合策略。

### 问题 17：如何证明三个 Validator 真正并行执行？

**参考回答：**

单元测试用三个替身节点同时等待 `threading.Barrier(3)`；若框架串行，第一个节点会超时，全部通过说明它们确实同时进入。再记录单调时钟的开始/结束区间，断言区间重叠。

单纯比较总耗时容易受缓存、网络和机器负载影响，只能作为辅助。线上应为每个节点创建 trace span，通过相同 iteration 的时间轴和并发区间观察，并同时监控各分支P95、错误率和外部限流。

### 问题 18：某个并行分支超时怎么办？

**参考回答：**

fan-in默认要等待全部前驱，因此每个外部分支必须设置自己的超时并把结果转成明确的 `passed/error/timeout/skipped` 状态。是否 fail-open 取决于风险：关键 Schema 或执行安全校验应 fail-closed；仅提供补充质量信息的远程服务可降级，但结果必须标记“未验证”，不能等同通过。

连续超时应触发熔断，半开探测恢复；同时限制重试次数并使用指数退避和抖动。总体超时再为全图兜底。

### 问题 19：Pipeline Pool 大小为什么是 3？

**参考回答：**

3 是当前配置默认值，不是通用最优值。容量受模型 QPS/TPM、MCP连接数、内存和同步线程数共同约束，应压测后设定。Pipeline封装了客户端、知识库和SQLite连接，不假定其可被多个请求同时安全使用，因此借出期间由单个任务独占。

100个任务到达时由线程池和有界资源池排队。生产中还要增加有界提交队列、等待超时、429/503拒绝策略、租户配额和动态并发控制，避免无限排队耗尽内存。

### 问题 20：同步与异步如何选择？

**参考回答：**

当前图使用同步 `invoke/stream`，服务通过线程池隔离阻塞工作。若直接在FastAPI异步端点调用同步模型、MCP或SQLite，会阻塞事件循环。`stream()`是同步迭代，`astream()`允许异步消费，但只有底层模型、HTTP、数据库和节点都异步时才获得完整收益。

真正异步化需要改造 LLM Client、MCP Client、Retriever、Checkpointer和节点签名，并使用异步连接池。CPU密集规则仍应进线程池/进程池，不能只把函数名改成 `async`。

## 5. Checkpoint 与故障恢复

### 问题 21：为什么系统需要 Checkpointer？

**参考回答：**

Checkpoint保存每个 thread 的状态、下一节点和必要写入，使长流程在进程异常、人工中断或外部失败后继续，也支持查看状态历史。若Worker完成但Checkpoint前崩溃，恢复时Worker可能重跑，所以系统语义通常是 at-least-once，不是严格 exactly-once。

Checkpoint只能保证图状态持久化，无法自动把文件、外部API和业务数据库纳入同一事务。所有副作用必须幂等，或使用事务Outbox、唯一键和补偿逻辑。

### 问题 22：解释四种 ID

**参考回答：**

- `task_id`：HTTP任务实例，供查询和SSE使用。
- `thread_id`：LangGraph执行上下文键，用于定位Checkpoint历史。
- `session_id`：业务会话，可跨多个任务复用并关联已有API。
- `checkpoint_id`：某个thread中的具体状态快照版本。

当前服务让 `task_id == thread_id`，因为一次提交对应一次图执行，定位简单。`session_id`不能直接当 `thread_id`，否则同一业务会话的多次任务会共享并污染状态。查询完整历史应以 `(thread_id, checkpoint_id)` 遍历Saver历史，并记录graph版本和业务关联ID。

### 问题 23：为什么本地选择 SQLite？

**参考回答：**

SQLite零运维、单文件、事务完整，适合本地开发和单机演示。WAL允许读写更好地并发，`synchronous=NORMAL`在性能和持久性间折中。当前每个Pipeline有独立连接但共享文件，SQLite负责文件级协调；高并发写入仍可能锁竞争。

多进程、多副本、网络文件系统和高写吞吐场景不适合SQLite，生产建议PostgreSQL以获得连接池、行级并发、备份和高可用。Redis适合缓存、锁和短期事件，但默认持久性、复杂查询及审计能力不如关系数据库，因此不作为唯一Checkpoint真相源。

### 问题 24：并行分支失败后如何恢复？

**参考回答：**

使用相同 `thread_id` 调用 `graph.invoke(None, config)`，新Pipeline实例通过共享SQLite找到最新快照并从待执行位置继续。若Saver支持并保存pending writes，已成功分支可以不重跑；否则可能重跑整个superstep，所以分支必须可重试或幂等。

验证方式是故障注入：让一个Validator首轮抛错，为三个分支计数，重建Pipeline后恢复，并断言最终成功、Checkpoint来自原thread，同时检查成功分支是否重跑。不能只凭文档假设恢复粒度。

### 问题 25：Checkpoint 中的自定义对象如何序列化？

**参考回答：**

JSON/MessagePack恢复后，映射键可能字符串化，嵌套Dataclass或Pydantic对象也可能变成普通dict。当前代码通过 `_normalize_state`、`_normalize_result` 和 `_current_iteration` 显式重建对象并兼容整数/字符串轮次键。

序列化器使用明确的类型白名单，减少任意类型加载风险。任意pickle反序列化可能执行恶意代码，也会把Checkpoint绑定到不稳定的Python类路径，不适合作为跨版本协议。

### 问题 26：如何处理 Checkpoint Schema 升级？

**参考回答：**

Checkpoint应同时记录 `state_version`、`graph_version` 和模型/Prompt版本。兼容新增字段可在标准化层补默认值；破坏性变更通过离线迁移或旧版本读取器转换。节点重命名和拓扑改变可能使“下一节点”无法映射，不能只迁移字段。

部署时让旧任务继续路由到旧图，新任务进入新图；确认旧任务结束或迁移成功后再下线旧版本。这比直接滚动替换更安全。

## 6. 幂等性与一致性

### 问题 27：反思候选如何避免重复写入？

**参考回答：**

当前 `source_id` 由 `reflection_candidate:{order_id}:{iteration_id}` 生成；先检查本次 `result.reflection_candidates`，再通过Store的 `find_by_source_id` 检查持久层，存在则不追加。因为节点可能在Checkpoint提交前完成外部写入，恢复后会再次执行，所以必须幂等。

JSONL在多进程并发、原子性和唯一约束方面较弱。生产中使用数据库表，以 `source_id` 建唯一索引，执行 `INSERT ... ON CONFLICT DO NOTHING/UPDATE`，并把状态和审计时间一并保存。

### 问题 28：能否实现严格 Exactly-once？

**参考回答：**

跨LangGraph Checkpoint、业务库、文件和外部API的严格exactly-once成本很高，除非它们共享同一事务边界。更现实的方案是at-least-once执行加幂等副作用。

同库操作可放入一个事务；跨系统使用事务Outbox：业务事务同时写结果和待发送事件，后台可靠投递，消费者按event ID去重。Checkpoint、结果写入要求强一致或可恢复；指标、通知、缓存通常允许最终一致。

### 问题 29：同一个 Thread ID 被两个请求同时执行怎么办？

**参考回答：**

两个执行者可能从同一Checkpoint读取并并发推进，造成重复模型调用、版本竞争和副作用重复。应在任务入口按thread获取租约，包含owner、过期时间和fencing token；只有租约持有者可提交业务副作用。

单机可用应用锁，单数据库部署可用行锁/顾问锁，多副本使用数据库租约或可靠分布式锁。恢复请求也走同一租约，超时后才能由新实例接管。

### 问题 30：节点副作用应该放在什么位置？

**参考回答：**

把计算和副作用拆成不同节点：先得到稳定业务结果，再通过专门的persist/publish节点写外部系统。这样重试策略、权限、审计和补偿更清晰。`interrupt()`之前的节点恢复后可能从头执行，因此其副作用必须幂等。

API查询用请求ID和幂等键；文件写入采用临时文件加原子替换或唯一记录；数据库使用唯一约束和事务。不能依赖“正常情况下只调用一次”。

## 7. RAG 与知识检索

### 问题 31：为什么需要三种知识源？

**参考回答：**

领域知识说明故障机理和排障策略；历史反思知识记录曾经失败及修复经验；API Schema约束工具名称、参数和返回结构。三者可信度、更新频率和过滤维度不同，所以需要metadata区分来源、故障类型、API名和版本。

Schema必须按目标API和版本强过滤，否则相似但错误的参数定义会造成高风险幻觉。冲突时以当前版本Schema和审核过的领域规则为硬事实，反思只作为经验建议；冲突应记录并触发知识治理。

### 问题 32：解释加权 RRF

**参考回答：**

RRF只使用各检索器的排名：某文档得分可写成 `Σ w_s /(k + rank_s)`。它避免直接比较FAISS余弦分、BM25分和不同知识源分数，因为这些分数尺度并不一致。

`k`越大，头部名次差异被平滑；越小，第一名优势越明显。来源权重通过离线检索集和最终任务指标调优，并设置每源候选上限、覆盖约束和去重，防止某一知识源长期垄断上下文。

### 问题 33：Adaptive RAG Router 如何工作？

**参考回答：**

Router根据任务特征决定检索哪些源及top-k：API数量多意味着Schema约束更复杂，排障步骤多意味着领域上下文需求更大，历史反思命中则增加反思检索。当前使用规则是因为便宜、稳定、可解释，且特征少。

路由结果应保存决策、特征、每源预算和原因。通过固定top-k与自适应策略的A/B测试，比较Recall@K、最终通过率、Prompt Token和延迟；只有质量不降且成本更低，才能证明有效。

### 问题 34：如何评估 RAG？

**参考回答：**

Recall@K衡量相关证据是否被召回；MRR关注首个相关结果位置；Source Coverage检查所需知识源是否齐全；metadata准确率检查过滤后是否仍属于正确API/版本；最终生成正确率反映端到端价值。

构建带“任务—必要证据—正确API结果”的独立标注集，按故障类型和难度分层。召回正确但生成错，查看上下文是否进入Prompt以及Worker是否遵循；召回错误则分析路由、过滤、切块和排序。评测集按时间或实体隔离，避免历史反思从测试答案泄漏。

### 问题 35：是否应该把三源检索也改成并行图？

**参考回答：**

是否改图取决于延迟和独立性。三个远程pgvector/服务查询可以并行，收益接近最长分支而非耗时之和；本地FAISS查询很快且可能争用CPU/内存，图调度开销反而可能更大。

并行时每源写独立候选字段，由fusion节点做过滤、去重和RRF。稳定的Schema、文档embedding和按查询特征生成的检索结果可缓存，但Key必须包含知识库版本、过滤条件、router版本和embedding模型版本。

## 8. 反思知识与数据飞轮

### 问题 36：Tutor 反思为什么不能直接进入知识库？

**参考回答：**

Tutor也会幻觉，若错误反思直接被召回，会把一次错误放大成持续性知识污染。当前候选按长度、得分、是否包含可执行修复词等特征形成质量分；高分为 `promoted`，低置信度为 `pending_review`，人工可置为 `approved` 或 `rejected`。

自动阈值应以人工审核集上的准确率和污染成本确定。人工修改不能覆盖原文，应保存原反思、修改稿、审核人、时间、理由和版本，形成完整审计链。

### 问题 37：如何证明数据飞轮有效？

**参考回答：**

在相同测试集、模型、Prompt、温度和调用预算下比较：A不使用反思，B使用未审核反思，C只使用审核通过反思。主要指标是一次通过率、最终通过率、平均轮次、硬规则失败率、人工修订率、Token成本和P95延迟。

结果还要按故障类型和新旧样本分层，并用置信区间或显著性检验报告。如果C优于A且B出现退化，就能同时证明反思价值和审核门禁价值；不能只展示几个成功案例。

### 问题 38：反思知识何时会失效？

**参考回答：**

Schema、设备版本、业务规则、模型Prompt或故障定义变化都会使反思失效。反思应带 `api_version`、适用故障、生成/审核时间、来源和有效期；检索时强过滤版本，变更时批量降权或下线。

用语义聚类和规则指纹去重，对同一适用范围内结论相反的候选建立冲突组，交给人工裁决。线上退回率升高也可触发自动过期。

### 问题 39：如何实现 Human-in-the-loop？

**参考回答：**

Quality Gate对低置信度候选调用 `interrupt()`，LangGraph保存State后立即释放执行线程，不会占用三天。人工通过独立审核系统提交批准、修改或拒绝；服务使用同一thread和Command恢复，审核结果写入State，再进入persist或结束分支。

审核任务要有截止时间、负责人和幂等decision ID。超时可转人工队列或拒绝自动入库。恢复入口仍需租约，防止多次审批同时推进。

## 9. FastAPI 与流式输出

### 问题 40：为什么同时提供轮询和 SSE？

**参考回答：**

轮询简单、兼容性好，适合查看最终状态；SSE为长任务提供单向实时节点进度，基于HTTP且自动重连，比WebSocket更适合“服务端持续推送、客户端很少上行”的场景。WebSocket适合高频双向交互。

当前事件带递增 `index`，客户端以 `after`续传。生产中事件不能无限保存在内存，应持久化到事件表或带保留期的Stream，并按任务设置TTL、最大数量及最终摘要。

### 问题 41：Checkpoint 已持久化，但 Task Registry 仍在内存中怎么办？

**参考回答：**

当前服务重启后，图状态仍在SQLite，但任务列表、HTTP状态和SSE事件会丢失，因此只能知道thread时主动查Checkpoint，不能完整恢复任务中心。Checkpoint也不一定包含提交者、队列状态和服务owner，不能完全替代Task Registry。

生产中把任务表持久化，记录task/thread、状态、租约owner、heartbeat、版本和时间；启动时扫描 `pending/running` 且租约过期的任务，重新入队或标记待恢复。实例定期续租，从而识别失联执行者。

### 问题 42：如何避免 SSE 遗漏终态事件？

**参考回答：**

先在同一临界区写入结果、追加终态事件，再把状态更新为终态并通知等待者。当前实现对成功和失败都在同一Condition锁内完成这些动作，订阅端取出pending events后才判断结束。

每个订阅者维护自己的 `after/index`，因此是广播式读取，不是竞争消费。生产事件表以 `(task_id, sequence)` 唯一，状态和Outbox事件最好同事务提交，断线后按sequence补读。

### 问题 43：如何设计任务状态机？

**参考回答：**

允许的主路径是 `pending → running → success/failed/timeout/interrupted/cancelled`，其中成功、失败、超时和取消通常是终态；`interrupted`可在人工输入或依赖恢复后回到running。当前枚举只实现pending、running、success、failed和not_found，生产版需要补齐状态。

所有转移通过一个函数校验允许边，并使用数据库条件更新，例如 `UPDATE ... WHERE status='running'`，避免并发覆盖。恢复不是把failed静默改成running，而应创建恢复记录或记录retry_count与前序原因。

## 10. 性能、成本与容量

### 问题 44：每天生成 100 万条数据如何扩展？

**参考回答：**

100万/天平均约11.6条/秒，但峰值、每条多轮和多次LLM调用才决定容量。架构上由API接收任务并写消息队列，无状态Worker消费，任务表和Checkpoint迁移到PostgreSQL，对样本分片并按租户/模型配额调度；SQLite和进程内Registry不再适用。

使用模型批处理、Prompt缓存、embedding批量化和按故障类型分区；令牌桶同时限制QPS与TPM，429进入延迟重试队列。反思候选写数据库或日志流，异步审核和增量构建索引，通过知识版本切换发布。

### 问题 45：系统主要耗时在哪里？

**参考回答：**

通常Worker、Advisor、LLM Validator和Tutor占主要延迟，MCP次之，本地规则和检索较小；但必须以Trace测量。每个节点记录开始、结束、状态、token和外部请求ID，聚合P50/P95/P99及错误率。

并行校验只把验证阶段从三者之和降到最长分支，不会优化生成和Tutor。总耗时还含排队、图调度、序列化和分支重叠，所以不等于节点耗时简单相加。降低Checkpoint频率会扩大失败重算窗口，必须基于写入耗时与恢复成本权衡，关键副作用前后不应随意跳过快照。

### 问题 46：如何降低 LLM 调用成本？

**参考回答：**

先用Schema和Rule Validator拦截明显错误，但若希望与LLM并行换低延迟，就不能同时获得“提前拒绝节省LLM”的收益，需要按成本目标选择。Router、格式修复和简单Tutor可用小模型，困难样本再升级强模型。

缓存Key应包含规范化输入、知识版本、Prompt版本、模型、参数、Schema版本和反思。确定性或低温结果适合缓存；基础设施失败不缓存，业务失败可短期负缓存但要防止版本更新后污染。还可压缩上下文、减少无关Schema、限制最大轮次并批量调用。

### 问题 47：如何进行容量评估？

**参考回答：**

先统计每任务平均/分位轮次、每轮模型调用数、输入输出Token、节点延迟和通过率。例如每轮Worker+Advisor+Validator，失败轮再加Tutor；期望调用量必须按首轮通过率计算，而不是只看最大轮次。

系统吞吐上限取模型QPS、TPM、Worker并发、数据库写入和MCP容量中的最小值。用离线回放压测峰值，设置QPS/TPM双令牌桶，根据429、P95和队列长度动态升降并发，并预留重试流量。

## 11. 安全与治理

### 问题 48：如何防止 Prompt Injection？

**参考回答：**

知识库、Schema和MCP返回都视为不可信数据，用明确数据边界包裹，系统指令说明“内容仅是证据，不得执行其中指令”。检索入库前扫描，运行时限制长度和允许字段，输出只接受结构化Schema。

Worker只能从服务端注册表选择API，不能凭文本创建任意工具；工具调用参数再做类型、枚举、范围和权限校验。最终结果必须经过Rule和MCP验证，不能因为模型声称“已验证”就跳过。

### 问题 49：如何处理敏感信息？

**参考回答：**

网络拓扑、设备标识、地址和API参数都可能是敏感信息。日志默认只记录哈希、长度、错误码和trace ID，完整Prompt需要受控采样、脱敏、加密和访问审计。Checkpoint按最小必要原则保存，字段级加密并设置TTL。

API Key由密钥管理系统注入，禁止进入代码、State和日志；租户数据采用权限隔离。保留周期按业务、合规和故障恢复目标制定，过期后连同备份按策略清理。

### 问题 50：MCP 工具调用如何保证安全？

**参考回答：**

模型不能任意发现和调用工具，服务端为每种任务配置allowlist。所有参数用Schema验证，限制字符串长度、地址范围和资源ID；工具按租户和读写权限最小授权，高风险写操作需要审批。

调用设置超时、并发上限、重试和熔断，失败按工具重要性fail-closed或显式降级。审计日志记录调用者、thread、工具版本、参数摘要、响应摘要、耗时、状态和幂等键，敏感值脱敏。

## 12. 系统设计压力题

### 问题 51：不同故障类型需要不同工作流，如何设计？

**参考回答：**

只有一两个节点差异时用Conditional Edge；一组步骤、State或恢复边界整体不同，则抽成子图。不要为每个故障复制整张图，可建立公共检索、验证和终止节点，通过故障策略注册表选择专属子图。

图发布时生成不可变 `graph_version`。新任务按配置进入新版本，运行中任务继续用创建时版本；紧急兼容变更才做显式迁移。

### 问题 52：如何拆分子图？

**参考回答：**

`Retrieval Subgraph`负责路由、三源检索与融合；`Validation Subgraph`负责三路fan-out/fan-in和统一结论；`Reflection Subgraph`负责Tutor、质量门禁、人工中断和持久化。父图只传子图所需的输入输出State，避免所有字段全局共享。

持久化语义取决于子图编译方式：作为父图节点时通常沿用父图执行上下文，需要跨调用记忆的子图才配置独立持久层和命名空间。每个子图用伪造边界State独立测试，再做父子图恢复测试。

### 问题 53：要求支持取消任务，如何设计？

**参考回答：**

任务表保存 `cancel_requested`，每个节点开始前和长操作边界检查；路由到 `finish_cancelled`，并禁止后续反思持久化。对于支持取消的异步HTTP/模型客户端传播取消信号；不支持时只能停止等待并忽略带旧fencing token的迟到结果。

取消是用户主动终止，超时是预算耗尽，两者审计、重试策略和终态不同。取消请求本身必须幂等，并与租约机制配合。

### 问题 54：要求支持多租户，如何设计？

**参考回答：**

使用 `(tenant_id, task_id)` 形成thread命名空间，任何查询都从认证上下文注入tenant，不能信任请求体。Checkpoint和任务表启用行级安全或独立Schema；向量库至少使用强制tenant metadata过滤，高安全租户可物理隔离。

调度层为每租户设置并发、QPS、TPM、存储和知识库配额，采用公平队列防止大租户挤占。缓存Key、审计日志和加密密钥也必须包含租户边界。

### 问题 55：当前系统距离生产可用还差什么？

**参考回答：**

当前版本已经具备显式图编排、并行Validator、SQLite Checkpoint、恢复、轮询和SSE，适合单机验证。主要缺口是：Task Registry与事件仍在内存、SQLite和JSONL不适合多副本、执行链仍以同步线程为主、缺少分布式租约、State/图版本迁移、完整指标告警、端到端金标评测以及多租户安全治理。

优先级应是：先持久化任务与事件并建立幂等/租约，接着迁移PostgreSQL和异步外部客户端，再补充质量评测、成本监控、权限脱敏和反思审核平台。不能只用“换数据库”概括生产化。

## 13. 现场代码题参考答案

### 代码题 1：校验路由

```python
from typing import Literal

Route = Literal["passed", "timeout", "reflect"]

def route_after_validation(state: dict) -> Route:
    if state.get("timed_out") is True:
        return "timeout"

    # 分数缺失、布尔值或不可解析时一律不放行。
    raw_score = state.get("validator_score")
    try:
        if raw_score is None or isinstance(raw_score, bool):
            raise ValueError("invalid score")
        score = float(raw_score)
    except (TypeError, ValueError):
        return "reflect"

    hard_checks_passed = bool(state.get("rule_passed", False)) and bool(
        state.get("mcp_schema_passed", False)
    )
    return "passed" if score >= 80 and hard_checks_passed else "reflect"
```

关键点是超时优先，解析失败fail-closed，并且硬约束不能被LLM分数抵消。实际项目的超时由起始时间和总预算判断，分数来自合并后的 `IterationResultModel`。

### 代码题 2：并行 Validator 图

```python
from typing import TypedDict
from langgraph.graph import START, END, StateGraph

class ValidationState(TypedDict):
    candidate: list[dict]
    llm_result: dict
    rule_result: dict
    mcp_result: dict
    merged: dict

def llm_validate(s):
    return {"llm_result": call_llm_validator(s["candidate"])}

def rule_validate(s):
    return {"rule_result": run_rules(s["candidate"])}

def mcp_validate(s):
    return {"mcp_result": call_mcp(s["candidate"])}

def merge(s):
    return {"merged": merge_results(
        s["llm_result"], s["rule_result"], s["mcp_result"]
    )}

g = StateGraph(ValidationState)
for name, node in {
    "llm": llm_validate, "rule": rule_validate, "mcp": mcp_validate
}.items():
    g.add_node(name, node)
g.add_node("merge", merge)
g.add_edge(START, "llm")
g.add_edge(START, "rule")
g.add_edge(START, "mcp")
g.add_edge(["llm", "rule", "mcp"], "merge")
g.add_edge("merge", END)
app = g.compile(checkpointer=checkpointer)
```

三个分支写独立Key，因此无冲突；多起点边使merge等待全部完成。失败恢复必须使用相同thread，分支本身保持幂等，并用故障注入确认Saver是否复用pending writes。

### 代码题 3：反思循环

```python
def route_validation(state):
    return "passed" if state["passed"] else "reflect"

def route_reflection(state):
    return "stop" if state["iteration"] >= state["max_rounds"] else "retry"

builder.add_conditional_edges(
    "validate", route_validation,
    {"passed": "finish", "reflect": "tutor"},
)
builder.add_edge("tutor", "persist_reflection")
builder.add_conditional_edges(
    "persist_reflection", route_reflection,
    {"stop": "finish_failed", "retry": "advance_round"},
)
builder.add_edge("advance_round", "worker")
```

Tutor先更新 `current_reflection`，persist节点无论是否最后一轮都会运行；只有持久化后才判断进入下一轮或失败结束。总体超时应在验证路由中拥有更高优先级。

### 代码题 4：幂等反思写入

```python
def persist_reflection(state: dict) -> dict:
    source_id = f"reflection_candidate:{state['task_id']}:{state['iteration_id']}"
    candidate = {
        "source_id": source_id,
        "reflection": state["current_reflection"],
        "status": "pending_review",
    }
    try:
        inserted = reflection_repo.insert_if_absent(candidate)
    except Exception as exc:
        return {"persist_ok": False, "persist_error": str(exc)}
    return {"persist_ok": True, "reflection_source_id": source_id,
            "newly_inserted": inserted}
```

数据库实现使用 `source_id UNIQUE` 和 `INSERT ... ON CONFLICT DO NOTHING`。JSONL版本只能作为单机原型；迁移时导入后按source_id去重，并以数据库为唯一真相源。

### 代码题 5：恢复失败任务

```python
def resume_task(thread_id: str):
    with lease_repo.acquire(thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise KeyError(f"checkpoint not found: {thread_id}")
        final_state = graph.invoke(None, config=config)
        if not final_state:
            final_state = graph.get_state(config).values
        return final_state["result"], bool(final_state["success"])
```

`lease_repo.acquire`在单机可用互斥锁，生产中使用带过期和fencing token的数据库租约。新实例必须连接同一Checkpoint库、加载兼容图版本，并对恢复State做类型标准化。

### 代码题 6：SSE 断线续传

```python
def events(task_id: str, after: int = 0):
    cursor = max(0, after)
    while True:
        batch = event_repo.list_after(task_id, cursor)
        for event in batch:
            cursor = event["sequence"]
            yield {**event, "id": cursor}
        status = task_repo.status(task_id)
        if status in TERMINAL_STATUSES and not event_repo.exists_after(task_id, cursor):
            return
        if not batch:
            yield {"event": "heartbeat", "id": cursor}
```

事件以 `(task_id, sequence)` 唯一并持久化，客户端通过Last-Event-ID或`after`续传。终态状态与终态事件应使用同一事务/Outbox提交，读完游标之后才能关闭连接。

## 14. 综合开放题参考答案

1. **最体现技术能力的设计：** 三路Validator使用独立State通道并行执行，再由fan-in统一裁决；同时把失败反思、Checkpoint和幂等写入串成可恢复闭环。这体现的不只是会调用模型，而是能处理并发状态和可靠性。
2. **最难排查的问题：** Checkpoint恢复后嵌套对象和轮次Key类型变化，导致正常执行和恢复执行行为不同。通过最小故障注入、检查快照和统一 `_normalize_state` 解决，而不是在每个节点零散兼容。
3. **不理想的设计：** Task Registry和SSE事件仍在内存，图状态虽然持久化，服务重启后任务视图不完整；JSONL反思Store也不适合多进程。
4. **再给两周优先改什么：** 先将任务、事件、反思和Checkpoint迁移到PostgreSQL，补任务租约与幂等约束；随后建立端到端评测和节点级可观测性，因为没有可靠运行和量化质量，其他优化无法验证。
5. **如何证明优于一次RAG+LLM：** 在同一金标集和调用预算下比较硬规则通过率、最终通过率、平均人工修订、成本和延迟；再做去掉Advisor、并行Validator、Reflection的消融实验。
6. **去掉LangGraph会失去什么：** 功能仍能手写，但会失去统一的State更新语义、可视化拓扑、条件循环、fan-out/fan-in、节点流事件、Checkpoint历史和标准恢复入口，需要重新自研。
7. **最大稳定性风险：** 多副本下同thread重复执行及外部副作用重复，而当前内存Registry无法跨实例协调。
8. **最大性能瓶颈：** 多轮LLM调用及其限流，尤其Worker、Advisor、Validator和失败后的Tutor；需要以Trace分位数验证。
9. **最大质量风险：** 错误反思进入知识库后被持续召回，形成放大回路；必须版本过滤、审核、离线实验和快速回滚。
10. **如何证明不是为了框架而使用框架：** 把需求映射到能力：多轮循环对应条件边，独立校验对应并行汇合，长任务对应Checkpoint/恢复，人工审核对应interrupt。若这些需求被删除，就应退回更简单的函数编排。

## 15. 面试回答原则

- 先说业务目标，再说LangGraph如何解决具体控制流问题。
- 明确区分当前实现、已验证能力和生产化设想，不把规划描述成现状。
- 所有“提升百分比”都必须来自实验；没有数据时说明指标和实验设计。
- 遇到Exactly-once、并行恢复和异步等问题，要说明边界条件，避免绝对化。
- 回答每道题可采用“结论 → 项目证据 → 取舍 → 演进方案”的结构。