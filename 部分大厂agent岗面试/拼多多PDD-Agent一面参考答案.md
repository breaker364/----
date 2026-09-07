# 拼多多（PDD）Agent 一面：问题与参考答案

> 来源：用户提供的“pdd agent 一面”截图，面试时长约 40 分钟。项目和经历题需替换成真实内容；以下答案用于组织技术表达。

## 一、项目与基础

### 1. Agent 怎么实现不同任务的隔离？

隔离要覆盖身份、上下文、工具权限、状态和资源。每次请求生成 `tenant_id/conversation_id/task_id/run_id`，状态、缓存、向量检索和日志都带租户与权限过滤；不同任务使用独立的工作区、临时目录和工具白名单，不能共享可变的 Prompt 历史。

运行时用状态机把任务状态限定为 `queued/running/waiting/succeeded/failed/cancelled`，通过租约和幂等键防止重复执行。高风险工具采用沙箱、最小权限和人工确认；并发上设置 CPU、内存、token、时长和下游 QPS 配额。多租户还要防止缓存 key、trace 和长期记忆串租户。

### 2. Agent 如何保证分析、输出结果准确？

把“准确”拆成可测指标：意图准确率、检索召回率、引用正确率、工具参数正确率、任务完成率和事实一致性。流程上采用结构化输出、证据绑定和程序化验收：模型先给结论、证据 ID 和置信度；服务端校验 Schema、数值范围、权限和业务规则；关键结论由规则、数据库状态或第二个校验器复核。

无证据或低置信度时明确拒答、追问或转人工。离线用覆盖正常、边界、对抗、无答案和工具失败的黄金集回归；线上采集 trace 和人工抽检，按 badcase 类型修检索、Prompt、工具或模型，不把所有问题归因于模型。

### 3. Agent 评测体系，测试用例覆盖范围？

评测分四层：

1. **组件层**：分类、Embedding、召回、Rerank、Schema 解析、工具适配器；
2. **链路层**：单步、多步、并行、重试、超时、恢复和人工审批；
3. **业务层**：任务完成率、答案相关性、事实/引用正确率、人工修改率；
4. **生产层**：P50/P95 延迟、token 成本、工具成功率、错误率、转人工率和安全拦截率。

用例覆盖高频任务、长尾表达、缺字段、矛盾约束、无答案、权限越界、Prompt 注入、下游 4xx/5xx、重复事件、超长上下文和模型降级。每次 Prompt、模型、工具或索引变更都跑版本化回归集，并保留 trace 供复现。

### 4. 完整业务流程讲一下

以企业知识问答为例：请求鉴权 → 意图/风险路由 → 查询改写 → 权限过滤后的混合检索 → Rerank → 上下文组装 → LLM 生成带引用的答案 → 引用/Schema/安全校验 → 返回或澄清。

若需要执行动作，则进入计划和工具循环：生成结构化步骤 → 校验依赖和权限 → 调用工具 → 写回 observation → 判断完成或有限重试/重规划 → 最终检查真实副作用。异步长任务通过消息队列和 checkpoint 驱动，失败进入可重放的死信队列；所有阶段写入 trace、指标和审计记录。

### 5. LangChain 与 LangGraph 的核心区别？

LangChain 更像 LLM 应用组件和编排抽象，适合把 Prompt、模型、解析器、Retriever、Tool 串成线性 Runnable/Agent；LangGraph 以有向图和共享 State 为核心，把节点、条件边、循环、checkpoint、暂停恢复和人工介入显式化。

固定的“检索 → 生成”流水线用 LangChain 足够；有自纠、审批、多 Agent 协作或长任务恢复需求时更适合 LangGraph。两者可以组合：LangChain 提供模型和工具，LangGraph 管生命周期。不要把 ReAct（策略）和框架（运行时）混为一谈。

### 6. 使用什么 Agent 开发框架？自研框架和开源框架的差异？

回答应基于真实项目。通用选型是：先用开源框架快速验证模型、工具和图编排；当出现统一鉴权、多租户、计费、调度、回放、协议兼容或性能瓶颈，再在内部建设 Runtime/Harness，并保留与开源组件的适配层。

自研优势是协议、状态、权限、观测和性能可按业务定制，代价是升级、兼容、测试和运维责任都由团队承担；开源优势是生态和迭代快，但要评估版本稳定性、扩展点、许可证和供应商锁定。不要因为“自研”就忽略复用成熟组件。

## 二、手撕：线程安全的 LRU

### 题目理解

实现固定容量缓存，`get` 命中后变为最近使用，`put` 插入/更新也变为最近使用，超过容量淘汰最久未使用项，并保证多线程并发下链表和哈希表状态一致。

### 参考实现（Python）

```python
from collections import OrderedDict
from threading import RLock


class ThreadSafeLRU:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = OrderedDict()
        self._lock = RLock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            value = self._data.pop(key)
            self._data[key] = value       # 移到 MRU 端
            return value

    def put(self, key, value) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)  # 淘汰 LRU

    def __len__(self):
        with self._lock:
            return len(self._data)
```

面试中也可以手写“哈希表 + 双向链表”：哈希表 O(1) 定位节点，链表头是最近使用、尾是最久未使用；`get/put` 的移动、插入、删除都在同一把锁内完成。读操作也会改变 LRU 顺序，因此不能只给写操作加锁。若需要高并发，可按分片拆锁，但要说明跨分片容量和淘汰语义会复杂化。

### 复杂度与测试

`get`、`put` 的平均时间复杂度为 O(1)，空间 O(capacity)。测试至少覆盖：容量为 1、更新已有 key、命中后顺序变化、淘汰顺序、缺失 key、并发读写和异常输入；并发测试应检查不会出现长度超过容量、丢更新或迭代器异常。
