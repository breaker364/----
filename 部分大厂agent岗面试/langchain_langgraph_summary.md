# LangChain 与 LangGraph 总结

> 本文基于一系列实操问答整理而成，覆盖 Chain 是否支持循环、`create_agent` 的归属、LangGraph 的 State 模型、节点本质，以及多个可运行示例。目标是给出一套**清晰、可落地**的心智模型。

---

## 目录

1. [核心结论：LangChain vs LangGraph](#1-核心结论langchain-vs-langgraph)
2. [Chain 是否支持循环？](#2-chain-是否支持循环)
3. [`create_agent` 是谁？是否支持循环？](#3-create_agent-是谁是否支持循环)
4. [LangGraph 构建 Agent 的函数](#4-langgraph-构建-agent-的函数)
5. [LangGraph 的状态（State）模型](#5-langgraph-的状态state-模型)
6. [节点（Node）是什么？](#6-节点node-是什么)
7. [State 是全图共享状态吗？](#7-state-是全图共享状态吗)
8. [可运行示例汇总](#8-可运行示例汇总)
   - 8.1 最小 ReAct Agent（`create_react_agent`）
   - 8.2 手写 StateGraph 等价 ReAct 环
   - 8.3 自纠循环（代码→测试→修复）
   - 8.4 人工审批（interrupt + resume）
   - 8.5 多 Agent 协作（Supervisor）
   - 8.6 带 Checkpointer 的多轮记忆
9. [选型对照表](#9-选型对照表)
10. [一句话心智模型](#10-一句话心智模型)

---

## 1. 核心结论：LangChain vs LangGraph

| 维度 | LangChain | LangGraph |
|---|---|---|
| 定位 | LLM 应用编排框架（Chain / Agent / Tool / Memory） | 基于**有向图**的 Agent 运行时，主打可控循环与状态 |
| 控制流 | 线性流水线（LCEL）或黑盒 Agent 循环 | 显式节点 + 边 + 条件边，**循环是一等公民** |
| 状态 | 通过 Memory / 外部传入 | **全局 State（TypedDict + reducer）**，图内原生共享 |
| 适用 | 固定几步流水线（RAG、摘要→翻译） | 自纠环、审批流、多 Agent 协作、需暂停恢复的长流程 |

**一句话**：LangChain 解决"怎么把 LLM/工具串成链"，LangGraph 解决"怎么让链带状态、带循环、带条件分支地跑起来"。`create_agent`（LangChain 1.0）底层就是 LangGraph 编译出的图。

---

## 2. Chain 是否支持循环？

**不支持原生循环。** 分三层看：

1. **LCEL / Chain 层**：`prompt | llm | parser` 是单向 DAG，跑一遍即结束；`RunnableBranch` 只做 if/elif/else，选一条分支后退出。
2. **AgentExecutor 层**：老式 `AgentExecutor`（ReAct）内部有 `while`（思考→选工具→执行→观察），靠 `max_iterations` / `max_execution_time` 防死循环；但这是**运行时黑盒循环**，不是 Chain 声明的控制流。
3. **LangGraph 层**：循环、条件回退、checkpoint 恢复均为原生能力。

```python
# Chain 自身不循环，靠外层 Python while 驱动
for _ in range(max_iter):
    out = chain.invoke(state)
    if out["done"]:
        break
    state = update(state, out)
```

---

## 3. `create_agent` 是谁？是否支持循环？

- **归属**：`langchain.agents.create_agent`（LangChain 1.0 顶层工厂），**底层是 LangGraph**，返回编译好的 `CompiledStateGraph`。不是老式 `AgentExecutor`，也不是 LCEL Chain。
- **循环能力**：支持，但**仅限固定的标准 ReAct 工具调用环**（`model ↔ tools`），由模型决定终止，`recursion_limit` 兜底。
- **局限**：环是预置黑盒，不能在中途插自定义回边；需自定义环时改用 `StateGraph`。

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5",
    tools=[my_tool],
    system_prompt="You are a helpful assistant",
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

**三个易混名字**：
- `langchain.agents.create_agent` → LangChain 1.0 新标准，底层 LangGraph（**推荐**）。
- `langgraph.prebuilt.create_react_agent` → LangGraph prebuilt，1.0 起逐步让位给上面。
- `AgentExecutor` → 更老的 Chain 时代黑盒循环类，已归 legacy。

---

## 4. LangGraph 构建 Agent 的函数

### 预置工厂（一行出图）
- `langgraph.prebuilt.create_react_agent`：LangGraph 自家预置，内部用 `StateGraph` 搭好 `agent → tools → agent…` 回边循环，返回 `CompiledStateGraph`。
- `langchain.agents.create_agent`：LangChain 门面，编译产物也是 LangGraph。

### 底层手写（完全自定义）
- `langgraph.graph.StateGraph`：核心构造器，加节点 / `add_node`、加回边 / `add_edge("tools","agent")`、`add_conditional_edges` 自控循环，最后 `.compile()`。
- 配套：`START`、`END`（来自 `langgraph.graph`）、`MessagesState` 或自定义 TypedDict、`ToolNode`。

---

## 5. LangGraph 的状态（State）模型

State 是**图内所有节点共享、随执行流转的全局字典**。每个节点以完整 state 入参，只返回**自己改动的字段（partial update）**，框架按 reducer 合并。

- Schema 用 `TypedDict`（也可用 Pydantic / dataclass）。
- **未声明 reducer 的字段** → 后写覆盖前写（last-write-wins）。
- **`Annotated[类型, reducer]` 字段** → 按 reducer 合并；`messages` 用 `add_messages` 追加而非替换。

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 追加/按 id 更新
    logs: Annotated[list[str], add]                      # 拼接
    cost: Annotated[int, add]                             # 累加
    step: str                                             # 默认覆盖
```

**循环中的关键作用**：回边每转一圈，模型/工具节点都读写同一份 state；若 `messages` 没挂 `add_messages`，每圈被覆盖，循环推理即断片。

---

## 6. 节点（Node）是什么？

`add_node` 添加的是一个**可调用对象**，最常见是函数，契约为：**输入 state，输出 state 的增量**。

```python
def node_fn(state: State) -> dict:
    return {"field": new_value}  # 只返回改动字段
```

除函数外，还可为：
- **类实例**（实现 `__call__`）；
- **LangChain Runnable**（`prompt | model` 等 Chain）；
- **另一个 LangGraph（子图）**——这是多 Agent 的基础。

> 节点 ≠ 线程/进程：默认单线程顺序执行，并行需显式声明。

---

## 7. State 是全图共享状态吗？

**是的。** 两点归纳：

1. **State 是全图唯一、全局、共享的状态表示**——无论图多复杂，所有节点读同一份 state 的最新快照。
2. **每个节点都在同一份 State 上更新**——只返回增量，LangGraph 按 reducer 合并；边决定"谁在什么时候跑"，节点只负责"做什么"。

```text
node_a  ❌→ node_b
   ↓
 State（全局共享）
   ↓
node_b
```

节点之间无函数调用关系，只通过 State 解耦通信——这是 LangGraph 与"普通 Python 函数调用链"的根本区别。

---

## 8. 可运行示例汇总

### 8.1 最小 ReAct Agent（`create_react_agent`）

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Return fake weather for a city."""
    return f"{city}: 18°C, cloudy"

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    tools=[multiply, get_weather],
    prompt="You are a helpful assistant. Use tools when needed.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "6乘7是多少？"}]})
print(result["messages"][-1].content)
```

### 8.2 手写 StateGraph 等价 ReAct 环

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    return a * b

tools = [multiply]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def agent(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # 循环边
app = builder.compile()
```

执行拓扑：`START → agent ⇄ tools → (不再要工具) END`。

### 8.3 自纠循环（代码→测试→修复）

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class State(TypedDict):
    code: str
    tests: str
    passed: bool

def write_code(s: State) -> dict:
    return {"code": "def add(a,b): return a+b"}

def run_tests(s: State) -> dict:
    return {"passed": "bug" not in s["code"]}

def fix_code(s: State) -> dict:
    return {"code": s["code"] + "  # fixed"}

def route(state: State) -> Literal["fix", "end"]:
    return "fix" if not state["passed"] else "end"

g = StateGraph(State)
g.add_node("write", write_code)
g.add_node("test", run_tests)
g.add_node("fix", fix_code)
g.add_edge(START, "write")
g.add_edge("write", "test")
g.add_conditional_edges("test", route, {"fix": "fix", "end": END})
g.add_edge("fix", "test")  # 自纠回边
app = g.compile()
```

### 8.4 人工审批（interrupt + resume）

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing import TypedDict

class State(TypedDict):
    draft: str
    approved: bool | None

def generate(s: State) -> dict:
    return {"draft": "转账 10000 元"}

def human_approval(s: State) -> dict:
    resp = interrupt({"draft": s["draft"]})
    return {"approved": resp["approve"]}

def route(state: State) -> str:
    return "end" if not state["approved"] else "execute"

def execute(s: State) -> dict:
    return {"draft": s["draft"] + " ✅ 已执行"}

g = StateGraph(State)
g.add_node("gen", generate)
g.add_node("approve", human_approval)
g.add_node("exec", execute)
g.add_edge(START, "gen")
g.add_edge("gen", "approve")
g.add_conditional_edges("approve", route, {"end": END, "execute": "exec"})
g.add_edge("exec", END)
app = g.compile()

thread = {"configurable": {"thread_id": "tx-1"}}
app.invoke({"draft": "", "approved": None}, config=thread)        # 停在 interrupt
app.invoke(None, config=thread, resume={"approve": True})         # 人工恢复
```

### 8.5 多 Agent 协作（Supervisor）

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from typing import TypedDict

class State(TypedDict):
    query: str
    answer: str

math_agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[], prompt="你是数学专家")
search_agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[], prompt="你是搜索专家")

def router(s: State) -> str:
    return "math" if "计算" in s["query"] else "search"

g = StateGraph(State)
g.add_node("math", math_agent)
g.add_node("search", search_agent)
g.add_edge(START, "router")
g.add_conditional_edges("router", router, {"math": "math", "search": "search"})
g.add_edge("math", END)
g.add_edge("search", END)
app = g.compile()
```

### 8.6 带 Checkpointer 的多轮记忆

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def chat(s: MessagesState):
    return {"messages": [ChatOpenAI().invoke(s["messages"])]}

g = StateGraph(MessagesState)
g.add_node("chat", chat)
g.add_edge(START, "chat")
g.add_edge("chat", END)
app = g.compile(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": "user-1"}}
app.invoke({"messages": [HumanMessage("我叫张三")]}, config=cfg)
app.invoke({"messages": [HumanMessage("我是谁？")]}, config=cfg)
print(app.get_state(cfg).values["messages"])  # 完整历史，模型"记得"
```

---

## 9. 选型对照表

| 场景 | 推荐方案 |
|---|---|
| 标准"模型↔工具"多轮循环，不想管图 | `create_agent`（LangChain 1.0，推荐）或 `create_react_agent`（LangGraph prebuilt，旧） |
| 自纠环 / 代码-测试-改码环 / 审批回退 / 多 Agent 协作 | `StateGraph` 手写 |
| 仅线性流水线（RAG、摘要→翻译），无需循环 | LCEL Chain |
| 需暂停/恢复、长流程记忆 | `StateGraph` + Checkpointer（`thread_id`） |

---

## 10. 一句话心智模型

> **LangGraph = 带 reducer 的全局 Store（State） + 由边驱动的执行调度器；节点是纯更新函数（接收 state、返回增量），边决定流程，reducer 负责合并。**

- State：全图唯一共享状态（TypedDict + reducer）。
- 节点：接收 state、返回增量 state 的可调用对象（函数 / Runnable / 子图）。
- 循环：通过回边（`add_edge("tools","agent")`）+ 条件路由实现，是图的一等公民。
- 记忆：`Checkpointer` + `thread_id` 持久化 state，支持 interrupt/resume。

---

*整理自系列问答：Chain 循环能力 → `create_agent` 归属 → LangGraph 构建函数 → State 模型 → 节点本质 → 共享状态确认 → 多示例实践。*
