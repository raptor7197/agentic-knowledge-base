## Large Context Handling in Agentic Systems

Build a multi-turn agent with large-context handling including tool calls. Aim is to reduce context rotting

# Problem Statement : 
Agentic applications are increasingly used for reasoning over vast amounts of information. However, LLMs are limited by context window size, and once exceeded, they lose critical information — leading to truncated, inconsistent, or inaccurate outputs.

In complex multi-agent workflows involving tool calls, knowledge retrieval, and multi-turn reasoning, this limitation becomes a major barrier to reliability and scalability.

# Challenge : 
Your challenge is to design and build a system that enables
multi-turn agent workflows capable of retaining and reasoning over large contexts — even when total memory exceeds the model’s context window.

# Requirements
1. Support multi-turn agents that can maintain context across
long interactions.
2. Enable tool integrations that may produce large outputs.
3. Allow agents to share and reuse relevant context efficiently.
4. Minimize context loss when exceeding model limits.
5. Adapt to different model context sizes gracefully.
# Evaluation Criteria

| CATEGORY | DESCRIPTION |
|----------|-------------|
| Accuracy & Context Retention | How effectively the system preserves and reuses relevant context. |
| Approach to Large Context Handling | Originality and soundness of the proposed architecture or method. |
| Architecture & Code Quality | Design clarity, modularity, and maintainability. |
| Scalability | Ability to scale across longer sessions and larger data volumes. |
| Cost Efficiency | Optimization of token and compute usage. |
| Latency | Speed and responsiveness of the system under load. |
| Innovation & Usability | Practicality and creativity of the overall approach. |
