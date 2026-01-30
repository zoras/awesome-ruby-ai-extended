# Awesome Ruby AI 🤖💎

A curated and **expanded** list of Ruby libraries, frameworks, tools, and learning resources for **Artificial Intelligence (AI)**, **Machine Learning (ML)**, **LLMs**, **Agents**, and **RAG systems**.

This list consolidates:

* Projects mentioned in *RoboRuby – Ruby AI News*
* Items from *alexrudall/awesome-ruby-ai*
* Additional high‑quality Ruby AI/ML projects discovered across the ecosystem

---

## Contents

* [LLM & Model Interfaces](#llm--model-interfaces)
* [Agent Frameworks & Orchestration](#agent-frameworks--orchestration)
* [RAG, Embeddings & Vector Search](#rag-embeddings--vector-search)
* [Machine Learning Libraries](#machine-learning-libraries)
* [Rails & Application Integrations](#rails--application-integrations)
* [Developer Tools & Evaluation](#developer-tools--evaluation)
* [Infrastructure, Protocols & Automation](#infrastructure-protocols--automation)
* [Tutorials & Learning Resources](#tutorials--learning-resources)

---

## LLM & Model Interfaces

* **RubyLLM** – High-level LLM abstraction with tools, streaming, structured output, and multi-provider support.
  [https://github.com/ruby-ai/ruby_llm](https://github.com/ruby-ai/ruby_llm)

* **ruby-openai** – OpenAI API client for Ruby.
  [https://github.com/alexrudall/ruby-openai](https://github.com/alexrudall/ruby-openai)

* **Anthropic Ruby SDK** – Official SDK for Claude models.
  [https://github.com/anthropics/anthropic-sdk-ruby](https://github.com/anthropics/anthropic-sdk-ruby)

* **durable_llm** – Provider-agnostic LLM client with fallback strategies.
  [https://github.com/alexrudall/durable_llm](https://github.com/alexrudall/durable_llm)

* **rllama** – Run llama.cpp locally from Ruby.
  [https://github.com/ankane/rllama](https://github.com/ankane/rllama)

* **ollama-ruby** – Ruby client for Ollama local models.
  [https://github.com/ollama/ollama-ruby](https://github.com/ollama/ollama-ruby)

* **cohere-ruby** – Ruby SDK for Cohere models.
  [https://github.com/cohere-ai/cohere-ruby](https://github.com/cohere-ai/cohere-ruby)

---

## Agent Frameworks & Orchestration

* **RubyLLM Agents** – Multi-agent orchestration built on RubyLLM.
  [https://github.com/ruby-ai/agents](https://github.com/ruby-ai/agents)

* **agentic** – Declarative plan-and-execute agent framework.
  [https://github.com/joaomdmoura/agentic](https://github.com/joaomdmoura/agentic)

* **LangGraph.rb** – Ruby bindings inspired by LangGraph agent workflows.
  [https://github.com/andreibondarev/langgraph-rb](https://github.com/andreibondarev/langgraph-rb)

* **llama_bot_rails** – Rails engine for agentic workflows using LangGraph.
  [https://github.com/manuelmeurer/llama_bot_rails](https://github.com/manuelmeurer/llama_bot_rails)

---

## RAG, Embeddings & Vector Search

* **rag_embeddings** – Simple embeddings + vector search utilities.
  [https://github.com/ankane/rag_embeddings](https://github.com/ankane/rag_embeddings)

* **ragdoll** – Opinionated Ruby RAG framework.
  [https://github.com/ankane/ragdoll](https://github.com/ankane/ragdoll)

* **neighbor** – Vector similarity search for ActiveRecord.
  [https://github.com/ankane/neighbor](https://github.com/ankane/neighbor)

* **pgvector** – PostgreSQL vector extension (Ruby-friendly).
  [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)

* **vector_mcp** – MCP tools for vector databases.
  [https://github.com/joshua-maros/vector_mcp](https://github.com/joshua-maros/vector_mcp)

---

## Machine Learning Libraries

* **Rumale** – Scikit-Learn–style ML library for Ruby.
  [https://github.com/yoshoku/rumale](https://github.com/yoshoku/rumale)

* **TensorFlow.rb** – Ruby bindings for TensorFlow.
  [https://github.com/somaticio/tensorflow.rb](https://github.com/somaticio/tensorflow.rb)

* **torch.rb** – PyTorch-style tensors and autograd for Ruby.
  [https://github.com/ankane/torch.rb](https://github.com/ankane/torch.rb)

* **numo-narray** – Numerical arrays for scientific computing.
  [https://github.com/ruby-numo/numo-narray](https://github.com/ruby-numo/numo-narray)

* **clusterkit** – Clustering and dimensionality reduction.
  [https://github.com/ankane/clusterkit](https://github.com/ankane/clusterkit)

---

## Rails & Application Integrations

* **Blazer AI** – Natural language → SQL interface for Blazer.
  [https://github.com/ankane/blazer-ai](https://github.com/ankane/blazer-ai)

* **rails_ai** – Rails generators and helpers for AI features.
  [https://github.com/ruby-ai/rails_ai](https://github.com/ruby-ai/rails_ai)

* **chatwoot-ai** – AI assistants integrated into Chatwoot (Rails).
  [https://github.com/chatwoot/chatwoot](https://github.com/chatwoot/chatwoot)

---

## Developer Tools & Evaluation

* **LLMTape** – Record/replay LLM calls in tests.
  [https://github.com/ankane/llm_tape](https://github.com/ankane/llm_tape)

* **Leva** – LLM evaluation and benchmarking.
  [https://github.com/ankane/leva](https://github.com/ankane/leva)

* **Internator** – AI-powered iterative code modification CLI.
  [https://github.com/aaronmallen/internator](https://github.com/aaronmallen/internator)

* **Botrytis** – Fuzzy AI matching for Cucumber steps.
  [https://github.com/westonganger/botrytis](https://github.com/westonganger/botrytis)

* **RSpec AI Agents** – Auto-generate RSpec tests with AI.
  [https://github.com/ruby-ai/rspec-ai-agents](https://github.com/ruby-ai/rspec-ai-agents)

---

## Infrastructure, Protocols & Automation

* **fast-mcp** – Ruby implementation of Model Context Protocol.
  [https://github.com/joshua-maros/fast-mcp](https://github.com/joshua-maros/fast-mcp)

* **ruby_llm-mcp** – MCP client/server built on RubyLLM.
  [https://github.com/ruby-ai/ruby_llm-mcp](https://github.com/ruby-ai/ruby_llm-mcp)

* **rails_mcp_server** – MCP server for Rails apps.
  [https://github.com/joshua-maros/rails_mcp_server](https://github.com/joshua-maros/rails_mcp_server)

* **FerrumMCP** – Browser automation via MCP.
  [https://github.com/joshua-maros/ferrum_mcp](https://github.com/joshua-maros/ferrum_mcp)

* **simplecov-mcp** – Expose coverage data to AI agents.
  [https://github.com/joshua-maros/simplecov-mcp](https://github.com/joshua-maros/simplecov-mcp)

---

## Tutorials & Learning Resources

* **Build RAG in Rails** – End-to-end RAG tutorial.
  [https://ankane.org/rag-rails](https://ankane.org/rag-rails)

* **Ruby AI Guides** – Practical Ruby + AI articles.
  [https://rubyai.dev](https://rubyai.dev)

* **Chatbot That Builds Rails Apps** – AI-driven Rails scaffolding.
  [https://github.com/raoulbond/rails-ai-generator](https://github.com/raoulbond/rails-ai-generator)

---

## Contributing

PRs welcome! Please include:

* Short description
* Canonical GitHub repo or homepage
* Appropriate category

---

## License

MIT
