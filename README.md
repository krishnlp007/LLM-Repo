# 👨🏻‍💻 LLM-Repo
This repository contains a curated list of awesome 150+ libs category wise.
<p align="center">
  <a href="https://www.linkedin.com/in/kalyanksnlp/">
    <img src="https://custom-icon-badges.demolab.com/badge/Kalyan%20KS-0A66C2?logo=linkedin-white&logoColor=fff" alt="LinkedIn">
  </a>
  <a href="https://x.com/kalyan_kpl">
    <img src="https://img.shields.io/badge/Kalyan%20KS-%23000000.svg?logo=X&logoColor=white" alt="Twitter">
  </a>
   <a href="https://www.youtube.com/@kalyanksnlp">
    <img src="https://img.shields.io/badge/Kalyan%20KS-%23FF0000.svg?logo=YouTube&logoColor=white" alt="Twitter">
  </a>
	
</p>

## Quick links
||||
|---|---|---|
| [LLM Training](#llm-training-and-fine-tuning) | [LLM Application Development](#llm-application-development) | [LLM RAG](#llm-rag) | 
| [LLM Inference](#llm-inference)| [LLM Serving](#llm-serving) | [LLM Data Extraction](#llm-data-extraction) |
| [LLM Data Generation](#llm-data-generation) | [LLM Agents](#llm-agents)|[LLM Evaluation](#llm-evaluation) | 
| [LLM Monitoring](#llm-monitoring) | [LLM Prompts](#llm-prompts) | [LLM Structured Outputs](#llm-structured-outputs) |
| [LLM Safety and Security](#llm-safety-and-security) | [LLM Embedding Models](#llm-embedding-models) | [Others](#others) |

<p align ="center">
 	<a href="https://www.linkedin.com/newsletters/ai-buzz-with-kalyan-ks-7014810110401150976/">
		<img src="image_2025-03-09_181027647.png" alt="AI Buzz with Kalyan KS" width="1200"/>
	</a>
</p>



## LLM Training and Fine-Tuning
| Library             | Description                                                                                     | Link |
|---------------------|-------------------------------------------------------------------------------------------------|------|
| PEFT                | State-of-the-art Parameter-Efficient Fine-Tuning library.                                       | [Link](https://github.com/huggingface/peft) |
| TRL                 | Train transformer language models with reinforcement learning.                                  | [Link](https://github.com/huggingface/trl) |
| unsloth            | Fine-tune LLMs faster with less memory.                                                          | [Link](https://github.com/unslothai/unsloth) |
| Transformers       | Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio. | [Link](https://github.com/huggingface/transformers) |
| LLMBox             | A comprehensive library for implementing LLMs, including a unified training pipeline and comprehensive model evaluation. | [Link](https://github.com/RUCAIBox/LLMBox) |
| LitGPT             | Train and fine-tune LLM lightning fast.                                                          | [Link](https://github.com/Lightning-AI/litgpt) |
| Mergoo            | A library for easily merging multiple LLM experts, and efficiently train the merged LLM.         | [Link](https://github.com/Leeroo-AI/mergoo) |
| Llama-Factory      | Easy and efficient LLM fine-tuning.                                                              | [Link](https://github.com/hiyouga/LLaMA-Factory) |
| Ludwig            | Low-code framework for building custom LLMs, neural networks, and other AI models.               | [Link](https://github.com/ludwig-ai/ludwig) |
| Txtinstruct       | A framework for training instruction-tuned models.                                               | [Link](https://github.com/neuml/txtinstruct) |
| Lamini            | An integrated LLM inference and tuning platform.                                                 | [Link](https://github.com/lamini-ai/lamini) |
| XTuring           | xTuring provides fast, efficient and simple fine-tuning of open-source LLMs, such as Mistral, LLaMA, GPT-J, and more. | [Link](https://github.com/stochasticai/xTuring) |
| RL4LMs            | A modular RL library to fine-tune language models to human preferences.                          | [Link](https://github.com/allenai/RL4LMs) |
| DeepSpeed         | DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective. | [Link](https://github.com/deepspeedai/DeepSpeed) |
| torchtune         | A PyTorch-native library specifically designed for fine-tuning LLMs.                             | [Link](https://github.com/pytorch/torchtune) |
| PyTorch Lightning | A library that offers a high-level interface for pretraining and fine-tuning LLMs.               | [Link](https://github.com/Lightning-AI/pytorch-lightning) |
| Axolotl           | Tool designed to streamline post-training for various AI models.                                 | [Link](https://github.com/axolotl-ai-cloud/axolotl/) |

## LLM Application Development
<p align = "center"> <b> Frameworks </b> </p>

| Library        | Description                                                                                               | Link  |
|--------------|------------------------------------------------------------------------------------------------------|-------|
| LangChain    | LangChain is a framework for developing applications powered by large language models (LLMs).          | [Link](https://github.com/langchain-ai/langchain) |
| Llama Index  | LlamaIndex is a data framework for your LLM applications.                                              | [Link](https://github.com/run-llama/llama_index) |
| HayStack     | Haystack is an end-to-end LLM framework that allows you to build applications powered by LLMs, Transformer models, vector search and more. | [Link](https://github.com/deepset-ai/haystack) |
| Prompt flow  | A suite of development tools designed to streamline the end-to-end development cycle of LLM-based AI applications. | [Link](https://github.com/microsoft/promptflow) |
| Griptape     | A modular Python framework for building AI-powered applications.                                        | [Link](https://github.com/griptape-ai/griptape) |
| Weave        | Weave is a toolkit for developing Generative AI applications.                                          | [Link](https://github.com/wandb/weave) |
| Llama Stack  | Build Llama Apps.                                                                                      | [Link](https://github.com/meta-llama/llama-stack) |

<p align = "center"> <b> Multi API Access </b> </p>

| Library        | Description                                                                                               | Link  |
|--------------|------------------------------------------------------------------------------------------------------|-------|
| LiteLLM      | Library to call 100+ LLM APIs in OpenAI format.                                                        | [Link](https://github.com/BerriAI/litellm) |
| AI Gateway   | A Blazing Fast AI Gateway with integrated Guardrails. Route to 200+ LLMs, 50+ AI Guardrails with 1 fast & friendly API.                                                 | [Link](https://github.com/Portkey-AI/gateway) |

<p align = "center"> <b> Routers </b> </p>

| Library        | Description                                                                                               | Link  |
|--------------|------------------------------------------------------------------------------------------------------|-------|
| RouteLLM     | Framework for serving and evaluating LLM routers.                                                      | [Link](https://github.com/lm-sys/RouteLLM) |


<p align = "center"> <b> Memory </b> </p>

| Library        | Description                                                                                               | Link  |
|--------------|------------------------------------------------------------------------------------------------------|-------|
| mem0         | The Memory layer for your AI apps.                                                                     | [Link](https://github.com/mem0ai/mem0) |
| Memoripy     | An AI memory layer with short- and long-term storage, semantic clustering, and optional memory decay for context-aware applications. | [Link](https://github.com/caspianmoon/memoripy) |

<p align = "center"> <b> Interface </b> </p>

| Library        | Description                                                                                               | Link  |
|--------------|------------------------------------------------------------------------------------------------------|-------|
| Simpleaichat | Python package for easily interfacing with chat apps, with robust features and minimal code complexity. | [Link](https://github.com/minimaxir/simpleaichat) |
| Chainlit     | Build production-ready Conversational AI applications in minutes.                                      | [Link](https://github.com/Chainlit/chainlit) |
| Streamlit    | A faster way to build and share data apps.                                                             | [Link](https://github.com/streamlit/streamlit) |
| Gradio       | Build and share delightful machine learning apps, all in Python.                                       | [Link](https://github.com/gradio-app/gradio) |
| AI SDK UI    | Build chat and generative user interfaces.                                                             | [Link](https://sdk.vercel.ai/docs/introduction) |
| AI-Gradio    | Create AI apps powered by various AI providers.                                                        | [Link](https://github.com/AK391/ai-gradio) |


<p align = "center"> <b> Low Code </b> </p>

| Library        | Description                                                                                               | Link  |
|--------------|------------------------------------------------------------------------------------------------------|-------|
| LangFlow     | LangFlow is a low-code app builder for RAG and multi-agent AI applications.                            | [Link](https://github.com/langflow-ai/langflow) |

<p align = "center"> <b> Cache </b> </p>

| Library        | Description                                                                                               | Link  |
|--------------|------------------------------------------------------------------------------------------------------|-------|
| GPTCache     | Semantic cache for LLMs. Fully integrated with LangChain and LlamaIndex.                               | [Link](https://github.com/zilliztech/gptcache) |


## LLM RAG

| Library         | Description                                                                                                      | Link  |
|---------------|----------------------------------------------------------------------------------------------------------------|-------|
| FastGraph RAG | Streamlined and promptable Fast GraphRAG framework designed for interpretable, high-precision, agent-driven retrieval workflows. | [Link](https://github.com/circlemind-ai/fast-graphrag) |
| Chonkie       | RAG chunking library that is lightweight, lightning-fast, and easy to use.                                      | [Link](https://github.com/chonkie-ai/chonkie) |
| RAGChecker    | A Fine-grained Framework For Diagnosing RAG.                                                                   | [Link](https://github.com/amazon-science/RAGChecker) |
| RAG to Riches | Build, scale, and deploy state-of-the-art Retrieval-Augmented Generation applications.                         | [Link](https://github.com/SciPhi-AI/R2R) |
| BeyondLLM     | Beyond LLM offers an all-in-one toolkit for experimentation, evaluation, and deployment of Retrieval-Augmented Generation (RAG) systems. | [Link](https://github.com/aiplanethub/beyondllm) |
| SQLite-Vec    | A vector search SQLite extension that runs anywhere!                                                           | [Link](https://github.com/asg017/sqlite-vec) |
| fastRAG       | fastRAG is a research framework for efficient and optimized retrieval-augmented generative pipelines, incorporating state-of-the-art LLMs and Information Retrieval. | [Link](https://github.com/IntelLabs/fastRAG) |
| FlashRAG      | A Python Toolkit for Efficient RAG Research.                                                                   | [Link](https://github.com/RUC-NLPIR/FlashRAG) |
| Llmware       | Unified framework for building enterprise RAG pipelines with small, specialized models.                        | [Link](https://github.com/llmware-ai/llmware) |
| Rerankers     | A lightweight unified API for various reranking models.                                                        | [Link](https://github.com/AnswerDotAI/rerankers) |
| Vectara       | Build Agentic RAG applications.                                                                                | [Link](https://vectara.github.io/py-vectara-agentic/latest/) |


## LLM Inference

| Library         | Description                                                                                               | Link  |
|---------------|------------------------------------------------------------------------------------------------------|-------|
| LLM Compressor | Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment. | [Link](https://github.com/vllm-project/llm-compressor) |
| LightLLM      | Python-based LLM inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance. | [Link](https://github.com/ModelTC/lightllm) |
| vLLM         | High-throughput and memory-efficient inference and serving engine for LLMs.                            | [Link](https://github.com/vllm-project/vllm) |
| torchchat     | Run PyTorch LLMs locally on servers, desktop, and mobile.                                              | [Link](https://github.com/pytorch/torchchat) |
| TensorRT-LLM  | TensorRT-LLM is a library for optimizing Large Language Model (LLM) inference.                        | [Link](https://github.com/NVIDIA/TensorRT-LLM) |
| WebLLM        | High-performance In-browser LLM Inference Engine.                                                     | [Link](https://github.com/mlc-ai/web-llm) |


## LLM Serving

| Library   | Description                                                              | Link  |
|-----------|--------------------------------------------------------------------------|-------|
| Langcorn  | Serving LangChain LLM apps and agents automagically with FastAPI.       | [Link](https://github.com/msoedov/langcorn) |
| LitServe  | Lightning-fast serving engine for any AI model of any size.             | [Link](https://github.com/Lightning-AI/LitServe) |


## LLM Data Extraction

| Library         | Description                                                                                                                           | Link  |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------|-------|
| Crawl4AI       | Open-source LLM Friendly Web Crawler & Scraper.                                                                                      | [Link](https://github.com/unclecode/crawl4ai) |
| ScrapeGraphAI  | A web scraping Python library that uses LLM and direct graph logic to create scraping pipelines for websites and local documents (XML, HTML, JSON, Markdown, etc.). | [Link](https://github.com/ScrapeGraphAI/Scrapegraph-ai) |
| Docling        | Docling parses documents and exports them to the desired format with ease and speed.                                                  | [Link](https://github.com/DS4SD/docling) |
| Llama Parse    | GenAI-native document parser that can parse complex document data for any downstream LLM use case (RAG, agents).                     | [Link](https://github.com/run-llama/llama_cloud_services) |
| PyMuPDF4LLM    | PyMuPDF4LLM library makes it easier to extract PDF content in the format you need for LLM & RAG environments.                        | [Link](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) |
| Crawlee        | A web scraping and browser automation library.                                                                                         | [Link](https://github.com/apify/crawlee-python) |
| MegaParse      | Parser for every type of document.                                                                                                    | [Link](https://github.com/quivrhq/megaparse) |
| ExtractThinker | Document Intelligence library for LLMs.                                                                                               | [Link](https://github.com/enoch3712/ExtractThinker) |


## LLM Data Generation

| Library       | Description                                                                                          | Link  |
|--------------|--------------------------------------------------------------------------------------------------|-------|
| DataDreamer  | DataDreamer is a powerful open-source Python library for prompting, synthetic data generation, and training workflows. | [Link](https://github.com/datadreamer-dev/DataDreamer) |
| fabricator   | A flexible open-source framework to generate datasets with large language models.                   | [Link](https://github.com/flairNLP/fabricator) |
| Promptwright | Synthetic Dataset Generation Library.                                                               | [Link](https://github.com/stacklok/promptwright) |
| EasyInstruct | An Easy-to-use Instruction Processing Framework for Large Language Models.                          | [Link](https://github.com/zjunlp/EasyInstruct) |


## LLM Agents

| Library         | Description                                                                                                 | Link  |
|----------------|---------------------------------------------------------------------------------------------------------|-------|
| OpenWebAgent   | An Open Toolkit to Enable Web Agents on Large Language Models.                                           | [Link](https://github.com/THUDM/OpenWebAgent/) |
| Agno          | Build AI Agents with memory, knowledge, tools, and reasoning. Chat with them using a beautiful Agent UI.  | [Link](https://github.com/agno-agi/agno) |
| Lagent        | A lightweight framework for building LLM-based agents.                                                   | [Link](https://github.com/InternLM/lagent) |
| LazyLLM       | A Low-code Development Tool For Building Multi-agent LLMs Applications.                                  | [Link](https://github.com/LazyAGI/LazyLLM) |
| Composio      | Production Ready Toolset for AI Agents.                                                                  | [Link](https://github.com/ComposioHQ/composio) |
| Swarms        | The Enterprise-Grade Production-Ready Multi-Agent Orchestration Framework.                               | [Link](https://github.com/kyegomez/swarms) |
| AutoGen       | An open-source framework for building AI agent systems.                                                  | [Link](https://github.com/microsoft/autogen) |
| gradio-tools  | A Python library for converting Gradio apps into tools that can be leveraged by an LLM-based agent to complete its task. | [Link](https://github.com/freddyaboulton/gradio-tools) |
| ChatArena     | ChatArena is a library that provides multi-agent language game environments and facilitates research about autonomous LLM agents and their social interactions. | [Link](https://github.com/Farama-Foundation/chatarena) |
| CrewAI        | Framework for orchestrating role-playing, autonomous AI agents.                                          | [Link](https://github.com/crewAIInc/crewAI) |
| Swarm         | Educational framework exploring ergonomic, lightweight multi-agent orchestration.                        | [Link](https://github.com/openai/swarm) |
| AgentStack    | The fastest way to build robust AI agents.                                                               | [Link](https://github.com/AgentOps-AI/AgentStack) |
| LangGraph     | Build resilient language agents as graphs.                                                               | [Link](https://github.com/langchain-ai/langgraph) |
| Archgw        | Intelligent gateway for Agents.                                                                          | [Link](https://github.com/katanemo/archgw) |
| Flow          | A lightweight task engine for building AI agents.                                                        | [Link](https://github.com/lmnr-ai/flow) |
| AgentOps      | Python SDK for AI agent monitoring.                                                                      | [Link](https://github.com/AgentOps-AI/agentops) |
| Langroid      | Multi-Agent framework.                                                                                   | [Link](https://github.com/langroid/langroid) |
| Smolagents    | Library to build powerful agents in a few lines of code.                                                 | [Link](https://github.com/huggingface/smolagents) |
| Memary        | Open Source Memory Layer For Autonomous Agents.                                                          | [Link](https://github.com/kingjulio8238/Memary) |
| Browser Use   | Make websites accessible for AI agents.                                                                 | [Link](https://github.com/browser-use/browser-use) |
| Agentarium    | Framework for creating and managing simulations populated with AI-powered agents.                        | [Link](https://github.com/Thytu/Agentarium) |
| Atomic Agents | Building AI agents, atomically.                                                                         | [Link](https://github.com/BrainBlend-AI/atomic-agents) |


## LLM Evaluation



## LLM Monitoring

## LLM Prompts

## LLM Structured Outputs

## LLM Safety and Security

## LLM Embedding Models

## Others






		



## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=krishnlp007/LLM-Repo&type=Date)](https://star-history.com/#)

Please consider giving a star, if you find this repository useful. 
