# 👨🏻‍💻 LLM-Repo
This repository contains a curated list of awesome 150+ libs category wise.

<img src="image_2025-03-09_181027647.png" alt="AI Buzz with Kalyan KS" width="300"/>

<p align="center">
  <a href="https://www.linkedin.com/in/kalyanksnlp/">
    <img src="https://custom-icon-badges.demolab.com/badge/Kalyan%20KS-0A66C2?logo=linkedin-white&logoColor=fff" alt="LinkedIn">
  </a>
  <a href="https://x.com/kalyan_kpl">
    <img src="https://img.shields.io/badge/Kalyan%20KS-%23000000.svg?logo=X&logoColor=white" alt="Twitter">
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

## LLM Inference

## LLM Serving

## LLM Data Extraction

## LLM Data Generation

## LLM Agents

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
