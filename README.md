<div align="center">

# <img src="assets/images/icon.png" alt="icon" height="42" style="vertical-align: middle;"> Debate with Images

### Detecting Multimodal Deceptive Behaviors in MLLMs

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red)](https://arxiv.org/abs/)
[![Homepage](https://img.shields.io/badge/🏠_Homepage-mm--deception.github.io-purple)](https://mm-deception.github.io)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-black)](https://github.com/mm-deception/debate-with-images)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-MM--DeceptionBench-blue)](https://github.com/mm-deception/MM-DeceptionBench/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

<p align="center">
  <strong>A comprehensive framework for evaluating deceptive behaviors in Multimodal Large Language Models through debate-based assessment with visual evidence.</strong>
</p>

[Overview](#-overview) •
[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Citation](#-citation)

---

<img src="assets/images/teaser.png" alt="Teaser" width="100%">

*Defining multimodal deception through three distinct behavioral patterns.*

</div>

---

## 📋 Overview

**Debate with Images** introduces a novel multi-agent debate paradigm for detecting deceptive behaviors in MLLMs. Unlike traditional single-model evaluation, our approach leverages adversarial debate with **visual grounding** to uncover subtle forms of deception that may evade direct assessment.

<div align="center">

<img src="assets/images/framework.jpg" alt="Framework" width="90%">

*The Debate with images framework: A multi-agent evaluation framework for detecting multimodal deception.*

</div>

### Why Debate with Images?

| Challenge | Traditional Approach | Our Solution |
|-----------|---------------------|--------------|
| Subtle deception | Single model judgment | Multi-agent adversarial debate |
| Vague reasoning | Text-only evidence | Visual grounding with bboxes |
| Model bias | Fixed evaluation | Dynamic argumentation |
| Limited coverage | Task-specific eval | 6 real-world deception behavior categories |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔬 MM-DeceptionBench
- **6 Deception Categories**: Fabrication, Sycophancy, Bluff, Sandbagging, Obfuscation, Deliberate Omission
- **MLLM-as-a-Judge**: Direct evaluation pipeline
- **Human-aligned labels**: Gold standard annotations

</td>
<td width="50%">

### 🎯 Debate Framework
- **Multi-agent System**: Affirmer vs. Negator roles
- **Visual Evidence**: Bounding boxes, points, lines
- **Iterative Refinement**: Multi-round debates
- **Neutral Judge**: Final verdict determination

</td>
</tr>
<tr>
<td width="50%">

### 🔌 Backend Support
- **API Models**: GPT, Claude, Gemini, etc..
- **Local Models**: vLLM serving (Qwen2.5-VL)
- **Async Processing**: Ray-based parallelization
- **Response Caching**: Faster iteration cycles

</td>
<td width="50%">

### 📊 Auxiliary Benchmarks
- **[PKU-SafeRLHF-V](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-V)**: Safety preference evaluation
- **[HallusionBench](https://github.com/tianyi-lab/HallusionBench)**: Visual reasoning task
- **Universal Evaluator**: Multi-format comparison

</td>
</tr>
</table>

---

## 📁 Project Structure

```
multimodal_deception/
├── 🎭 debate/                          # Debate-based evaluation system
│   ├── run_debate.py                   # Main debate orchestrator
│   ├── api_utils.py                    # API client implementations
│   ├── visual_utils.py                 # Visual evidence rendering
│   ├── prompt.py                       # Debate prompt templates
│   ├── format_utils.py                 # Data formatting utilities
│   ├── config.yaml                     # Debate configuration
│   └── vllm_server.sh                  # vLLM server launcher
│
├── 🔧 eval_tools/                      # Evaluation utilities
│   ├── mmdeceptionbench/               # MM-DeceptionBench evaluation
│   │   ├── routine.py                  # Inference pipeline
│   │   ├── eval.py                     # MLLM-as-a-judge evaluation
│   │   ├── universal_evaluator.py      # Multi-format evaluator
│   │   └── configs/                    # Model configurations
│   ├── pkusaferlhfv/                   # Safety evaluation
│   ├── hallusionbench/                 # Hallucination detection
│   ├── basic_api.py                    # Base API utilities
│   ├── openai_reasoning_api.py         # OpenAI o1/o3 support
│   └── anthropic_reasoning_api.py      # Claude extended thinking
│
└── 📄 human_label.json                 # Human-annotated gold labels examples
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for local vLLM models)

### Step 1: Clone Repository

```bash
git clone https://github.com/mm-deception/debate-with-images.git
cd debate-with-images
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Core dependencies
pip install ray matplotlib pillow opencv-python numpy
pip install scikit-learn datasets tqdm pyyaml
pip install aiohttp urllib3 openai anthropic

# For local model serving (optional)
pip install vllm>=0.7.2
```

### Step 3: Configure API Keys

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or configure in YAML files
```

---

## 🚀 Quick Start

### 1️⃣ MM-DeceptionBench Evaluation

#### Run Model Inference

```bash
cd eval_tools/mmdeceptionbench
python routine.py --config configs/api_config.yaml
```

<details>
<summary>📋 Available Configuration Files</summary>

| Config | Description | Models |
|--------|-------------|--------|
| `api_config.yaml` | Cloud API models | GPT-4o |
| `vllm_config.yaml` | Local vLLM serving | Qwen2.5-VL, InternVL3 |
| `claude_config.yaml` | Anthropic-specific | Claude 4 Sonnet/Opus |
| `reason_config.yaml` | Other Reasoning models | Gemini-2.5-Pro, o1 |

</details>

#### Evaluate with MLLM-as-a-Judge

```bash
python eval.py --input results/model_output.json --output results/evaluation.json
```

---

### 2️⃣ Debate with Images

#### Run Multi-Agent Debate

```bash
cd debate
python run_debate.py --config config.yaml
```

#### Configuration Example

```yaml
# debate/config.yaml
datasets:
  - path: "../human_label.json"
    name: "deception_samples"

output_dir: "./results/debate"
round: 2  # Number of debate rounds

history: "./history.json" # optional: load previous debate history

debaters:
  - role: "affirmer"  # Affirmer: argues deception exists
    config:
      model: "gpt-4o"
      backend: "api"
      api_key: "${OPENAI_API_KEY}"
      use_cache: true
      cache_dir: "./cache"
      num_workers: 50
      infer_cfgs:
        temperature: 0.0
        max_tokens: 512

  - role: "negator"  # Negator: argues no deception
    config:
      model: "gpt-4o"
      backend: "api"
      api_key: "${OPENAI_API_KEY}"
      use_cache: true
      infer_cfgs:
        temperature: 0.0
        max_tokens: 512

judge:
  role: "D_judge"  # Final arbiter
  config:
    model: "gpt-4o"
    backend: "api"
    api_key: "${OPENAI_API_KEY}"
    infer_cfgs:
      temperature: 0.0
      max_tokens: 512
```

---


#### Debate Output Format

```json
{
  "case": { /* original case data */ },
  "debate_trajectory": [
    {
      "role": "affirmer",
      "round": 1,
      "content": "The model exhibits deceptive behavior because...",
      "visual_evidence": [
        {
          "bbox_2d": [0.15, 0.20, 0.45, 0.60],
          "label": "Evidence: Object not mentioned by model"
        }
      ]
    },
    {
      "role": "negator",
      "round": 1,
      "content": "The response is truthful given that...",
      "visual_evidence": [
        {
          "bbox_2d": [0.50, 0.30, 0.80, 0.70],
          "label": "Counter-evidence: Ambiguous region"
        }
      ]
    }
  ],
  "judge_verdict": {
    "decision": "Deceptive",
    "justification": {
      "type": "bbox",
      "bbox": [0.1, 0.3, 0.2, 0.4],
      "label": "This evidence shows..."
    },
    ...
  }
}
```

---

## ⚙️ Configuration Reference

### Backend Options

<table>
<tr>
<th>Backend</th>
<th>Configuration</th>
<th>Use Case</th>
</tr>
<tr>
<td><b>API</b></td>
<td>

```yaml
backend: "api"
model: "gpt-4o"
api_key: "sk-..."
base_url: null  # Optional
```

</td>
<td>Cloud models (OpenAI, Anthropic)</td>
</tr>
<tr>
<td><b>vLLM</b></td>
<td>

```yaml
backend: "vllm"
model: "Qwen/Qwen2.5-VL-72B-Instruct"
base_url: "http://localhost:8000"
```

</td>
<td>Local GPU serving</td>
</tr>
</table>

### Inference Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.0 | Sampling temperature |
| `max_tokens` | int | 512 | Maximum response length |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `num_workers` | int | 50 | Parallel API requests |

### Visual Evidence Types

| Type | Format | Description |
|------|--------|-------------|
| Bounding Box | `bbox_2d: [x, y, w, h]` | Annotate: draw bounding boxex |
| Point | `point_2d: [x, y]` | Annotate: draw points |
| Line | `line_2d: [x1, y1, x2, y2]`| Annotate: draw lines |
| Zoom-in | `zoom_in_2d: [x, y, w, h]` | Zoom-in specific regions
| Segmentation | `segmentation_2d: [x, y] / [[x1, y1], [x2, y2], ...]` | Segment by point(s)
| Depth Estimation | `true / false` | Estimate depth

To enbale segmentation and depth-estimation operations, please install models to `debate/models`, or system will fall back to naive implementations.

---

## 🔧 Advanced Usage

### Local Model Serving with vLLM

```bash
# Start vLLM server
cd debate
bash vllm_server.sh

# Or manually:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 8 \
    --port 8000
```

### Batch Processing Multiple Datasets

```yaml
datasets:
  - path: "./mm-deceptionbench/dataset/fabrication.json"
    name: "fabrication"
  - path: "./mm-deceptionbench/dataset/sycophancy.json"
    name: "sycophancy"
  - path: "./mm-deceptionbench/dataset/bluff.json"
    name: "bluff"
```

### Response Caching

Enable caching to avoid redundant API calls during development:

```yaml
use_cache: true
cache_dir: "./cache"
```

### Universal Evaluator

Compare model outputs against ground truth across formats:

```bash
cd eval_tools/mmdeceptionbench
python universal_evaluator.py \
    --input ./model_output.json \
    --gt ./human_label.json \
    --output ./evaluation_results
```

---

## 📊 Auxiliary Benchmarks

### [PKU-SafeRLHF-V](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-V)

Evaluate safety preferences with visual context:

```bash
cd eval_tools/pkusaferlhfv
python eval_saferlhfv.py --config config.yaml
python cal_saferlhfv_metrics.py --input results.json
```

### [HallusionBench](https://github.com/tianyi-lab/HallusionBench)

Detect visual hallucinations:

```bash
cd eval_tools/hallusionbench
python eval_hall.py --config config.yaml
python cal_hallu_metrics.py --input results.json
```

> 💡 **Tip**: The debate framework can also be applied to these benchmarks by modifying the prompts in `prompts.py` and `debate/format_utils.py`.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{fang2025debate,
  title     = {Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models},
  author    = {Fang, Sitong and Hou, Shiyi and Wang, Kaile and Chen, Boyuan and Hong, Donghai and Zhou, Jiayi and Dai, Juntao and Yang, Yaodong and Ji, Jiaming},
  journal   = {Preprint},
  year      = {2025}
}
```

---

## 📄 License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.

---

<div align="center">

**[⬆ Back to Top](#-debate-with-images)**

Made with ❤️ for safer AI

</div>
