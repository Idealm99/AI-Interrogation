
---

## 🧠 Fast LLaMA Fine-tuning with Unsloth

**LLaMA 기반 모델을 저비용·고속으로 파인튜닝**하기 위한 프로젝트입니다.  
[Unsloth 라이브러리](https://github.com/unslothai/unsloth)를 활용하여 Flash Attention, LoRA 기반 학습을 지원하며, Colab 환경에서 빠르게 실험할 수 있도록 구성되었습니다.

---

### 📌 프로젝트 개요

- **프로젝트명**: Fast Finetuning with LLaMA & Unsloth
- **모델**: LLaMA 2 7B / 13B (Hugging Face 기반)
- **기술 스택**: 
  - `Unsloth` (LoRA + Flash Attention 기반 가속 라이브러리)
  - `Transformers`, `PEFT`, `TRLL`, `Accelerate`, `BitsandBytes`
  - Google Colab 환경 기반
- **주요 기능**:
  - Colab에서 LLaMA 모델 파인튜닝 전체 파이프라인 제공
  - 사용자 정의 데이터셋 기반 학습 (JSON 형식)
  - 로컬/허깅페이스 저장 및 추론 가능

---
## 프로젝트 순서도 

![AI 심문하기 drawio](https://github.com/user-attachments/assets/1008d5af-e3b3-4348-a9d1-7f5a335b9b76)

### 🛠️ 주요 코드 구성

#### 1. 라이브러리 설치
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install xformers trl peft accelerate bitsandbytes
```

#### 2. 모델 로드 (Unsloth 방식)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-bf16-hf",
    max_seq_length = 2048,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)
```

#### 3. 데이터셋 포맷 예시 (JSON)
```json
{
  "conversations": [
    {"role": "user", "content": "고양이에 대해 알려줘"},
    {"role": "assistant", "content": "고양이는 포유류 동물로..." }
  ]
}
```

#### 4. 모델 학습
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    task_type = "CAUSAL_LM",
)

trainer = SFTTrainer(...)  # trl 라이브러리 기반
trainer.train()
```

---

### 🧪 성능 및 결과

| 항목 | 결과 |
|------|------|
| 파인튜닝 속도 | 기존 대비 약 2배 이상 빠름 |
| 메모리 사용 | 4bit quantization, Flash Attention으로 효율적 |
| 적용 범위 | QA, 대화형, 요약 등 다양한 텍스트 생성 task |

---

### 💾 모델 저장 및 사용

```python
model.save_pretrained("path/to/save")
tokenizer.save_pretrained("path/to/save")
```

---

### 🔁 회고 및 한계

> Unsloth 기반 파인튜닝은 **빠르고 저렴하게 LLaMA를 실험할 수 있는 강력한 방법**이었다.  
> 다만, 커스텀 데이터셋의 품질과 범용성 확보가 성능 향상에 핵심적임을 체감했으며, 앞으로는 다양한 task-specific 데이터셋을 조합하여 더 정교한 fine-tuning을 시도할 계획이다.

---

### 📎 참고 링크

- [Unsloth 공식 GitHub](https://github.com/unslothai/unsloth)
- [LLaMA 2 공식 모델 카드](https://huggingface.co/meta-llama)
- [파인튜닝 영상 참고 링크](https://www.youtube.com/watch?v=QaOIcJDDDjo)

---
