
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
[Uploadin<mxfile host="app.diagrams.net" agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" version="26.0.6">
  <diagram name="페이지-1" id="6L7_7K4dRQS6INN4i_TD">
    <mxGraphModel grid="1" page="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="8r8z8gKgrVPisf2p9DPh-7" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="8r8z8gKgrVPisf2p9DPh-1" target="8r8z8gKgrVPisf2p9DPh-4">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-1" value="시작" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;" vertex="1" parent="1">
          <mxGeometry x="320" y="70" width="80" height="80" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-9" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="8r8z8gKgrVPisf2p9DPh-2" target="8r8z8gKgrVPisf2p9DPh-3">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-2" value="질문 입력" style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="300" y="330" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-10" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="8r8z8gKgrVPisf2p9DPh-3" target="8r8z8gKgrVPisf2p9DPh-5">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-3" value="AI&amp;nbsp;&lt;div&gt;(질문 대답)&lt;/div&gt;" style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="300" y="440" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-8" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="8r8z8gKgrVPisf2p9DPh-4" target="8r8z8gKgrVPisf2p9DPh-2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-4" value="상황 부여" style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="300" y="230" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-11" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="8r8z8gKgrVPisf2p9DPh-5" target="8r8z8gKgrVPisf2p9DPh-2">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="200" y="595" />
              <mxPoint x="200" y="360" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-12" value="Yes" style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];" connectable="0" vertex="1" parent="8r8z8gKgrVPisf2p9DPh-11">
          <mxGeometry x="-0.7778" y="3" relative="1" as="geometry">
            <mxPoint as="offset" />
          </mxGeometry>
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-15" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="8r8z8gKgrVPisf2p9DPh-5">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="360" y="730" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-17" value="No" style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];" connectable="0" vertex="1" parent="8r8z8gKgrVPisf2p9DPh-15">
          <mxGeometry x="-0.44" y="2" relative="1" as="geometry">
            <mxPoint as="offset" />
          </mxGeometry>
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-5" value="질문 횟수&amp;lt;5" style="rhombus;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="315" y="550" width="90" height="90" as="geometry" />
        </mxCell>
        <mxCell id="8r8z8gKgrVPisf2p9DPh-6" value="끝" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;" vertex="1" parent="1">
          <mxGeometry x="320" y="1040" width="80" height="80" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-6" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="rWI0O_FinwYjYTAR-48q-1" target="rWI0O_FinwYjYTAR-48q-5">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-1" value="Lama AI" style="whiteSpace=wrap;html=1;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="490" y="290" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-4" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="rWI0O_FinwYjYTAR-48q-3" target="rWI0O_FinwYjYTAR-48q-1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-3" value="데이터셋" style="shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;" vertex="1" parent="1">
          <mxGeometry x="490" y="130" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-10" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.992;entryY=0.4;entryDx=0;entryDy=0;entryPerimeter=0;" edge="1" parent="1" source="rWI0O_FinwYjYTAR-48q-5" target="8r8z8gKgrVPisf2p9DPh-3">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="570" y="465" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-5" value="Fine_tuning&amp;nbsp;" style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="490" y="440" width="120" height="50" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-15" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;" edge="1" parent="1" source="rWI0O_FinwYjYTAR-48q-11">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="360" y="1040" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-11" value="결과 출력" style="shape=document;whiteSpace=wrap;html=1;boundedLbl=1;" vertex="1" parent="1">
          <mxGeometry x="300" y="890" width="120" height="80" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-14" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="rWI0O_FinwYjYTAR-48q-13" target="rWI0O_FinwYjYTAR-48q-11">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="rWI0O_FinwYjYTAR-48q-13" value="ai 가 범인인지 선택 Y/N&amp;nbsp;" style="rhombus;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="310" y="730" width="100" height="100" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
g AI 심문하기.drawio…]()

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
