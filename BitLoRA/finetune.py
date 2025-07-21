# python -m BitLoRA_finetune.finetune

import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from .peft import BitLoraConfig, get_peft_model
from typing import Optional
import torch
import torch.nn as nn
from .BitNet.bitnet import BitLinear

# 1. 모델 및 토크나이저 로드
model_id = "tiiuae/Falcon-E-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision="prequantized")
tokenizer.pad_token = tokenizer.eos_token  # SFTTrainer에서 pad token 명시

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    revision="prequantized"
)

print("\n🔍 [초기 모델 구조: nn.Linear만 있는 상태]")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"[Linear] {name} ({module.in_features} → {module.out_features})")




def replace_linear_with_bitnet_linear(model, previous_dtype: Optional[torch.dtype] = None):
    """
    """
    # Recursively replace linear layers
    if previous_dtype is None:
        previous_dtype = torch.get_default_dtype()

        model_dtype = model.dtype
        torch.set_default_dtype(model_dtype)

        previous_dtype = model_dtype

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_with_bitnet_linear(module, previous_dtype=previous_dtype)
        
        # Replace nn.Linear layers, but skip 'lm_head'
        if name != 'lm_head' and isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            with torch.device(module.weight.device):
                # Create a new instance of the custom linear layer
                new_layer = BitLinear(in_features, out_features, bias=bias)
                # Copy weights and biases
                with torch.no_grad():
                    new_layer.weight.copy_(module.weight)
                    if bias:
                        new_layer.bias.copy_(module.bias)
            
            # Replace the layer in the model
            setattr(model, name, new_layer)
    return model

# 2. BitLinear로 Linear 대체
model = replace_linear_with_bitnet_linear(model)

# ✅ 여기에 삽입 (BitLinear로 잘 교체됐는지 확인)
print("\n🔍 [BitLinear 변환 후 구조 확인]")
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        print(f"[BitLinear] {name}")

# 3. BitLoRA 구성
bitlora_config = BitLoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model = get_peft_model(model, bitlora_config)

print("\n🔍 [LoRA 실제 삽입 계층 확인]")
for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        print(f"[LoRA Layer] {name}")
        for adapter in module.lora_A:
            print(f"  - adapter: {adapter}, A.shape: {module.lora_A[adapter].weight.shape}")

# 4. 예시 데이터셋 불러오기
dataset = load_dataset("Abirate/english_quotes")
dataset = dataset["train"].train_test_split(test_size=0.1)

# 5. 전처리 함수 정의
def preprocess(example):
    return {"text": example["quote"]}

tokenized_dataset = dataset.map(preprocess)

# 6. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./bitlora-falcon3b-sft",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    remove_unused_columns=False,
    report_to="none",
    push_to_hub=False
)

# 7. SFTTrainer 정의
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    max_seq_length=128,
    packing=True,
    dataset_text_field="text",
)

# ✅ 여기에 4단계 체크 코드 삽입 (trainer.train() 바로 전!)
print("\n🔍 [학습 가능한 파라미터 목록]")
trainable_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)
print(f"총 학습 가능한 파라미터 수: {len(trainable_params)}")
for name in trainable_params:
    print(f" - {name}")

# 8. 학습 수행
trainer.train()

# 9. 필요시 저장
trainer.save_model("./bitlora-falcon3b-sft-final")
tokenizer.save_pretrained("./bitlora-falcon3b-sft-final")
