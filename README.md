# OCIP

일단은 실험부터 진행해보자.
이 연구를 진행하고 말고는 실험을 진행해서 실제로 overfitting이 효과적일까를 보고 나서 판단해보는 것으로 하자.

SFTTrainer에서 DataCollator가 잘 작동하는지 확인할 필요도 있을 듯

- PPO 코드 상태가 이상함. 얘는 돌려보면서 그냥 여러 epoch 학습을 다 돌리는 것이 좋을 것 같음.

Dataset
- SFT: [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots?row=0)
- RM: [yitingxie/rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)
- PPO & DPO: [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 전체 사용


Model
- Llama-2-7b(SFT: 1ep)
- Llama-2-7b(SFT: 3ep)
- RM: SFT-Llama-2-7b(RM: 1ep)
- SFT-Llama-2-7b(PPO: 1ep)
- SFT-Llama-2-7b(PPO: 3ep)
- SFT-Llama-2-7b(DPO: 1ep)
- SFT-Llama-2-7b(DPO: 3ep)

- Mistral에 대해서도 똑같이 실험을 진행하는 것이 좋을 것 같음.
- 보다 효율적인 튜닝을 위해 Flash-Attention-2 & LoRA를 활용할 계획

Parameters
- batch_size = 64
- micro_batch_size = 8
- graident_accumulation_steps = 8
- lr_rate = 2e-5
- gradient_checkpointing = False(batch_size 변형이 필요할 수도 있음.)
- lr_scheduler_type = "cosine"
- max_length = 2048
- weight_decay = 0
- bf16 = True

LoRA Parameters
- lora_r = 8
- lora_alpha = 16
- lora_dropout = 0.05
- lora_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
- bias = None
- task_type = "CAUSAL_LM"

Benchmark
- 솔직히 Zephyr paper에서는 model-based evaluation만 진행해서 overfitting이 academic benchmark에서도 효과적인지는 잘 모름.
- 따라서 두 분야의 벤치마크에서 모두 성능을 평가해보는 것이 좋을 것 같음. 
- Chat evaluation: MT-Bench, AlpacaEval
- HF Open-LLM Leaderboard

overfitting이 성능 향상을 알려주는 지표로 사용될 수 있을까?

- 여러 training method의 overfitting 정도를 비교해보며 성능 평가
   - SFT, RLHF(PPO), DPO의 overfitting 정도에 따른 벤치마크 성능 평가 진행
     - 이때 옵션을 달리해보며 성능 평가 진행(epoch 등)

   - 여러 모델과 벤치마크에서 성능 평가를 진행
     - 다양성에 따른 성능 평가를 해야 보다 완벽한 평가가 가능할 것 같음
     - 벤치마크의 경우 PPO와 DPO를 사용해야 하니 chat benchmark에서만 성능 평가가 가능할 것 같음.
     - Model: Mistral-7B, Llama2-7B, etc.
     - Benchmark: AlpacaEval, KoalaEval, MT-Bench

   - qualitative test를 통해서 overfitting으로 인해 발생하는 reward hacking이 발생하지 않는지 확인
     - overfitting이 발생하게 되면 RL의 경우 reward hacking의 문제가 발생할 수 있으므로 이를 확인하기 위해 qualitative test 또한 진행해야 함.
       - RL에서는 reward hacking의 문제가 발생하기 떄문에 overfitting이 문제를 야기할 수 있으나, DPO의 경우 overfitting이 발생할 경우 보다 높은 퀄리티의 output을 내놓도록 개선되니 이러한 점을 토대로 DPO의 가능성을 제안하면 좋을 것 같음. 물론 이건 가설이기 떄문에 틀릴 가능성도 존재함.
     - 만약 degeneration이 발생한다면 이를 해결하기 위한 regularization 등의 term을 탐색해봐야 할 것 같음.

   - regularization을 활용해서 DPO의 overfitting을 방지한다면 성능에 어떤 변화가 발생할까?
     - 이를 통해 DPO에서 보다 overfitting이 효과적이라는 것을 발견하거나 아니면 그렇지 않다는 것을 분석할 수 있을 것 같음.



벤치마크에 따른 데이터셋도 여러가지가 있을 것임. 그렇다면 어떤 경우에 어떤 모델을 사용해야 할 지 생각해야 함. 일단 나의 컴퓨팅 자원 특성 상 엄청난 양의 데이터에서 학습은 불가할 것 같고, 어느 정도 샘플링을 거치거나 근데 생각해보니까 HF4 팀에서 만든 데이터셋은 200K 정도의 양이니까 충분히 사용 가능하지 않을까 


Training Setup
- SFT Dataset: HuggingFaceH4/ultrachat_200k
- Reward Modeling & DPO: HuggingFaceH4/ultrafeedback_binarized(train_prefs, test_prefs)
- PPO: HuggingFaceH4/ultrafeedback_binarized(train_gen, test_gen)

### Weights & Bias login
```
wandb login
```

### Finetuning Command

**SFT**
```
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes GPU_NUMS finetuning/SFT.py \
    --hf_token HF_ACCESS_TOKEN \
    --
```