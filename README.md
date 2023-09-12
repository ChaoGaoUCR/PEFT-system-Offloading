# PEFT-system-Offloading

Running command with opt-13b: deepspeed --num_gpus 1 run_model.py --model facebook/opt-13b --batch-size 8 --prompt-len 512 --gen-len 32 --disk-offload
//
Running command with LLama-13b: deepspeed --num_gpus 1 run_model.py --model decapoda-research/llama-13b-hf --batch-size 8 --prompt-len 512 --gen-len 32 --disk-offload
