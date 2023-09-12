import argparse
import gc
import multiprocessing as mp
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from accelerate import init_empty_weights
# from timer import timers
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
                          BloomForCausalLM, OPTForCausalLM, LlamaForCausalLM,
                        )
from transformers.deepspeed import HfDeepSpeedConfig
from utils import (GB, add_model_hooks, cache_bytes, disable_torch_init,
                   get_filename, get_quant_config, hidden_bytes, meta_to_cpu,
                   model_bytes, write_benchmark_log)
from packaging import version
from datasets import load_dataset

def get_model_config(model_name):
    if "175b" in model_name:
        config = AutoConfig.from_pretrained("facebook/opt-66b")
        config.hidden_size = 12288
        config.word_embed_proj_dim = 12288
        config.ffn_dim = 12288 * 4
        config.num_attention_heads = 96
        config.num_hidden_layers = 96
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if 'bloom' in model_name:
        config.model_type = 'bloom'

    return config

def get_ds_model(
    model_name,
    dtype,
    cpu_offload,
    disk_offload,
    offload_dir,
    dummy_weights,
    bits,
    group_size,
):

    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    pin_memory = bool(args.pin_memory)

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size, # 0, 
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": args.batch_size,
        "wall_clock_breakdown": False,
    }

    if bits == 4:
        quant_config = get_quant_config(config, bits=bits, group_size=group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )

    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=pin_memory,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=9 * GB if config.model_type == 'bloom' else 2 * GB,
        )
        ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
        }

    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    get_accelerator().empty_cache()
    gc.collect()

    if config.model_type in ["bloom", "bloom-7b1"]:
        model = BloomForCausalLM.from_pretrained(
            dummy_weights or model_name, torch_dtype=dtype,
        )
    elif config.model_type == "opt":
        model = OPTForCausalLM.from_pretrained(
            dummy_weights or model_name, torch_dtype=dtype,
        )
    elif config.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            dummy_weights or model_name, torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unexpected model type: {config.model_type}")

    model = model.eval()


    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(f"model.config = {model.config}")

    return model

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def run_generation(
    model_name,
    batch_size,
    prompt_len,
    gen_len,
    cpu_offload,
    disk_offload,
    offload_dir,
    num_nodes,
    num_gpus_per_node,
    dummy,
    output_file,
    verbose,
    kv_offload,
    quant_bits,
    quant_group_size,
    pin_kv_cache,
    async_kv_offload,
):
    # Load tokenizer
    config = get_model_config(model_name)
    return_token_type_ids = True 
    padding_side = "left" if config.model_type in ["opt"] else "right"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        return_token_type_ids=return_token_type_ids,
        padding_side=padding_side
    )

    tokenizer.pad_token = tokenizer.eos_token

    if hasattr(config, 'torch_dtype'):
        dtype = config.torch_dtype
    else:
        dtype = torch.float

    if dummy:
        filename = os.path.join(
            offload_dir, f"{model_name.replace('/', '-')}-hf-weights/"
        )
        if not os.path.exists(filename):
            print("create dummy weights")
            with init_empty_weights():
                if config.model_type == 'opt':
                    model = OPTForCausalLM(config)
                elif config.model_type in ["bloom", "bloom-7b1"]:
                    model = BloomForCausalLM(config)
                elif config.model_type == "llama":
                    model = LlamaForCausalLM(config)
                else:
                    raise ValueError(f"Unexpected model type: {config.model_type}")                    
            model.save_pretrained(
                filename, state_dict=meta_to_cpu(model.state_dict(), torch.float16)
            )
        dummy_weights = filename
    else:
        dummy_weights = None

    file_path = './dataset/openbookqa/train.json'
    dataset = load_dataset("json", data_files=file_path)
    seq = dataset['train'][0].get('instruction')
    prompt = generate_prompt(seq, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    print("load model")
    with torch.no_grad():
        model = get_ds_model(
            model_name,
            dtype,
            cpu_offload,
            disk_offload,
            offload_dir,
            dummy_weights,
            quant_bits,
            quant_group_size,
        )
        print("generation begin!")
        # generation_output = model.generate(
        #     input_ids=input_ids,
        #     # generation_config=generation_config,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     max_new_tokens=32,
        #     use_cache=True,
        # )
        # print(generation_output)
        out = model(input_ids)
        print(out)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="model name or path; currently only supports OPT and BLOOM models")
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights for benchmark purposes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen-len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")
    parser.add_argument("--pin-memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
    parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
    parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
    parser.add_argument("--kv-offload", action="store_true", help="Use kv cache cpu offloading.")
    parser.add_argument("--log-file", type=str, default="auto", help="log file name")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--pin_kv_cache", action="store_true", help="Allocate kv cache in pinned memory for offloading.")
    parser.add_argument("--async_kv_offload", action="store_true", help="Using non_blocking copy for kv cache offloading.")
    args = parser.parse_args()

    deepspeed.init_distributed()    
    num_gpus_per_node = get_accelerator().device_count()
    num_nodes = dist.get_world_size() // num_gpus_per_node
    run_generation(
    args.model,
    args.batch_size,
    args.prompt_len,
    args.gen_len,
    args.cpu_offload,
    args.disk_offload,
    os.path.abspath(os.path.expanduser(args.offload_dir)),
    num_nodes,
    num_gpus_per_node,
    args.dummy,
    args.log_file,
    args.verbose,
    args.kv_offload,
    args.quant_bits,
    args.quant_group_size,
    args.pin_kv_cache,
    args.async_kv_offload,
)