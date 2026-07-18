import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
import shortuuid
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.CLMoE.moe_routing_monitor import MoERoutingMonitor
from llava.model import *
import math

#  代码通过挂载hook的形式，在生成过程中实时监控每一步的路由决策和生成词的概率分布，记录到日志文件中，方便后续分析和排查问题。
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
def eval_model(args):
    # Model
    disable_torch_init()
    set_global_seed(args.seed)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    from CLMoE.peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"  # 强制指定 Llama-2 的停止符
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Mounting LoRA: {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path, cka_beta=args.cka_beta)
    model.eval()
    
    image_processor = None
    context_len = 4096    
    question_file_path = os.path.expanduser(args.question_file)
    questions = []
    global_definition = ""  
    
    try:
        with open(question_file_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            # 兼容 SuperNI 字典或普通列表
            if isinstance(loaded, dict):
                questions = loaded.get("Instances", [])
                
                # 在根节点提前把 Definition 抓出来
                def_list = loaded.get("Definition", [""])
                if isinstance(def_list, list) and len(def_list) > 0:
                    global_definition = def_list[0]
                elif isinstance(def_list, str):
                    global_definition = def_list
            else:
                questions = loaded
    except json.decoder.JSONDecodeError:
        # 兼容 JSONL 多行格式
        with open(question_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
                    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(answers_file), "logs"), exist_ok=True)

    ans_file = open(answers_file, "w")
    count = 0

    monitor = MoERoutingMonitor(top_k=2)
    monitor.attach(model)
    
    for i, line in enumerate(tqdm(questions)):
        monitor.reset()
        count += 1
        
        idx = line.get("id", str(i))  
        
        definition = line.get("definition", line.get("Definition", global_definition))
        if isinstance(definition, list) and len(definition) > 0:
            definition = definition[0]
            
        user_input = line.get("prompt", line.get("inputs", line.get("input", "")))
        
        qs = f"Instruction: {definition}\n\nInput: {user_input}" if definition else f"Input: {user_input}"
        cur_prompt = qs
        # prompt 
        prompt = f"[INST] {qs} [/INST]" 
        # 3. Tokenize
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        input_token_len = input_ids.shape[1]
        
        # 4. Monitor Strings 
        monitor_strings = []
        for tid in input_ids[0]:
            monitor_strings.append(tokenizer.convert_ids_to_tokens([tid])[0])

        stop_str = tokenizer.eos_token
        keywords = [stop_str] if stop_str else []
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids) if keywords else None
        
        meta_info = {"img_path": None} 
        monitor.set_generation_state(monitor_strings, meta=meta_info)
        
        # 提前算好 next_token_logits (可选)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_ids = torch.topk(probs, 5, dim=-1)

        # 5. 生成过程与参数配置
        gen_kwargs = {
            "max_new_tokens": 1024,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "num_beams": args.num_beams,
            "repetition_penalty": 1.05,     
            "use_cache": True,              
            "return_dict_in_generate": True,
            "output_scores": True           
        }
        
        # 动态处理采样参数，防止 transformers 报错
        if args.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                gen_kwargs["top_p"] = args.top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            generate_outputs = model.generate(
                input_ids=input_ids,
                images=None, 
                stopping_criteria=[stopping_criteria] if stopping_criteria else None,
                **gen_kwargs
            )
        
        # 提取序列和 ID
        output_ids = generate_outputs.sequences
        gen_ids = output_ids[:, input_token_len:]
    
        print(f"\n[Debug] Question ID: {idx} | Probabilities:")
        generated_probs_log = []
        
        if generate_outputs.scores:
            for step, step_logits in enumerate(generate_outputs.scores):
                step_probs = torch.nn.functional.softmax(step_logits, dim=-1)
                
                if step < len(gen_ids[0]):
                    chosen_token_id = gen_ids[0][step].item()
                    chosen_prob = step_probs[0, chosen_token_id].item()
                    token_str = tokenizer.convert_ids_to_tokens(chosen_token_id)
                    
                    # 存入列表，用于写进最终的 jsonl 里
                    generated_probs_log.append({
                        "step": step+1, 
                        "token": str(token_str), 
                        "id": chosen_token_id, 
                        "prob": round(chosen_prob, 4)
                    })
                    
                    # 终端只打印前15个词，防止刷屏
                    if step < 15: 
                        print(f"  Step {step+1:02d} | Token: {str(token_str):<15} | ID: {chosen_token_id:<5} | Prob: {chosen_prob:.4f}")
            
            if len(gen_ids[0]) > 15:
                print(f"  ... (Skipped {len(gen_ids[0]) - 15} tokens)")
        # ==========================================================

        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
        outputs_tokens = tokenizer.convert_ids_to_tokens(gen_ids[0])
        outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        full_text = " ".join(monitor_strings) + " ".join(outputs_tokens)

        outputs = outputs.strip()

        # 兜底清理可能残留的停止词文本
        if stop_str and outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        ans_id = shortuuid.uuid()
        

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {"generation_probs": generated_probs_log}
        }) + "\n")
        ans_file.flush()
        
        # 记录路由数据
        routing_path = os.path.join(os.path.dirname(answers_file), f"logs/routing_{idx}log.json")
        routing_data = monitor.group()
        len_text = len(monitor_strings) + len(outputs_tokens) - 1
        
        if routing_data:
            first_layer = next(iter(routing_data))
            first_field = next(iter(routing_data[first_layer]))
            first_len = len(routing_data[first_layer][first_field])

        with open(routing_path, "w", encoding="utf-8") as f:
            json.dump({"text": full_text,"len_text":len_text, "routing": routing_data}, f, indent=2, ensure_ascii=False)
        
    ans_file.close()
    monitor.detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cka_beta", type=float, required=True)
    
    args = parser.parse_args()
    eval_model(args)