import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.CLMoE.moe_routing_monitor import MoERoutingMonitor
from llava.model import *
from PIL import Image
import math,re


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
def get_num_image_tokens_by_name(vision_tower_path: str):
    # e.g. vision_tower_path = "/srv/.../clip-vit-large-patch14-336" 这里注意换模型要改视觉塔名称
    m = re.search(r"patch(\d+)-(\d+)", vision_tower_path)
    if not m:
        raise ValueError(f"Cannot parse vision_tower path: {vision_tower_path}")
    patch_size = int(m.group(1))
    image_size = int(m.group(2))
    return (image_size // patch_size) ** 2

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # with open(os.path.expanduser(args.question_file), "r") as f:
    #     questions = json.load(f)
    question_file_path = os.path.expanduser(args.question_file)
    questions = []
    try:
        with open(question_file_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            # 兼容 SuperNI 字典或普通列表
            questions = loaded.get("Instances", loaded) if isinstance(loaded, dict) else loaded
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
    for line in tqdm(questions):
        monitor.reset()
        count += 1
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        #这里讲token以及image token放到一起
        print(args.image_folder)
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        input_token_len = input_ids.shape[1]
        #通过视觉塔的名称知道<image>要预留多少空位
        num_image_tokens = get_num_image_tokens_by_name("clip-vit-large-patch14-336")
        monitor_strings = []
        # print(input_ids)
        for tid in input_ids[0]:
            if tid == IMAGE_TOKEN_INDEX:  # 即 -200
                # 展开成连续的视觉tokens占位符
                for vi in range(num_image_tokens):
                    monitor_strings.append(f"<IMAGE_{vi}>")
            else:
                # 正常文本token转成可读形式
                monitor_strings.append(tokenizer.convert_ids_to_tokens([tid])[0])

        
        # print("num_image_tokens:",num_image_tokens)
        # print("token per batch:",num_image_tokens+input_token_len)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        meta_info = {
            "img_path": image_file if "image_file" in locals() else None,
        }
    
        monitor.set_generation_state(monitor_strings, meta=meta_info)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
        
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs_tokens = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
        gen_ids = output_ids[:, input_token_len:]
        outputs_tokens = tokenizer.convert_ids_to_tokens(gen_ids[0])
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        full_text = " ".join(monitor_strings) + " ".join(outputs_tokens)

        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # output_visual = output_visual.strip()
        # print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        # monitor_strings + " <s> " + outputs
        routing_path = os.path.join(os.path.dirname(answers_file), f"logs/routing_{idx}log.json")
        routing_data = monitor.group()
        len_text = len(monitor_strings) + len(outputs_tokens) -1
        if routing_data:
            # 拿第一层 layer 名称
            first_layer = next(iter(routing_data))
            # 拿第一层里的第一个字段，比如 topk_idx
            first_field = next(iter(routing_data[first_layer]))
            # 获取长度
            first_len = len(routing_data[first_layer][first_field])
        
            if first_len != len_text:
                print(f"[Warning] '{first_layer}' -> '{first_field}' length {first_len} ≠ expected {len_text}")

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
    
    args = parser.parse_args()

    eval_model(args)
