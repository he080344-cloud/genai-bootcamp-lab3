import gradio as gr
import torch
from PIL import Image
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          AutoTokenizer, AutoModelForCausalLM)

APP_TITLE   = "Yazeed — Image Storyteller"
STORY_STYLE = "an epic fantasy saga"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

blip_id = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_id)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_id, dtype=dtype).to(device)

llm_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_id)
llm = AutoModelForCausalLM.from_pretrained(llm_id, dtype=dtype).to(device)

def caption_image(img):
    inputs = blip_processor(img, return_tensors="pt").to(device, dtype)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def write_story(caption):
    prompt = f"Write a short, vivid 3-sentence story inspired by: '{caption}'. Make it {STORY_STYLE}."
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=False).to(device)
    out = llm.generate(inputs, max_new_tokens=120, do_sample=True, temperature=0.9,
                       top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

def app(img):
    if img is None:
        return "Please upload an image.", ""
    cap = caption_image(img.convert("RGB"))
    return cap, write_story(cap)

gr.Interface(
    fn=app,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[gr.Textbox(label="Caption"), gr.Textbox(label="Story")],
    title=APP_TITLE,
    description=f"Upload a photo — it gets captioned, then turned into {STORY_STYLE}.",
).launch()
