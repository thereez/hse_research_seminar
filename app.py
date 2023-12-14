import gradio as gr
import random
import time
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

model_name = "/path/to/Nous-Hermes-llama-2-7b"
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
last_msg = ""


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    def user(user_message, history):
        global last_msg
        last_msg = str(user_message)
        print(last_msg)
        return "", history + [[user_message, None]]

    def bot(history, top_p, top_k, temperature, repetition_penalty, max_new_tokens):
        print('Running')
        global last_msg
        global model
        global tokenizer
        b_prompt = "### Input:\n{}\n### Response:\n".format(last_msg)
        inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
        outputs = model.generate(inputs, interactive=True, ignore_prompt=True, do_sample=True,
                                temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens)
        outputs = tokenizer.decode(outputs[0][:-1])
        history[-1][1] = outputs
        return history

    top_p = gr.Slider(
        minimum=0,
        maximum=1.0,
        value=1,
        step=0.05,
        interactive=True,
        label="Top-p",
    )
    top_k = gr.Slider(
        minimum=0,
        maximum=100,
        value=50,
        step=1,
        interactive=True,
        label="Top-k",
    )
    temperature = gr.Slider(
        minimum=0.1,
        maximum=2.0,
        value=1,
        step=0.1,
        interactive=True,
        label="Temperature",
    )
    repetition_penalty = gr.Slider(
        minimum=0,
        maximum=10,
        value=1,
        step=0.1,
        interactive=True,
        label="Repetition penalty",
    )
    max_length_tokens = gr.Slider(
        minimum=0,
        maximum=512,
        value=32,
        step=8,
        interactive=True,
        label="Max Generation Tokens",
    )
    print("Loaded")
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, top_p, temperature, top_k, repetition_penalty, max_length_tokens], chatbot
    )
    submit.click(user, [msg, chatbot], [msg, chatbot]).then(
        bot, [chatbot, top_p, temperature, top_k, repetition_penalty, max_length_tokens], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_name="0.0.0.0")