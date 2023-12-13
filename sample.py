from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

# Please change to local path to model, llama2 does not support online conversion, currently.
model_name = "/home/apyanzin/repos/draft/Nous-Hermes-llama-2-7b"
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

while True:
    prompt = input("> ").strip()
    if prompt == "quit":
        break
    b_prompt = "### Instruction: \n\
                {} \n\
                \n\
                ### Response: ".format(prompt)  # prompt template for llama2
    inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True)
