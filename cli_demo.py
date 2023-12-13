from argparse import ArgumentParser
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig


def main(args):
    woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=woq_config, trust_remote_code=True)

    while True:
        prompt = input("> ").strip()
        if prompt == "quit":
            break
        b_prompt = "### Instruction: \n\
                    {} \n\
                    \n\
                    ### Response: ".format(prompt)
        inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
        outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True, do_sample=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str)
    args = parser.parse_args()
    main(args)
