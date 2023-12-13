from argparse import ArgumentParser
import jsonlines
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig



PROMPT = "### Instruction:\n{question}\n\n### Input:\n{context}\nAnswer with a single letter.\n\n### Response:\n"


def main():
    model_name = args.model_path
    woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)


    arc_ds = load_dataset("parquet", data_files={"test": args.input_path})

    results = []

    out_path = args.output_path
    writer = jsonlines.open(out_path, 'w')

    for i, item in enumerate(tqdm(arc_ds['test'])):
        question = item['question']
        statements = item['choices']
        statements_list = [f"{c}. {s}" for s, c in zip(statements['text'], statements['label'])]
        statements_str = '\n'.join(statements_list)
        prompt = PROMPT.format(question=question, context=statements_str)

        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        response = model.generate(inputs,
                                  streamer=streamer,
                                  interactive=True,
                                  ignore_prompt=True,
                                  do_sample=True,
                                  max_new_tokens=5)[0]
        response_text = tokenizer.decode(response)

        pred = {
            'id': item['id'],
            'answer': response_text[0]
        }
        results.append(pred)
        writer.write(pred)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()
	main(args)
