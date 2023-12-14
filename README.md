# Advanced LLM Model Optimization and Quantization for Local Inference
## Description
This work is a result of Research Work on seminars in HSE. The authors of this work are grateful to Professor Krylov Vladimir Vladimirovich for setting the problem and valuable recommendations.

This work is done by:

 - Artyom Pyanzin
 - Kristina Ovakimyan
 - Dmirty Vorob`ev
 - Vadim Voevodkin
## Installation
```bash
git clone https://github.com/intel/intel-extension-for-transformers
cd intel-extension-for-transformers
pip install -r requirements.txt
pip install transformers==4.33.1
python setup.py install
```

## Model
### Nous-Hermes-Llama2-7b
[Original weights](https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b) 12.5 Gb
<br>
[Quantized weights](https://drive.google.com/file/d/1_tTD-De4vXCf4Os_b0v4MOUoU0OmXr9G/view?usp=drive_link) 4.7 Gb
<br>
Make `runtime_outs` directory, download quantized weights and place them to `runtime_outs`

## Command line demo
Quantized weights will be loaded if available, otherwise model will be quantized from scratch (requires ~10Gb RAM).
```bash
python cli_demo.py --model-path `path/to/original/weights`
```

## Evaluation

### Usage
```bash
python eval_arc.py --model-path '/path/to/original/weights' \
                   --input-path '/path/to/parquet' \
                   --output-path '/path/to/output'

python accuracy.py --gt-path '/path/to/parquet' \
                   --pred-path '/path/to/output'
```

### ARC-Easy
[Download test](https://huggingface.co/datasets/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Easy/test/0000.parquet?download=true)
### ARC-Challenge
[Download test](https://huggingface.co/datasets/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Challenge/test/0000.parquet?download=true)

## Metrics
| model/accuracy | ARC-Easy | ARC-Challenge |
|----------------|----------|---------------|
| original       | **0.7946**   | **0.4735**        |
| quantized      | 0.5320   | 0.3993        |

## Performance
i7-12700k, 16Gb
| model | RAM | Tokens/s |
|----------------|----------|---------------|
| quantized | ~6Gb | ~4.8 |
