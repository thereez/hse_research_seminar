# Advanced LLM Model Optimization and Quantization for Local Inference

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