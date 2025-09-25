 # Overview 

 - Define keys
```
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"  # if using Claude
```
- Install
```
pip install -r human-eval/requirements.txt 
python -m simple-evals.simple_evals --list-models
```
- Quick test:
`python -m simple-evals.simple_evals --model=gpt-4.1 --eval=mmlu --examples=10`



!git clone https://github.com/openai/simple-evals.git
!pip install openai human-eval
!pip install -q --upgrade torch
!pip install -q transformers triton==3.4 kernels
!pip uninstall -q torchvision torchaudio -y
%pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" trackio
!pip install anthropic

