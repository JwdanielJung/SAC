# SAC (Search-Augmented unsafe prompts Classification) Frameworks for LLMs

![Framework Overview](assets/SAC_framework.png)

1. **Vector Storing of Unsafe Prompts**

2. **Threshold Optimization**

3. **Similarity Search Based Threshold Filtering**
   - `confi_unsafe`: Confidently unsafe in filtering phase
   - `confi_safe`: Confidently safe in filtering phase
   - `unconfident`: Can't determine
   - `losses`: Incorrect filtering

4. **Classification for Remaining prompts Using Previous Classifiers**
   - `Moderation API`
   - `Perspective API`
   - `Llama-Guard 7B`
   - `GradSafe`
   - `Zero shot prompting GPT-4`

## Implementation

Install the required dependencies using the following command:

```bash
conda create -n sac
conda activate sac
pip install -r requirements.txt
```

Create a `.env` file and add the following line:

```bash
OPENAI_API_KEY = "YOUR_API_KEY"
PERSPECTIVE_API_KEY = "YOUR_API_KEY"
HUGGINGFACE_TOKEN="YOUR_API_KEY"
```

Example implementation codes command:

```bash
python main.py
```

```bash
python main.py --embed_model openai --model llama_guard --is_prepared False
```

```bash
python main.py --embed_model openai --model llama_guard --is_prepared True
```
