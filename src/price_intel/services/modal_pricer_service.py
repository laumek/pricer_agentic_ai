"""
Modal app serving the fine-tuned pricing model.
"""

import modal
from modal import App, Volume, Image

from price_intel.config import (
    BASE_MODEL,
    FINETUNED_MODEL,
    REVISION,
    CACHE_DIR,
    GPU_TYPE,
    MIN_CONTAINERS,
    QUESTION,
    PREFIX,
)

app = App("pricer-service")

image = (
    Image.debian_slim()
    .pip_install(
        "huggingface_hub",
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "peft",
    )
)

# Modal secret for HF token
secrets = [modal.Secret.from_name("hf-secret")]

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU_TYPE,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            set_seed,
        )
        from peft import PeftModel

        set_seed(42)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model,
            FINETUNED_MODEL,
            revision=REVISION
        )

    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)

        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")

        outputs = self.fine_tuned_model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=5,
            num_return_sequences=1,
        )
        result = self.tokenizer.decode(outputs[0])

        contents = result.split("Price is $")[1]
        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0.0
