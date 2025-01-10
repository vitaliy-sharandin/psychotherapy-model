from datasets import load_dataset
import json
import yaml
import torch
import transformers
from transformers import GenerationConfig, Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, AutoPeftModelForCausalLM
import numpy as np
from evaluate import load
import optuna

import warnings
warnings.filterwarnings('ignore')


# # Dataset instruction transformation

# Depression dataset with 51 q&a entries was taken for mvp to see if model is indeed training.

# In[2]:


depression_dataset = load_dataset("vitaliy-sharandin/depression-instruct")


# First, we modify our dataset to Alpaca format and create two datasets. One - for testing and evaluation and second one for inference, where responses are not available in formatted prompt.

# In[3]:


def formatting_func_train(example):
  if example.get("context", "") != "":
      input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n"
      f"{example['response']}")

  else:
    input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response:\n"
      f"{example['response']}")

  return {"text" : input_prompt}

def formatting_func_test(example):
  if example.get("context", "") != "":
      input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n")

  else:
    input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response: \n")

  return {"text" : input_prompt}


# In[4]:


formatted_depression_dataset_train = depression_dataset.map(formatting_func_train)
formatted_depression_dataset_test = depression_dataset.map(formatting_func_test)


# # Model load

# We are loading our Llama3.1 model in 4bit quantized form as well as applying Lora for peft training.

# In[6]:


model_name = "NousResearch/Hermes-3-Llama-3.1-8B"

def get_tokenizer():
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  return tokenizer

def get_model():


  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )

  qlora_config = LoraConfig(lora_alpha=16,
                          lora_dropout=0.1,
                          r=64,
                          bias="none",
                          task_type="CAUSAL_LM")

  base_training_model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      device_map = {"": 0}
  )

  base_training_model = prepare_model_for_kbit_training(base_training_model)
  base_training_model = get_peft_model(base_training_model, qlora_config)
  return base_training_model.to('cuda')

base_training_model = get_model()
tokenizer = get_tokenizer()

torch.manual_seed(42)
print(base_training_model)


# # Model instruction fine-tuning

# Here we are defining inference function which returns bleu, rouge and f1 metrics after comparison of predicted and reference responses. My tests have shown that standard trainer `compute_metrics` method is quite inefficient and is not quite suitable for instruction fine-tuning during manual observations or generated results.

# In[2]:


target_folder = 'target/'
model_trained_checkpoint = target_folder + 'model-trained-checkpoint'
model_merged = target_folder + 'final-model'

def bleu_rouge_f1_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=-1)

  labels = [[idx for idx in label if idx != -100] for label in labels]
  predictions = [[idx for idx in prediction if idx != -100] for prediction in predictions]

  decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  bleu = load("bleu")
  bleu_results = bleu.compute(predictions=decoded_predictions, references=decoded_labels)

  rouge = load('rouge')
  rouge_results = rouge.compute(predictions=decoded_predictions, references=decoded_labels)

  f1 = 2 * (bleu_results['bleu'] * rouge_results['rouge1']) / (bleu_results['bleu'] + rouge_results['rouge1'])

  scores = {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "f1": f1
    }

  return scores

def fine_tune(model, tokenizer, train_dataset, eval_dataset, metrics_func, only_evaluate=False):

  supervised_finetuning_trainer = SFTTrainer(model=model,
                                            train_dataset=train_dataset,
                                            eval_dataset=eval_dataset,
                                            args=TrainingArguments(
                                                output_dir=target_folder + "training_results",
                                                num_train_epochs=20,
                                                per_device_train_batch_size=8,
                                                per_device_eval_batch_size=1,
                                                gradient_accumulation_steps=2,
                                                optim="paged_adamw_8bit",
                                                save_steps=1000,
                                                logging_steps=30,
                                                learning_rate=2e-4,
                                                weight_decay=0.001,
                                                fp16=False,
                                                bf16=False,
                                                max_grad_norm=0.3,
                                                max_steps=-1,
                                                warmup_ratio=0.3,
                                                group_by_length=True,
                                                lr_scheduler_type="constant"
                                            ),
                                            compute_metrics=metrics_func,
                                            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))

  if only_evaluate:
    return supervised_finetuning_trainer.evaluate()

  else:
    supervised_finetuning_trainer.train()
    supervised_finetuning_trainer.model.save_pretrained(model_trained_checkpoint)
    tokenizer.save_pretrained(model_trained_checkpoint)

    return supervised_finetuning_trainer.evaluate()


# ## Evaluating base model

# In[12]:


params = {
    "model": base_training_model,
    "tokenizer": tokenizer,
    "train_dataset": formatted_depression_dataset_train["train"],
    "eval_dataset": formatted_depression_dataset_train["train"],
    "metrics_func": bleu_rouge_f1_metrics,
    "only_evaluate": True
}

fine_tune(**params)


# Evaluation metrics are quite low, as out model is not responding in a manner expected.

# ## Fine-tuning base and evaluating tuned model

# In[14]:


params = {
    "model": base_training_model,
    "tokenizer": tokenizer,
    "train_dataset": formatted_depression_dataset_train["train"],
    "eval_dataset": formatted_depression_dataset_train["train"],
    "metrics_func": bleu_rouge_f1_metrics
}

fine_tune(**params)


# We can see that all scores are significantly improved after training.

# # Saving fine-tuned model with adapters to Huggingface

# Now we are able to merge our model with saved peft adapters and push it to Hugging Face repo.
# 
# We have to launch this cell separately, as VRAM will be filled up to fullest point when we merge adapters with original model.

# In[3]:


tuned_model = AutoPeftModelForCausalLM.from_pretrained(
    model_trained_checkpoint,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map = {"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
tokenizer = AutoTokenizer.from_pretrained(model_trained_checkpoint)

merged_model = tuned_model.merge_and_unload()

merged_model.save_pretrained(model_merged, safe_serialization=True)
tokenizer.save_pretrained(model_merged)