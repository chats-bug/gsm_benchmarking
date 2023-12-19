from accelerate import Accelerator
from dataclasses import dataclass
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import yaml

accelerator = Accelerator()
device = accelerator.device


@dataclass
class Model:
	hf_model_id: str
	hf_tokenizer_id: str


@dataclass
class Dataset:
	name: Optional[str]
	path: str
	type: str = 'csv'


@dataclass
class Config:
	model_config: Model
	dataset_config: Dataset
	batch_size: int = 1
	output_path: str = 'results.csv'
	max_new_tokens: int = 1500


def load_config(path):
	with open(path) as f:
		yaml_config = yaml.load(f, Loader=yaml.FullLoader)
	
	yaml_config = Config(**yaml_config)
	return yaml_config


def load_objects(config):
	model = AutoModelForCausalLM.from_pretrained(config.model_config.hf_model_id, trust_remote_code=True)
	tokenizer = AutoTokenizer.from_pretrained(config.model_config.hf_tokenizer_id)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_size = "right"
	model.to(device)
	return model, tokenizer


def load_data(config):
	if config.dataset_config.type == 'hf':
		# dataset = load_dataset(config.dataset_config.name)
		# return dataset
		raise NotImplementedError
	elif config.dataset_config.type == 'csv':
		print(f"Loading data: {config.dataset_config.name} from {config.dataset_config.path}")
		df = pd.read_csv(config.dataset_config.path)
		return df


def prepare_prompt(question, system_message=None, test_prompt=None):
	if not system_message:
		system_message = "Follow these instructions : \n\
			1)At the end of solution give final answer value exactly like this #### Final Answer : <answer value>\n"
	
	if not test_prompt:
		test_prompt = "<|im_start|>system\n{system_message}<|im_end|>\n\
						<|im_start|>user\n{question}<|im_end|>\n\
						<|im_start|>assistant"
	return test_prompt.format(question=question, system_message=system_message)


def main():
	config = load_config('config.yaml')
	model, tokenizer = load_objects(config)
	dataset = load_data(config)
	
	questions = dataset['question'].to_list()
	# prepare the prompt
	# TODO: Parallelize this
	prompts = [prepare_prompt(q) for q in questions]
	results = []
	
	for i in range(0, len(prompts), config.batch_size):
		batch = prompts[i:i + config.batch_size]
		model_inputs = tokenizer(
			batch,
			return_tensors='pt'
		).to(device)
		greedy_output = model.generate(**model_inputs, max_new_tokens=config.max_new_tokens)
		output = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
		# make a list of tuples of question and answer
		results.extend(list(zip(batch, output)))
	
	# save the results
	results_df = pd.DataFrame(results, columns=['question', 'generated_answer'])
	results_df.to_csv(config.output_path, index=False)
	
	
