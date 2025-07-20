import os
import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from trl import TrlParser, ModelConfig

# Custom imports
#from diffu_grpo_trainer import DiffuGRPOTrainer
from diffu_grpo_trainer_mini import DiffuGRPOTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from data_utils import (
    get_countdown_questions,
    get_cd4,
    set_random_seed,
    preprocess_dataset
)
from loader import load_tokenizer, load_model
from diffugpt_model import DiffusionModel, DiffusionArguments
os.environ["WANDB_PROJECT"] = "diffuGPT-GRPO"


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset based on configuration
    if grpo_config.dataset == "cd4":
        dataset = get_cd4("train")
        #tokenizer
        tokenizer = load_tokenizer('model_config')
        #dataset
        dataset = preprocess_dataset(dataset, tokenizer, cutoff_len=64)
        #reward model
        reward_functions = [countdown_reward_func]
        #model : load_ckpt
        config = AutoConfig.from_pretrained(grpo_config.model_path)
        #适配自训练的模型
        config.vocab_size = 31
        config.n_positions = 512
        model = AutoModelForCausalLM.from_config(config,attn_implementation="eager").to(device)
        model = DiffusionModel(model, config, grpo_config)
        raw_ckpt = torch.load("./model_config/pytorch_model.bin", map_location="cpu")

        missing, unexpected = model.load_state_dict(raw_ckpt, strict=False)
        print("Missing:", len(missing), "Unexpected:", len(unexpected))  
    

    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
        #tokenizer
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(grpo_config.model_path)   # 如有需要可再改 config.vocab_size 等字段
        model = AutoModel.from_config(config) .to(device)
        '''
        model = AutoModel.from_config(
            grpo_config.model_path
        ).to(device)
        '''

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation  
    
    model.config.use_cache = False

    # Initialize and run trainer
    if grpo_config.dataset == "cd4":
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            processing_class=tokenizer
        )
    elif grpo_config.dataset == "countdown":
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            
        )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)

    '''
    conda env create -f env.yml
    pip install -r requirements.txt
    export HF_ENDPOINT=https://hf-mirror.com
    CUDA_VISIBLE_DEVICES=0, bash run_mini.sh
    fb7dbb7954b41447816878e051e5e6aeaba89774

    git config --global user.email '791054860@qq.com'
    git config --global user.name Liudongxu1121

    pip源: 'https://repo.huaweicloud.com/repository/pypi/simple'
    config源:  
    
    '''
