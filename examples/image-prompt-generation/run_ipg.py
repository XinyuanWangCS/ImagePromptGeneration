import os
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append('../../')

from rlprompt.models import (ImagePromptModelConfig, SinglePromptModelConfig,
                             make_image_prompt_model, make_single_prompt_model)
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from ipg_helpers import (ImagePromptGenerationRewardConfig,
                         ImagePromptGenerationDatasetConfig,
                         make_image_prompt_generation_reward,
                         make_image_prompot_generation_dataset)


# Compose default config
config_list = [ImagePromptGenerationRewardConfig,
                ImagePromptGenerationDatasetConfig, 
                ImagePromptModelConfig,
                SinglePromptModelConfig, 
                SQLModuleConfig, 
                TrainerConfig]
cs = compose_hydra_config_store('base_ipg', config_list)

@hydra.main(version_base=None, config_path="./", config_name="fsc_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    (train_dataset, val_dataset, test_dataset, num_classes, verbalizers, template) = make_image_prompot_generation_dataset(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    # LM with a MLP layer, use greedy search to generate a prompt
    policy_model = make_image_prompt_model(config)
    # Same model wrapped by SinglePromptModel
    prompt_model = make_single_prompt_model(policy_model, config)
    reward = make_image_prompt_generation_reward(num_classes, verbalizers, template, config) #reward use GPT
    algo_module = make_sql_module(prompt_model, reward, config)

    print(policy_model)
    print("****************")
    print(prompt_model)
    print("****************")
    print(reward)
    print("****************")
    print(algo_module)
    # Hack for few-shot classification - Each batch contains all examples
    config.train_batch_size = len(train_dataset)
    config.eval_batch_size = len(val_dataset)
    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()
