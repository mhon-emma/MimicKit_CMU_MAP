Training ADD steering: (Maybe rewards are getting overridden by agent config file):

python mimickit/run.py --mode train --num_envs 8192 --env_config data/envs/add_steering_humanoid_env.yaml --agent_config data/agents/add_humanoid_agent.yaml --visualize false --log_file output/add_steering.txt --out_model_file output/add_steering.pt 


ADD Steerable latest command (use this):

python mimickit/run.py --mode train --num_envs 8192 --env_config data/envs/add_steering_humanoid_env.yaml --agent_config data/agents/add_task_humanoid_agent.yaml --visualize false --log_file output/add_steering.txt --out_model_file output/add_steering.pt 
