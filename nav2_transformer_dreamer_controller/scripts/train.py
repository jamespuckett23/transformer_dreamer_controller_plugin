import os
import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import threading

from geometry_msgs.msg import PoseStamped

from nav2_dt_planner.env.nav2_gz import Nav2Sim
from nav2_dt_planner.models.transDreamer.model.transdreamer import TransDreamer
from nav2_dt_planner.models.transDreamer.train import anneal_learning_rate, anneal_temp, simulate_test
from nav2_dt_planner.config.configs import cfg

# Use the planning class to interface with the nav2 stack

# Once the model has been trained, use the planning class to interface with the trained model

def start_sim():
    # send ROS msg through ROS-GZ bridge to start GZ sim
    return


# def main():
#     # configure training parameters
#     # load Transformer Dreamer model
#     # gazebo sim and nav2 stack static stuff ready to go -> gym.reset()
#     # training loop
#         # Generate task & dynamic sim and nav2 stack
#             # send start/goal to Nav2 autonomy stack
            
#         # interact with Gym environment to train

#     return 0


def main():
    # start ROS node - training node
    global visualization_enabled

    # Create the environment with visualization initially off
    env = Nav2Sim()
        # maybe rename for GZ

    # Start a separate thread to listen for visualization toggle key
    # threading.Thread(target=toggle_visualization, args=(env,), daemon=True).start()

    # Define options with necessary parameters
    # class Config:
    #     def __init__(self):
    #         self.gamma = 0.99        # Discount factor
    #         self.alpha = 0.001       # Learning rate
    #         self.epsilon = 0.5       # Starting epsilon for exploration
    #         self.epsilon_min = 0.01  # Minimum epsilon
    #         self.epsilon_decay = 0.999 # Decay rate for epsilon
    #         self.num_episodes = 5000 # Number of episodes to train
    #         self.steps = 1000        # Maximum steps per episode
    #         self.layers = [128, 128, 64]
    #         self.replay_memory_size = 500000
    #         self.batch_size = 64
    #         self.update_target_estimator_every = 500

    # cfg = Config()

    # Create an instance of Transformer Dreamer
    model = TransDreamer()
        # maybe convert to use GRPO eventually
    # agent = DDPG(env, eval_env, options)
    # agent.actor_critic.load_state_dict(torch.load("actor_critic_5000_from_good_version.pth"))
    # agent.target_actor_critic.load_state_dict(torch.load("target_actor_5000_from_good_version.pth"))

    # rewards = []
    # smoothed_rewards = []  # To store smoothed rewards
    # smoothing_window = 200  # Window size for moving average

    # # Training loop
    # for episode in range(options.num_episodes):
    #     print(f"Episode {episode + 1} started.")
    #     # To complete an episode, have the following running:
    #         # GZ map/world
    #             # vehicle origin
    #             # world model, etc
    #         # composition of Nav2 autonomy stack
    #         # use the TransformerDreamer controller plug-in (insert direction into move task)
    #             # as TransformerDreamer is both a planner and controller
        
    #     # Send start and goal to orchistration node

    #     start = PoseStamped() 
    #     end = PoseStamped()

    #     rewards.append(agent.train_episode(start, end))
    #     # smoothed_rewards = compute_moving_average(rewards, smoothing_window)

    #     # Optionally, print progress
    #     if (episode + 1) % 10 == 0:
    #         torch.save(agent.actor_critic.state_dict(), "actor_critic_6000_from_good_version.pth")
    #         torch.save(agent.target_actor_critic.state_dict(), "target_actor_6000_from_good_version.pth")
        
    #     # update_plot(episode + 1, rewards, smoothed_rewards)
    #     if len(rewards) > smoothing_window and smoothed_rewards[-1] > 6000:
    #         break

    # env.close()



    model = model.to(device)

    optimizers = get_optimizer(cfg, model)
    checkpointer_path = os.path.join(cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id)
    checkpointer = Checkpointer(checkpointer_path, max_num=cfg.checkpoint.max_num)
    with open(checkpointer_path + '/config.yaml', 'w') as f:
        cfg.dump(stream=f, default_flow_style=False)
        print(f"config file saved to {checkpointer_path + '/config.yaml'}")

    if cfg.resume:
        checkpoint = checkpointer.load(cfg.resume_ckpt)

        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            for k, v in optimizers.items():
                if v is not None:
                    v.load_state_dict(checkpoint[k])
            env_step = checkpoint['env_step']
            global_step = checkpoint['global_step']

        else:
            env_step = 0
            global_step = 0

    else:
        env_step = 0
        global_step = 0

    writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id), flush_secs=30)

    datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'train_episodes')
    test_datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'test_episodes')
    train_env = make_env(cfg, writer, 'train', datadir, store=True)
    test_env = make_env(cfg, writer, 'test', test_datadir, store=True)

    # fill in length of 5000 frames
    train_env.reset()
    steps = count_steps(datadir, cfg)
    length = 0
    while steps < cfg.arch.prefill:
        action = train_env.sample_random_action()
        next_obs, reward, done = train_env.step(action[0])
        length += 1
        steps += done * length
        length = length * (1. - done)
        if done:
            train_env.reset()

    steps = count_steps(datadir, cfg)
    print(f'collected {steps} steps. Start training...')
    train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4)
    train_iter = iter(train_dl)
    global_step = max(global_step, steps)

    obs = train_env.reset()
    state = None
    action_list = torch.zeros(1, 1, cfg.env.action_size).float() # T, C
    action_list[0, 0, 0] = 1.
    input_type = cfg.arch.world_model.input_type
    temp = cfg.arch.world_model.temp_start

    while global_step < cfg.total_steps:

        with torch.no_grad():
            model.eval()
            next_obs, reward, done = train_env.step(action_list[0, -1].detach().cpu().numpy())
            prev_image = torch.tensor(obs[input_type])
            next_image = torch.tensor(next_obs[input_type])
            action_list, state = model.policy(prev_image.to(device), next_image.to(device), action_list.to(device),
                                                global_step, 0.1, state, context_len=cfg.train.batch_length)
            obs = next_obs
            if done:
                train_env.reset()
                state = None
                action_list = torch.zeros(1, 1, cfg.env.action_size).float()  # T, C
                action_list[0, 0, 0] = 1.

        if global_step % cfg.train.train_every == 0:

            temp = anneal_temp(global_step, cfg)

            model.train()

            traj = next(train_iter)
            for k, v in traj.items():
                traj[k] = v.to(device).float()

            logs = {}

            model_optimizer = optimizers['model_optimizer']
            model_optimizer.zero_grad()
            transformer_optimizer = optimizers['transformer_optimizer']
            if transformer_optimizer is not None:
                transformer_optimizer.zero_grad()

            # pass in A* complete path and world model curriculum learning level
            model_loss, model_logs, prior_state, post_state = model.world_model_loss(global_step, traj, temp)
            grad_norm_model = model.world_model.optimize_world_model(model_loss, model_optimizer, transformer_optimizer, writer, global_step)
            if cfg.arch.world_model.transformer.warm_up:
                lr = anneal_learning_rate(global_step, cfg)
                for param_group in transformer_optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = cfg.optimize.model_lr

            actor_optimizer = optimizers['actor_optimizer']
            value_optimizer = optimizers['value_optimizer']
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()

            # pass in A* path, agent curriculum level, and next control action
            actor_loss, value_loss, actor_value_logs = model.actor_and_value_loss(global_step, post_state, traj, temp)
            grad_norm_actor = model.optimize_actor(actor_loss, actor_optimizer, writer, global_step)
            grad_norm_value = model.optimize_value(value_loss, value_optimizer, writer, global_step)

            if global_step % cfg.train.log_every_step == 0:

                logs.update(model_logs)
                logs.update(actor_value_logs)
                model.write_logs(logs, traj, global_step, writer)

                writer.add_scalar('train_hp/lr', lr, global_step)

                grad_norm = dict(
                    grad_norm_model = grad_norm_model,
                    grad_norm_actor = grad_norm_actor,
                    grad_norm_value = grad_norm_value,
                )

                for k, v in grad_norm.items():
                    writer.add_scalar('train_grad_norm/' + k, v, global_step=global_step)

        # evaluate RL
        # every N trials, check the average success over the last T trials to move up curriculum
        if global_step % cfg.train.eval_every_step == 0:
            simulate_test(model, test_env, cfg, global_step, device)

        # only save success after a curriculum move and after a 200 (??) simulations
        if global_step % cfg.train.checkpoint_every_step == 0:
            env_step = count_steps(datadir, cfg)
            checkpointer.save('', model, optimizers, global_step, env_step)

        global_step += 1



if __name__ == "__main__":
    def get_config():
        parser = argparse.ArgumentParser(description='args for Seq_ROOTS project')

        parser.add_argument('--task', type=str, default='train',
                                help='which task to perfrom: train')
        parser.add_argument('--config-file', type=str, default='',
                                help='config file')
        parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                            help='using command line to modify configs.')

        args = parser.parse_args()

        if args.config_file:
            cfg.merge_from_file(args.config_file)

        if args.opts:
            cfg.merge_from_list(args.opts)

        if cfg.exp_name == '':
            if not args.config_file:
                raise ValueError('exp name can not be empty when config file is not provided')
            else:
                cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]

        task = {
            'train': main,
        }[args.task]

        return task, cfg

    task, cfg = get_config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(cfg, device, cfg.seed)
    # task(model, cfg, device)

    main()
