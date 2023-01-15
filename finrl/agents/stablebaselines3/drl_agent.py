# DRL models from Stable Baselines 3
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.agents.stablebaselines3.models import MODELS, MODEL_KWARGS, NOISE
from finrl.agents.stablebaselines3.tensorboard_callback import TensorboardCallback
from finrl.meta.env_custom.custom_eval_callback import CustomEvalCallback


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(self, model_name, policy="MlpPolicy", policy_kwargs=None, model_kwargs=None, verbose=1, seed=None,
                  tensorboard_log=None, ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    def train_model(self, model, tb_log_name, total_timesteps=5000,
                    eval_during_train=False, eval_env=None, episode_size=379,
                    save_checkpoints=True):
        callbacks = [TensorboardCallback()]

        if save_checkpoints:
            checkpoint_callback = CheckpointCallback(
                save_freq=200_000,
                save_path='./trained_models/model_checkpoints/',
                name_prefix=tb_log_name,
                verbose=1
            )
            callbacks.append(checkpoint_callback)

        if eval_during_train:
            if not is_wrapped(eval_env, Monitor):
                eval_env = Monitor(eval_env)
            eval_env = DummyVecEnv([lambda: eval_env])

            eval_callback = CustomEvalCallback(
                eval_env,
                best_model_save_path=f'eval_log/{tb_log_name}',
                log_path=f'eval_log/{tb_log_name}',
                eval_freq=episode_size * 100,
                n_eval_episodes=1,
                deterministic=True
            )
            callbacks.append(eval_callback)

        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=callbacks)
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        try:
            test_env, test_obs = environment.get_sb_env()
        except ValueError:
            test_env, test_obs = environment.get_sb_env()

        """make a prediction"""
        account_memory = []
        actions_memory = []
        #         state_memory=[] #add memory pool to store states
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                state_memory = test_env.env_method(method_name="save_state_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print(f"done in step {i}!")
                break
        return state_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = (
                    environment.amount
                    + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets
