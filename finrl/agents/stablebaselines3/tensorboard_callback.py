from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            env = self.locals['env'].envs[0]
            day = env.day
            max_days = env.max_days - 1

            if day >= max_days:
                stock_dim = env.stock_dim
                asset_memory = env.asset_memory
                start_money = asset_memory[0]
                last_prices = np.array(env.state[1:1 + stock_dim])
                last_amounts = np.array(env.state[(stock_dim + 1): (stock_dim * 2 + 1)])
                end_money = env.state[0] + sum(last_prices * last_amounts)
                total_reward = end_money - start_money

                sharpe, sortino = env.get_sharpe_sortino()

                self.logger.record(key="my-stats/episode", value=env.episode)
                self.logger.record(key="my-stats/episode-reward", value=total_reward)
                self.logger.record(key="my-stats/trades", value=env.trades)
                self.logger.record(key="my-stats/sharpe", value=sharpe)
                self.logger.record(key="my-stats/sortino", value=sortino)
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True
