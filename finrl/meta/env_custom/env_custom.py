from __future__ import annotations

import math
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.meta.env_custom.crypto_env_normalizer import CryptoEnvNormalizer

matplotlib.use("Agg")


class CustomTradingEnv(gym.Env):
    """A custom trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, stock_dim: int, hmax: int, initial_amount: int, num_stock_shares: list[float],
                 buy_cost_pct: list[float], sell_cost_pct: list[float], reward_scaling: float, state_space: int,
                 action_space: int, tech_indicator_list: list[str], turbulence_threshold=None,
                 risk_indicator_col="turbulence", make_plots: bool = False, print_verbosity=200, day=0,
                 initial=True, previous_state=[], model_name="", mode="", iteration="", root_dir='.', seed=None,
                 strategy_name="crypto_single", run_name="run1", random_initial=False):
        self.day = day
        self.df = df
        self.max_days = len(self.df.index.unique())
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares  # initial amount of shares
        self.initial_amount = initial_amount  # get the initial cash
        self.random_initial_amount = [0] * (1 + self.stock_dim)
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.env_normalizer = CryptoEnvNormalizer(stock_dim, tech_indicator_list, state_space, df)
        self.observation_space = self.env_normalizer.get_observation_space()
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.random_initial = random_initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()
        self.root_dir = root_dir
        self.strategy_name = strategy_name
        self.run_name = run_name
        self.main_path = f"{self.root_dir}/results/{self.strategy_name}"
        self.run_path = f"{self.main_path}/{self.model_name}/{self.run_name}"

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.missed_trades = 0
        self.episode = 0

        # memorize all the total balance change
        # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.asset_memory_buffer_size = len(self.df.date.unique())
        self.empty_asset_buffer = [0] * self.asset_memory_buffer_size
        self.asset_memory = self.empty_asset_buffer
        self.asset_memory[0] = self.initial_amount + np.sum(np.array(self.num_stock_shares)
                                                            * np.array(self.state[1: 1 + self.stock_dim]))
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = ([])
        self.date_memory = self.df.date.unique().tolist()
        self.stock_names = self.df.tic.unique().tolist()
        self.state_memory_names = [
            ["cash"] + [x + "_price" for x in self.stock_names] + [x + "_amount" for x in self.stock_names]]

        self._seed(seed)

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # check if the stock is able to sell
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day,
                # the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if we own > 0 of the current asset
                    sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                    sell_amount = (self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index]))
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index])
                    if sell_num_shares > 0:
                        self.trades += 1
                    else:
                        self.missed_trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                                self.state[index + 1]
                                * sell_num_shares
                                * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                                self.state[index + 1]
                                * sell_num_shares
                                * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:
                # check if the stock is able to buy
                # Buy only if the price is > 0 (no missing data in this particular date)
                # when buying stocks, we should consider the cost of trading when calculating available_amount,
                # or we maybe have cash<0
                available_amount = self.state[0] / (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index]))
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index])
                if buy_num_shares > 0:
                    self.trades += 1
                else:
                    self.missed_trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        filename = f"{self.run_path}/account_value_{self.mode}_{self.episode}.png"
        plt.savefig(filename)

        # with open(f"{self.root_dir}/results/account_value_episodes.csv", 'a') as fd:
        #     writer = csv.writer(fd)
        #     writer.writerow(self.asset_memory)
        plt.close()

    def get_sharpe_sortino(self):
        sharpe = 0
        sortino = 0

        df_total_value = pd.DataFrame(self.asset_memory)
        df_total_value.columns = ["account_value"]
        daily_return = df_total_value["account_value"].pct_change(1)

        # sharpe
        if daily_return.std() != 0:
            # TODO: improve for intraday trading
            sharpe = ((365 ** 0.5) * daily_return.mean() / daily_return.std())

        # sortino
        temp_expectation = np.mean(np.minimum(0, daily_return) ** 2)
        downside_dev = np.sqrt(temp_expectation)
        sortino = np.mean(daily_return) / downside_dev

        return sharpe, sortino

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            last_prices = np.array(self.state[1: (self.stock_dim + 1)])
            last_amounts = np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            end_total_asset = self.state[0] + sum(last_prices * last_amounts)

            tot_reward = end_total_asset - self.asset_memory[0]

            sharpe, sortino = self.get_sharpe_sortino()

            stats = {
                'model_name': self.model_name,
                'run': self.run_name,
                'mode': self.mode,
                'episode': self.episode,
                'end_total_asset': end_total_asset,
                'total_reward': tot_reward,
                'total_cost': self.cost,
                'trades': self.trades,
                'sharpe': sharpe,
                'sortino': sortino,
                'missed_trades': self.missed_trades
            }

            if not self.episode % self.print_verbosity:
                if self.make_plots:
                    self._make_plot()

                stats_df = pd.DataFrame([stats])
                stats_df.to_csv(f"{self.main_path}/episode_stats.csv", header=False, index=False, mode='a')

                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print(f"missed_trades: {self.missed_trades}")
                print(f"Sharpe: {sharpe:0.3f}")
                print(f"Sortino: {sortino:0.3f}")
                print("=================================")

                if (self.model_name != "") and (self.mode != ""):
                    df_actions = self.save_action_memory()
                    filename_prefix = f"{self.run_name}_ep{self.episode:05.0f}"
                    df_actions.to_csv(f"{self.run_path}/{filename_prefix}_actions.csv")

                    state_memory = pd.DataFrame(self.state_memory)
                    state_memory.index = self.date_memory[:-1]
                    if self.random_initial:
                        init_state = pd.DataFrame([[self.random_initial_amount[0]] + self.df.loc[
                            0].close.tolist() + self.random_initial_amount[1:self.stock_dim + 1]])
                    else:
                        init_state = pd.DataFrame(
                            [[self.initial_amount] + self.df.loc[0].close.tolist() + self.num_stock_shares])
                    state_memory = pd.concat([init_state, state_memory])
                    state_memory.columns = self.state_memory_names
                    state_memory['account_value'] = self.asset_memory
                    state_memory['reward'] = [0] + self.rewards_memory
                    state_memory.to_csv(f"{self.run_path}/{filename_prefix}_state.csv")
        else:
            actions_unscaled = actions
            actions = actions * self._get_action_normalizer()

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions_unscaled)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            self.reward = end_total_asset - begin_total_asset
            self.asset_memory[self.day] = end_total_asset
            if not self.episode % self.print_verbosity:
                self.actions_memory.append(np.concatenate([actions, actions_unscaled]))
                self.rewards_memory.append(self.reward)
                self.state_memory.append(self.state[0:2 * self.stock_dim + 1])
            self.reward = self.reward * self.reward_scaling

        normalized_state = self.env_normalizer.get_normalized_state(self.day, self.state)
        return normalized_state, self.reward, self.terminal, {}

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]  # has to be reset before _initiate_state()

        # initiate state
        self.state = self._initiate_state()

        asset_prices = np.array(self.state[1: 1 + self.stock_dim])
        self.asset_memory = self.empty_asset_buffer
        if self.initial:
            self.asset_memory[0] = self.state[0] + np.sum(
                np.array(self.state[1 + self.stock_dim:1 + 2 * self.stock_dim]) * asset_prices)
        else:
            previous_asset_amounts = np.array(self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            previous_total_asset = self.previous_state[0] + sum(asset_prices * previous_asset_amounts)
            self.asset_memory[0] = previous_total_asset

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.missed_trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []

        self.episode += 1

        normalized_state = self.env_normalizer.get_normalized_state(self.day, self.state)
        return normalized_state

    def render(self, mode="human", close=False):
        return self.state

    def get_random_start_values(self):
        probs = np.random.rand(1 + self.stock_dim)
        money_bal = probs[0] * self.initial_amount
        money_asset = self.initial_amount - money_bal
        probs = np.delete(probs, 0)
        ratio = probs / probs.sum()
        money = (ratio * money_asset)
        amount = money / self.data.close.tolist()
        return [money_bal] + amount.tolist()

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                if self.random_initial:
                    self.random_initial_amount = self.get_random_start_values()
                    state = (
                            [self.random_initial_amount[0]]
                            + self.data.close.values.tolist()
                            + self.random_initial_amount[1:self.stock_dim + 1]
                            + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
                    )
                    # print(state[0], state[11:21])
                else:
                    state = (
                            [self.initial_amount]
                            + self.data.close.values.tolist()
                            + self.num_stock_shares
                            + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
                    )  # append initial stocks_share to initial state, instead of all zero

            else:
                # for single stock
                state = (
                        [self.initial_amount] + [self.data.close] + [0] * self.stock_dim
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.previous_state[0]]
                        + self.data.close.values.tolist()
                        + self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                        + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
                )
            else:
                # for single stock
                state = (
                        [self.previous_state[0]] + [self.data.close]
                        + self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )

        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                    [self.state[0]] + self.data.close.values.tolist()
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
            )
        else:
            # for single stock
            state = (
                    [self.state[0]] + [self.data.close]
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )
        return state

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = np.concatenate(
                [self.data.tic.values, [x + "_u" for x in self.data.tic.values]])  # self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def _get_action_normalizer(self):
        return (100_000 / self.data.close).tolist()  # maximum amount of $ each action can hold