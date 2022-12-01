import numpy as np
import pandas as pd
from gym import spaces

from sklearn.preprocessing import MinMaxScaler


class CryptoEnvNormalizer:

    def __init__(self, stock_dimension, indicators, state_space, df, root_path='.'):
        self.stock_dim = stock_dimension
        self.indicators = indicators
        self.state_space = state_space
        self.root_path = root_path
        self.config_path = f"{self.root_path}/datasets/thesis/feature_normalization"
        self.df = df

        # used for close, sma7, sma30, bollub, bolllb
        self.price_scaler = MinMaxScaler(clip=True)
        price_max_norm = pd.read_csv(f"{self.config_path}/price_norm_clean.csv", index_col=0)
        self.price_scaler.fit(price_max_norm)
        # used for dx, rsi
        self.normal_minmax = MinMaxScaler(clip=True)
        self.normal_minmax.fit(pd.DataFrame([np.array([100] * 10), np.array([0] * 10)], columns=price_max_norm.columns))
        # used for cci
        self.cci_scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
        self.cci_scaler.fit(pd.read_csv(f"{self.config_path}/cci_limits.csv", index_col=0))
        # used for macd
        self.macd_scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
        self.macd_scaler.fit(pd.read_csv(f"{self.config_path}/macd_limits.csv", index_col=0))

        self.scalers = {
            "close": self.price_scaler,
            "macd": self.macd_scaler,
            "boll_ub": self.price_scaler,
            "boll_lb": self.price_scaler,
            "rsi_30": self.normal_minmax,
            "cci_30": self.cci_scaler,
            "dx_30": self.normal_minmax,
            "close_7_sma": self.price_scaler,
            "close_30_sma": self.price_scaler
        }
        self.transformed_data = self._get_normalizations()

    def get_observation_space(self):
        lower_bounds = self._get_lower_bounds()
        upper_bounds = self._get_upper_bounds()
        return spaces.Box(low=lower_bounds, high=upper_bounds,
                          shape=(self.state_space,), dtype=np.float64)

    def _get_lower_bounds(self):
        balance_lower = np.array([0])
        price_lower = np.array([0] * self.stock_dim)
        asset_balance_lower = np.array([0] * self.stock_dim)
        lower_list = [balance_lower, price_lower, asset_balance_lower]

        indicators_zero_bounded = ["boll_ub", "boll_lb", "rsi_30", "dx_30", "close_7_sma", "close_30_sma"]
        indicators_minus_one_bounded = ["macd", "cci_30"]

        for indicator in self.indicators:
            if indicator in indicators_zero_bounded:
                lower_list.append(np.array([0] * self.stock_dim))
            elif indicator in indicators_minus_one_bounded:
                lower_list.append(np.array([-1] * self.stock_dim))
            else:
                raise ValueError(f"Unsupported indicator {indicator}")

        observation_space_lower = np.concatenate(lower_list)

        if len(observation_space_lower) != self.state_space:
            raise ValueError(
                f"LowerObsSpace Bounds expected to be shape {self.state_space} but is {len(observation_space_lower)}")
        return observation_space_lower

    def _get_upper_bounds(self):
        balance_upper = np.array([np.inf])
        price_upper = np.array([1] * self.stock_dim)
        asset_balance_upper = np.array([np.inf] * self.stock_dim)

        upper_list = [balance_upper, price_upper, asset_balance_upper]

        normal_bounded_indicators = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_7_sma",
                                     "close_30_sma"]
        for indicator in self.indicators:
            if indicator in normal_bounded_indicators:
                upper_list.append(np.array([1] * self.stock_dim))
            else:
                raise ValueError(f"Unsupported indicator {indicator}")

        observation_space_upper = np.concatenate(upper_list)

        if len(observation_space_upper) != self.state_space:
            raise ValueError(
                f"UpperObsSpace Bounds expected to be shape {self.state_space} but is {len(observation_space_upper)}")
        return observation_space_upper

    def _get_normalizations(self):
        transformed_data = {}
        # normalize close price + all indicators
        norm_elements = ["close"] + self.indicators

        for item in norm_elements:
            data = self.df.pivot(index="date", columns="tic", values=item)
            transformed = self.scalers[item].transform(data)
            data_normalized = pd.DataFrame(transformed, columns=data.columns)
            transformed_data[item] = data_normalized

        return transformed_data

    def get_normalized_state(self, day, original_state):
        balance = original_state[0]
        asset_amounts = original_state[1 + self.stock_dim:1 + (self.stock_dim * 2)]
        close_t = self.transformed_data['close'].loc[day].tolist()
        transformed_indicators = sum((self.transformed_data[tech].loc[day].tolist() for tech in self.indicators), [])
        normalized_state = ([balance] + close_t + asset_amounts + transformed_indicators)
        return normalized_state

