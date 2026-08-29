import unittest

from snake_rl.environment import SnakeEnv
from snake_rl.models import build_model, parameter_count
from snake_rl.visualize import grid_from_env


class ArchitectureContractTest(unittest.TestCase):
    def test_documented_parameter_counts(self):
        expected = {
            "dqn": 18_563,
            "double_dqn": 18_563,
            "dueling_dqn": 35_204,
            "lightweight_snake_ddqn": 9_412,
        }
        for model_name, count in expected.items():
            with self.subTest(model=model_name):
                model = build_model(model_name, input_size=12, action_count=3)
                self.assertEqual(parameter_count(model), count)

    def test_neural_models_return_three_q_values(self):
        import torch

        inputs = torch.zeros(2, 12)
        for model_name in (
            "dqn",
            "double_dqn",
            "dueling_dqn",
            "lightweight_snake_ddqn",
        ):
            with self.subTest(model=model_name):
                output = build_model(model_name, 12, 3)(inputs)
                self.assertEqual(tuple(output.shape), (2, 3))


class VisualizationContractTest(unittest.TestCase):
    def test_grid_contains_head_body_and_food(self):
        env = SnakeEnv(board_size=10)
        env.reset(seed=7)
        grid = grid_from_env(env)

        self.assertEqual(grid.shape, (10, 10))
        self.assertEqual(int((grid == 2).sum()), 1)
        self.assertEqual(int((grid == 3).sum()), 1)
        self.assertEqual(int((grid == 1).sum()), len(env.snake) - 1)


if __name__ == "__main__":
    unittest.main()
