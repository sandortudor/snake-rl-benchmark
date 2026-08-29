import unittest

from snake_rl.agents import GreedyAgent, RandomAgent, TabularQAgent
from snake_rl.environment import SnakeEnv
from snake_rl.train import run_episode


class AgentSmokeTest(unittest.TestCase):
    def test_random_agent_finishes_episode(self):
        env = SnakeEnv(board_size=6, max_steps_without_food=20)
        result = run_episode(env, RandomAgent(), seed=1, training=False)
        self.assertIn("score", result)

    def test_greedy_agent_returns_valid_action(self):
        env = SnakeEnv(board_size=8)
        obs = env.reset(seed=1)
        action = GreedyAgent().act(env, obs)
        self.assertIn(action, [0, 1, 2])

    def test_tabular_q_learning_updates_table(self):
        env = SnakeEnv(board_size=6, max_steps_without_food=20)
        agent = TabularQAgent()
        run_episode(env, agent, seed=1, training=True)
        self.assertGreater(agent.model_size(), 0)


if __name__ == "__main__":
    unittest.main()

