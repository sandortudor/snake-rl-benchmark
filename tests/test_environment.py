import unittest

from snake_rl.environment import SnakeEnv


class SnakeEnvTest(unittest.TestCase):
    def test_reset_is_deterministic(self):
        env = SnakeEnv(board_size=8)
        first = env.reset(seed=123).tolist()
        first_food = env.food
        second = env.reset(seed=123).tolist()
        self.assertEqual(first, second)
        self.assertEqual(first_food, env.food)

    def test_snake_eats_food_and_grows(self):
        env = SnakeEnv(board_size=8)
        env.reset(seed=1)
        head_x, head_y = env.snake[0]
        env.food = (head_x + 1, head_y)
        start_len = len(env.snake)
        result = env.step(1)
        self.assertFalse(result.done)
        self.assertEqual(env.score, 1)
        self.assertEqual(len(env.snake), start_len + 1)

    def test_wall_collision_ends_episode(self):
        env = SnakeEnv(board_size=5)
        env.reset(seed=1)
        done = False
        for _ in range(5):
            result = env.step(1)
            done = result.done
            if done:
                break
        self.assertTrue(done)
        self.assertLess(result.reward, 0)


if __name__ == "__main__":
    unittest.main()

