import unittest

from src.environment import GameEnvironment


class TestGameEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = GameEnvironment()

    def tearDown(self):
        self.env.close_game()

    def test_run(self):
        for _ in range(15):
            self.env.step(3)

    def test_run_walk(self):
        for _ in range(15):
            self.env.step(0)
            self.env.step(1)

    def test_walk_change_directions(self):
        for _ in range(5):
            self.env.step(1)
            self.env.step(2)

    def test_run_change_directions(self):
        for _ in range(5):
            self.env.step(0)
            self.env.step(3)

    def test_walk_drop(self):
        for _ in range(5):
            self.env.step(2)
            self.env.step(4)

    def test_run_drop(self):
        for _ in range(5):
            self.env.step(0)
            self.env.step(4)

    def test_pay(self):
        for _ in range(5):
            self.env.step(5)


if __name__ == "__main__":
    unittest.main()
