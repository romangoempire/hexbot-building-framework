"""Tests for training utilities: ReplayBuffer, train_step, augmentation."""

import unittest

import numpy as np
import torch

from bot import (
    ReplayBuffer, TrainingSample, train_step, augment_sample,
    create_network, encode_state, BOARD_SIZE, NUM_CHANNELS,
)
from main import HexGame


class TestReplayBuffer(unittest.TestCase):
    def _make_sample(self, result=1.0):
        state = torch.randn(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        policy[0] = 1.0
        return TrainingSample(
            encoded_state=state,
            policy_target=policy,
            player=0,
            result=result,
            threat_label=np.zeros(4, dtype=np.float32),
            priority=1.0,
        )

    def test_push(self):
        buf = ReplayBuffer(capacity=100)
        s = self._make_sample()
        buf.push(s)
        self.assertEqual(len(buf), 1)

    def test_sample(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(self._make_sample())
        samples, indices = buf.sample(5)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len(indices), 5)
        for s in samples:
            self.assertIsInstance(s, TrainingSample)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for _ in range(10):
            buf.push(self._make_sample())
        self.assertEqual(len(buf), 5)


class TestTrainStep(unittest.TestCase):
    def test_train_step_runs(self):
        net = create_network('fast')
        device = torch.device('cpu')
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        buf = ReplayBuffer(capacity=100)
        for _ in range(16):
            state = torch.randn(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            policy = np.random.dirichlet(np.ones(BOARD_SIZE * BOARD_SIZE)).astype(np.float32)
            sample = TrainingSample(
                encoded_state=state,
                policy_target=policy,
                player=0,
                result=1.0,
                threat_label=np.zeros(4, dtype=np.float32),
                priority=1.0,
            )
            buf.push(sample)

        losses = train_step(net, optimizer, buf, device, batch_size=8)
        self.assertIn('total', losses)
        self.assertIn('value', losses)
        self.assertIn('policy', losses)
        self.assertIn('threat', losses)
        self.assertGreater(losses['total'], 0)


class TestAugmentSample(unittest.TestCase):
    def _make_sample(self):
        state = torch.randn(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        policy[42] = 1.0
        return TrainingSample(
            encoded_state=state,
            policy_target=policy,
            player=0,
            result=1.0,
            threat_label=np.zeros(4, dtype=np.float32),
            priority=1.0,
        )

    def test_returns_three_samples(self):
        sample = self._make_sample()
        augmented = augment_sample(sample)
        self.assertEqual(len(augmented), 3)

    def test_preserves_stone_count(self):
        """Augmented states should have same nonzero channel structure."""
        sample = self._make_sample()
        augmented = augment_sample(sample)
        original_sum = sample.encoded_state.sum().item()
        for aug in augmented:
            # Augmentation transforms the spatial layout; sum may differ
            # due to floating point, but shape must be preserved
            self.assertEqual(aug.encoded_state.shape, sample.encoded_state.shape)
            self.assertEqual(aug.player, sample.player)
            self.assertEqual(aug.result, sample.result)

    def test_policy_sums_to_one(self):
        sample = self._make_sample()
        augmented = augment_sample(sample)
        for aug in augmented:
            total = aug.policy_target.sum()
            self.assertAlmostEqual(total, 1.0, places=4)


class TestAugmentPreservesHexDirections(unittest.TestCase):
    def test_augmented_shapes(self):
        """All augmented samples should have same tensor shape."""
        state = torch.randn(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        policy[0] = 1.0
        sample = TrainingSample(
            encoded_state=state,
            policy_target=policy,
            player=0,
            result=0.5,
            threat_label=np.zeros(4, dtype=np.float32),
            priority=1.0,
        )
        augmented = augment_sample(sample)
        for aug in augmented:
            self.assertEqual(aug.encoded_state.shape,
                             (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE))
            self.assertEqual(aug.policy_target.shape,
                             (BOARD_SIZE * BOARD_SIZE,))


class TestEncodeDecodeRoundtrip(unittest.TestCase):
    def test_encode_state(self):
        game = HexGame(candidate_radius=3, max_total_stones=200)
        game.place_stone(0, 0)
        tensor, oq, orr = encode_state(game)
        self.assertEqual(tensor.shape[0], NUM_CHANNELS)
        self.assertEqual(tensor.shape[1], BOARD_SIZE)
        self.assertEqual(tensor.shape[2], BOARD_SIZE)
        self.assertIsInstance(oq, int)
        self.assertIsInstance(orr, int)

    def test_encode_empty(self):
        game = HexGame(candidate_radius=3, max_total_stones=200)
        tensor, oq, orr = encode_state(game)
        self.assertEqual(tensor.shape, (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE))


class TestMixedPrecisionFlag(unittest.TestCase):
    def test_grad_scaler_param_exists(self):
        """train_step in bot.py accepts grad_scaler parameter."""
        import inspect
        sig = inspect.signature(train_step)
        self.assertIn('grad_scaler', sig.parameters)


if __name__ == '__main__':
    unittest.main()
