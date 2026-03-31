"""Tests for neural network creation and forward/backward passes."""

import unittest

import torch

from bot import create_network, HexNet, BOARD_SIZE, NUM_CHANNELS


class TestAllArchitecturesForward(unittest.TestCase):
    """Test that each config produces correct output shapes."""

    def _check_forward(self, config):
        net = create_network(config)
        net.eval()
        batch = torch.randn(2, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        with torch.no_grad():
            policy, value, threat = net(batch)
        bs2 = BOARD_SIZE * BOARD_SIZE
        self.assertEqual(policy.shape, (2, bs2), f"{config} policy shape")
        self.assertEqual(value.shape, (2, 1), f"{config} value shape")
        self.assertEqual(threat.shape, (2, 4), f"{config} threat shape")

    def test_fast(self):
        self._check_forward('fast')

    def test_standard(self):
        self._check_forward('standard')

    def test_large(self):
        self._check_forward('large')

    def _try_optional_config(self, config):
        """Try an optional config that may have missing dependencies."""
        try:
            self._check_forward(config)
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"{config} dependencies not available: {e}")

    def test_orca_transformer(self):
        self._try_optional_config('orca-transformer')

    def test_hex_gnn(self):
        self._try_optional_config('hex-gnn')

    def test_multiscale(self):
        self._try_optional_config('multiscale')


class TestForwardPV(unittest.TestCase):
    def test_forward_pv(self):
        net = create_network('fast')
        net.eval()
        batch = torch.randn(2, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        with torch.no_grad():
            policy, value = net.forward_pv(batch)
        bs2 = BOARD_SIZE * BOARD_SIZE
        self.assertEqual(policy.shape, (2, bs2))
        self.assertEqual(value.shape, (2, 1))


class TestBackwardPass(unittest.TestCase):
    def test_backward_doesnt_crash(self):
        net = create_network('fast')
        net.train()
        batch = torch.randn(2, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        policy, value, threat = net(batch)
        loss = policy.sum() + value.sum() + threat.sum()
        loss.backward()
        # Check gradients exist
        for p in net.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                break


class TestCreateNetworkInvalid(unittest.TestCase):
    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            create_network('nonexistent-config-xyz')


class TestNetworkParamsCount(unittest.TestCase):
    def test_standard_params(self):
        net = create_network('standard')
        count = sum(p.numel() for p in net.parameters())
        # Standard: 128 filters, 12 blocks -> roughly 3.9M params
        self.assertGreater(count, 3_000_000)
        self.assertLess(count, 5_000_000)

    def test_fast_params(self):
        net = create_network('fast')
        count = sum(p.numel() for p in net.parameters())
        # Fast: 64 filters, 4 blocks -> roughly 500K params
        self.assertGreater(count, 200_000)
        self.assertLess(count, 1_500_000)

    def test_large_params(self):
        net = create_network('large')
        count = sum(p.numel() for p in net.parameters())
        # Large: 256 filters, 12 blocks -> roughly 15M params
        self.assertGreater(count, 10_000_000)
        self.assertLess(count, 25_000_000)


class TestHexNetPredict(unittest.TestCase):
    def test_predict_method(self):
        net = create_network('fast')
        net.eval()
        state = torch.randn(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        policy_logits, value = net.predict(state)
        bs2 = BOARD_SIZE * BOARD_SIZE
        self.assertEqual(policy_logits.shape, (bs2,))
        self.assertIsInstance(value, float)


if __name__ == '__main__':
    unittest.main()
