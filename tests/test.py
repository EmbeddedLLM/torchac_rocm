import unittest
import torch
import torchac_cuda  # Your CUDA extension module

class TestTorchacCuda(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_encode_fast_new(self):
        cdf = torch.randint(0, 32768, (2, 4, 16), dtype=torch.int16, device=self.device)
        symbols = torch.randint(0, 14, (2, 100, 4), dtype=torch.int8, device=self.device)  # Changed to int8

        output_buffer = torch.zeros((2, 4, 200), dtype=torch.uint8, device=self.device)
        output_lengths = torch.zeros((2, 4), dtype=torch.int32, device=self.device)
        torchac_cuda.encode_fast_new(cdf, symbols, output_buffer, output_lengths)
        
        self.assertGreater(output_lengths.sum().item(), 0)

    def test_decode_fast_new(self):
        cdf = torch.randint(0, 32768, (2, 4, 16), dtype=torch.int16, device=self.device)
        encoded = torch.randint(0, 256, (2, 4, 200), dtype=torch.uint8, device=self.device)
        lengths = torch.randint(1, 201, (2, 4), dtype=torch.int32, device=self.device)

        output = torch.zeros((2, 100, 4), dtype=torch.uint8, device=self.device)
        torchac_cuda.decode_fast_new(cdf, encoded, lengths, output)

        self.assertFalse(torch.all(output == 0))

    def test_decode_fast_prefsum(self):
        cdf = torch.randint(0, 32768, (2, 4, 16), dtype=torch.int16, device=self.device)
        encoded = torch.randint(0, 256, (800,), dtype=torch.uint8, device=self.device)
        lengths_prefsum = torch.cumsum(torch.randint(1, 201, (2, 4), dtype=torch.int64, device=self.device), dim=1)

        output = torch.zeros((2, 100, 4), dtype=torch.uint8, device=self.device)
        torchac_cuda.decode_fast_prefsum(cdf, encoded, lengths_prefsum, output)

        self.assertFalse(torch.all(output == 0))

    def test_calculate_cdf(self):
        input_tensor = torch.randint(0, 16, (2, 100, 4), dtype=torch.int8, device=self.device)
        max_bins = 16

        cdf = torchac_cuda.calculate_cdf(input_tensor, max_bins)

        self.assertEqual(cdf.shape, (2, 4, max_bins + 1))
        self.assertTrue(torch.all(cdf[:, :, -1] == 65535))  # Last CDF value should be 2^16 - 1
        self.assertTrue(torch.all(torch.diff(cdf, dim=2) >= 0))  # CDF should be monotonically increasing

if __name__ == '__main__':
    unittest.main()