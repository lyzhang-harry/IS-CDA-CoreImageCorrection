import numpy as np
import torch


class TorchPermutationPrefix:
    def __init__(self):
        # CPU torch RNG state has a 624-word MT19937 key, a next index and a flag.
        state = np.frombuffer(torch.get_rng_state().numpy(), dtype="<u8")
        if state.size != 632 or state[2] > 624:
            raise RuntimeError("Unrecognized CPU RNG state. Use torch.randperm.")
        self.generator = np.random.MT19937()
        self.generator.state = {
            "bit_generator": "MT19937",
            "state": {"key": state[3:627].astype(np.uint32), "pos": int(state[2]) or 624},
        }

    def prefix(self, n, k):
        if n >= 2**32 // 20:
            raise ValueError("Population exceeds the verified forward-shuffle range.")
        k = min(n, k)
        draws = min(k, n - 1)
        values = self.generator.random_raw(draws) % np.arange(n, n - draws, -1, dtype=np.uint64)
        # The first k positions cannot change in any of the remaining swaps.
        # Consume those random numbers to align both domains AND later iterations.
        self.generator.random_raw(n - 1 - draws, output=False)
        result = np.empty(k, dtype=np.int64)
        mapping = {}
        for i, offset in enumerate(values.tolist()):
            j = i + offset
            result[i] = mapping.get(j, j)
            mapping[j] = mapping.pop(i, i)
        if draws < k:
            result[-1] = mapping.get(k - 1, k - 1)
        return torch.from_numpy(result)
