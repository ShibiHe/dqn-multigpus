__author__ = 'frankhe'
import numpy as np


def dot_with_mod(a, b, mod):
    it = np.nditer([a, b], op_flags=[['readonly'], ['readonly']])
    sum = 0
    for (a, b) in it:
        sum = (sum + (a % mod) * (b % mod)) % mod
    return sum


class SimHash(object):
    def __init__(self,item_dim, dim_key=128, bucket_sizes=None):
        self.item_dim = item_dim
        self.dim_key = dim_key
        if bucket_sizes is None:
            bucket_sizes = [999979, 999979]
        self.projection_matrix = np.random.normal(size=(len(bucket_sizes), item_dim, dim_key))
        mods_list = []
        for bucket_size in bucket_sizes:
            mod, mods = 1, []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod << 1) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))

    def compute_keys(self, items):
        keys = []
        for i in xrange(len(self.bucket_sizes)):
            binaries = np.sign(np.asarray(items).dot(self.projection_matrix[i, ...])).astype(np.int32)
            key = dot_with_mod(binaries, self.mods_list[:, i], self.bucket_sizes[i])
            keys.append(key)
        return keys

    def inc_keys(self, keys, update_delta = 1):
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], update_delta)

    def query_keys(self, keys):
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        counts = np.asarray(all_counts).min(axis=0)
        return counts

    def reset(self):
        self.tables = np.zeros(
            (len(self.bucket_sizes), np.max(self.bucket_sizes))
        )

if __name__=="__main__":
    random_item = np.random.randint(0, 256, size=(32, 84*84))
    simhash = SimHash(item_dim = 84*84, dim_key = 128)
    keys = simhash.compute_keys(random_item)
    import pprint
    pprint.pprint(keys)

    
