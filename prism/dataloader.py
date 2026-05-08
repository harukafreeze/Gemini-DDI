# TFRecord async dataloader
# prism/dataloader.py (V7.5 - Type Safe)
# ===================================================================
# FIXES:
# 1. Explicitly typed Padding Values to match Tensor dtypes (Int32 vs Float32).
# 2. Supports Smart Bucketing + Global Features.
# ===================================================================
import tensorflow as tf
import os

class PrismDdiDataloader:
    def __init__(self, config):
        self.config = config
        self.tfrec_dir = os.path.join(config.PROCESSED_DATA_PATH, 'tfrecords')
        print(f"--- Initializing Dataloader (V7.5 Type Safe) ---")
        if not os.path.exists(self.tfrec_dir):
            raise FileNotFoundError("TFRecords not found!")

    def _parse_function(self, example_proto):
        feature_description = {
            'atomic_features_a': tf.io.FixedLenFeature([], tf.string),
            'atomic_adj_a':      tf.io.FixedLenFeature([], tf.string),
            'motif_ids_a':       tf.io.FixedLenFeature([], tf.string),
            'motif_adj_a':       tf.io.FixedLenFeature([], tf.string),
            'global_features_a': tf.io.FixedLenFeature([], tf.string),
            'atomic_features_b': tf.io.FixedLenFeature([], tf.string),
            'atomic_adj_b':      tf.io.FixedLenFeature([], tf.string),
            'motif_ids_b':       tf.io.FixedLenFeature([], tf.string),
            'motif_adj_b':       tf.io.FixedLenFeature([], tf.string),
            'global_features_b': tf.io.FixedLenFeature([], tf.string),
            'label':             tf.io.FixedLenFeature([], tf.string),
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        def decode(b, dt): return tf.io.decode_raw(b, dt)

        # Drug A
        af_a = tf.reshape(decode(parsed['atomic_features_a'], tf.float32), [-1, self.config.ATOM_FEATURE_DIM])
        n_a = tf.shape(af_a)[0]
        adj_a = tf.reshape(decode(parsed['atomic_adj_a'], tf.int32), [n_a, n_a]) # INT32
        mi_a = decode(parsed['motif_ids_a'], tf.int32) # INT32
        n_ma = tf.shape(mi_a)[0]
        ma_a = tf.reshape(decode(parsed['motif_adj_a'], tf.float32), [n_ma, n_ma]) # FLOAT
        gf_a = decode(parsed['global_features_a'], tf.float32)

        # Drug B
        af_b = tf.reshape(decode(parsed['atomic_features_b'], tf.float32), [-1, self.config.ATOM_FEATURE_DIM])
        n_b = tf.shape(af_b)[0]
        adj_b = tf.reshape(decode(parsed['atomic_adj_b'], tf.int32), [n_b, n_b]) # INT32
        mi_b = decode(parsed['motif_ids_b'], tf.int32) # INT32
        n_mb = tf.shape(mi_b)[0]
        ma_b = tf.reshape(decode(parsed['motif_adj_b'], tf.float32), [n_mb, n_mb]) # FLOAT
        gf_b = decode(parsed['global_features_b'], tf.float32)

        label = tf.reshape(decode(parsed['label'], tf.float32), [self.config.NUM_CLASSES])

        inputs = {
            "atomic_features_a": af_a, "atomic_adj_a": adj_a,
            "motif_ids_a": mi_a, "motif_adj_a": ma_a, "global_features_a": gf_a,
            "atomic_features_b": af_b, "atomic_adj_b": adj_b,
            "motif_ids_b": mi_b, "motif_adj_b": ma_b, "global_features_b": gf_b
        }
        return inputs, label

    def get_dataset(self, mode='train'):
        file_path = os.path.join(self.tfrec_dir, f'{mode}.tfrec')
        dataset = tf.data.TFRecordDataset(file_path)
        
        if mode == 'train':
            dataset = dataset.shuffle(20000).repeat()
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        # Shapes
        # Note: Global Features use [None] to be safe with varying RDKit versions
        shapes = { 
            "atomic_features_a": [None, self.config.ATOM_FEATURE_DIM], 
            "atomic_adj_a": [None, None],
            "motif_ids_a": [None], "motif_adj_a": [None, None], 
            "global_features_a": [None],
            
            "atomic_features_b": [None, self.config.ATOM_FEATURE_DIM], 
            "atomic_adj_b": [None, None],
            "motif_ids_b": [None], "motif_adj_b": [None, None], 
            "global_features_b": [None]
        }
        
        # [CRITICAL FIX] Explicitly Typed Padding Values
        padding = {
            "atomic_features_a": 0.0, 
            "atomic_adj_a": tf.constant(0, dtype=tf.int32), # INT32 PADDING
            "motif_ids_a": tf.constant(0, dtype=tf.int32),  # INT32 PADDING
            "motif_adj_a": 0.0, 
            "global_features_a": 0.0,
            
            "atomic_features_b": 0.0, 
            "atomic_adj_b": tf.constant(0, dtype=tf.int32), # INT32 PADDING
            "motif_ids_b": tf.constant(0, dtype=tf.int32),  # INT32 PADDING
            "motif_adj_b": 0.0, 
            "global_features_b": 0.0
        }

        if mode == 'train':
            bucket_boundaries = [20, 40, 60, 80]
            bucket_batch_sizes = [self.config.BATCH_SIZE] * (len(bucket_boundaries) + 1)
            dataset = dataset.bucket_by_sequence_length(
                element_length_func=lambda x, y: tf.shape(x['atomic_features_a'])[0],
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padded_shapes=(shapes, [self.config.NUM_CLASSES]),
                padding_values=(padding, 0.0),
                drop_remainder=True
            )
        else:
            dataset = dataset.padded_batch(
                self.config.BATCH_SIZE, padded_shapes=(shapes, [self.config.NUM_CLASSES]),
                padding_values=(padding, 0.0), drop_remainder=False
            )
        return dataset.prefetch(tf.data.AUTOTUNE)