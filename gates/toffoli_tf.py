import tensorflow as tf
import numpy as np
from states import RandomState
from toffoli_matrices_tf import (
    U,
    toff_a,
    toff_b,
    toff_c,
    toff_d,
    toff_e,
    toff_f,
    toff_g,
    toff_h,
    toff_i,
)

n_epochs = 100
epoch_samples = 200
test_samples = 100

jzz_12 = tf.Variable([[1]], dtype=tf.float64)
jzz_23 = tf.Variable([[1]], dtype=tf.float64)
jzz_13 = tf.Variable([[1]], dtype=tf.float64)
jyy_12 = tf.Variable([[1]], dtype=tf.float64)
jxx_23 = tf.Variable([[1]], dtype=tf.float64)
jxx_13 = tf.Variable([[1]], dtype=tf.float64)
hz_1 = tf.Variable([[1]], dtype=tf.float64)
hz_2 = tf.Variable([[1]], dtype=tf.float64)
hx_3 = tf.Variable([[1]], dtype=tf.float64)

params = [jzz_12, jzz_23, jzz_13, jyy_12, jxx_23, jxx_13, hz_1, hz_2, hx_3]

optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

test_states = [RandomState(3, False).ket for _ in range(test_samples)]


def calculate_fidelity(chain_U, batch_states):
    fidelities = tf.TensorArray(
        tf.float64, size=0, dynamic_size=True, clear_after_read=False
    )
    for i, state in enumerate(batch_states):
        state_tf = tf.convert_to_tensor(state, dtype=tf.complex128)
        chain_state = tf.linalg.matvec(chain_U, state_tf)
        toffoli_state = tf.linalg.matvec(U, state_tf)
        overlap = tf.tensordot(toffoli_state, chain_state, 1)
        fi = tf.multiply(tf.math.conj(overlap), overlap)
        fidel = tf.math.abs(fi)
        fidelities.write(i, fidel)
    fidelities_ = fidelities.stack()
    fidelity = -1 * tf.math.reduce_mean(fidelities_)
    return fidelity


def train_step(test_states, batch_states):
    with tf.GradientTape() as tape:
        a = tf.complex(jzz_12, tf.zeros([1], dtype=tf.float64))
        b = tf.complex(jzz_23, tf.zeros([1], dtype=tf.float64))
        c = tf.complex(jzz_13, tf.zeros([1], dtype=tf.float64))
        d = tf.complex(jyy_12, tf.zeros([1], dtype=tf.float64))
        e = tf.complex(jxx_23, tf.zeros([1], dtype=tf.float64))
        f = tf.complex(jxx_13, tf.zeros([1], dtype=tf.float64))
        g = tf.complex(hz_1, tf.zeros([1], dtype=tf.float64))
        h = tf.complex(hz_2, tf.zeros([1], dtype=tf.float64))
        i = tf.complex(hx_3, tf.zeros([1], dtype=tf.float64))
        toffoli_3 = -1j * (
            tf.math.multiply(a, toff_a)
            + tf.math.multiply(b, toff_b)
            + tf.math.multiply(c, toff_c)
            + tf.math.multiply(d, toff_d)
            + tf.math.multiply(e, toff_e)
            + tf.math.multiply(f, toff_f)
            + tf.math.multiply(g, toff_g)
            + tf.math.multiply(h, toff_h)
            + tf.math.multiply(i, toff_i)
        )
        toffoli_U = tf.linalg.expm(toffoli_3)
        fidelity = calculate_fidelity(toffoli_U, batch_states)
    grads = tape.gradient(fidelity, params)
    print(f"fidelity = {fidelity}")
    optimizer.apply_gradients(zip(grads, params))


def train(epochs):
    for epoch in range(epochs):
        epoch_states = [RandomState(3, False).ket for _ in range(epoch_samples)]
        batch_size = 100
        batches = np.split(np.array(epoch_states), epoch_samples // batch_size)
        for batch_states in batches:
            train_step(test_states, batch_states)
        print("Epoch {} finished".format(epoch))


train(n_epochs)
