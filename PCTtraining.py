import os
import h5py
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Lambda, Reshape


# ============================================================================
# 0. GPU setup
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config failed: {e}")


# ============================================================================
# 1. Load base images
# ============================================================================
def load_image(file_path):
    return np.array(Image.open(file_path).convert("L"), dtype=np.float32)


up_images_path = "/root/autodl-tmp/phy/up_images"
low_images_path = "/root/autodl-tmp/phy/low_images"

m11_up = load_image(os.path.join(up_images_path, "m11_up.png"))
m12_up = load_image(os.path.join(up_images_path, "m12_up.png"))
m13_up = load_image(os.path.join(up_images_path, "m13_up.png"))
m22_up = load_image(os.path.join(up_images_path, "m22_up.png"))
m23_up = load_image(os.path.join(up_images_path, "m23_up.png"))
m33_up = load_image(os.path.join(up_images_path, "m33_up.png"))

m11_low = load_image(os.path.join(low_images_path, "m11_low.png"))
m12_low = load_image(os.path.join(low_images_path, "m12_low.png"))
m13_low = load_image(os.path.join(low_images_path, "m13_low.png"))
m22_low = load_image(os.path.join(low_images_path, "m22_low.png"))
m23_low = load_image(os.path.join(low_images_path, "m23_low.png"))
m33_low = load_image(os.path.join(low_images_path, "m33_low.png"))


def resize_base_images(img, target_h=14, target_w=28):
    pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
    resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(resized, dtype=np.float32)


m11_up = resize_base_images(m11_up)
m12_up = resize_base_images(m12_up)
m13_up = resize_base_images(m13_up)
m22_up = resize_base_images(m22_up)
m23_up = resize_base_images(m23_up)
m33_up = resize_base_images(m33_up)

m11_low = resize_base_images(m11_low)
m12_low = resize_base_images(m12_low)
m13_low = resize_base_images(m13_low)
m22_low = resize_base_images(m22_low)
m23_low = resize_base_images(m23_low)
m33_low = resize_base_images(m33_low)


# ============================================================================
# 2. Normalize and convert to TensorFlow constants
# ============================================================================
def normalize_image(image):
    """Per-sample min-max normalization, supports batch dimension"""
    min_val = tf.reduce_min(image, axis=[1, 2], keepdims=True)
    max_val = tf.reduce_max(image, axis=[1, 2], keepdims=True)
    return (image - min_val) / (max_val - min_val + 1e-8)


m11_up_tf = tf.constant(m11_up, dtype=tf.float32)
m12_up_tf = tf.constant(m12_up, dtype=tf.float32)
m13_up_tf = tf.constant(m13_up, dtype=tf.float32)
m22_up_tf = tf.constant(m22_up, dtype=tf.float32)
m23_up_tf = tf.constant(m23_up, dtype=tf.float32)
m33_up_tf = tf.constant(m33_up, dtype=tf.float32)

m11_low_tf = tf.constant(m11_low, dtype=tf.float32)
m12_low_tf = tf.constant(m12_low, dtype=tf.float32)
m13_low_tf = tf.constant(m13_low, dtype=tf.float32)
m22_low_tf = tf.constant(m22_low, dtype=tf.float32)
m23_low_tf = tf.constant(m23_low, dtype=tf.float32)
m33_low_tf = tf.constant(m33_low, dtype=tf.float32)

theta_angles = tf.constant(np.arange(0, 91, 1), dtype=tf.float32)
phi_angles = tf.constant(np.arange(0, 361, 1), dtype=tf.float32)


# ============================================================================
# 3. Physics-based image generation (differentiable)
# ============================================================================
def compute_physical_images_tf(theta_soft, phi_soft):
    theta_rad = theta_soft * tf.constant(np.pi / 180.0, dtype=tf.float32)
    phi_rad = phi_soft * tf.constant(np.pi / 180.0, dtype=tf.float32)

    theta_rad = theta_rad[:, tf.newaxis, tf.newaxis]
    phi_rad = phi_rad[:, tf.newaxis, tf.newaxis]

    I_up = (
        tf.sin(theta_rad) ** 2 * tf.cos(phi_rad) ** 2 * m11_up_tf
        + tf.sin(2.0 * phi_rad) * m12_up_tf
        + tf.sin(phi_rad) ** 2 * m22_up_tf
        + tf.sin(2.0 * theta_rad) * tf.cos(phi_rad) * m13_up_tf
        + tf.sin(phi_rad) * m23_up_tf
        + tf.cos(theta_rad) ** 2 * m33_up_tf
    )

    I_low = (
        tf.sin(theta_rad) ** 2 * tf.cos(phi_rad) ** 2 * m11_low_tf
        + tf.sin(2.0 * phi_rad) * m12_low_tf
        + tf.sin(phi_rad) ** 2 * m22_low_tf
        + tf.sin(2.0 * theta_rad) * tf.cos(phi_rad) * m13_low_tf
        + tf.sin(phi_rad) * m23_low_tf
        + tf.cos(theta_rad) ** 2 * m33_low_tf
    )

    return normalize_image(I_up), normalize_image(I_low)


# ============================================================================
# 4. Learnable dynamic weight module
# ============================================================================
class LearnableWeightModule(tf.keras.layers.Layer):
    def __init__(self, num_components=2, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.log_vars = self.add_weight(
            name="log_vars",
            shape=(num_components,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, losses):
        if len(losses) != self.num_components:
            raise ValueError(
                f"Expected {self.num_components} losses, got {len(losses)}"
            )

        weights = tf.exp(-self.log_vars)
        weights = weights / tf.reduce_sum(weights)

        weighted_losses = [weights[i] * losses[i] for i in range(self.num_components)]
        total = tf.add_n(weighted_losses)

        reg = tf.reduce_sum(self.log_vars)
        return total + 0.01 * reg, weights


# ============================================================================
# 5. Custom loss
# ============================================================================
def custom_loss(y_true, y_pred, original_up_images, original_low_images, weight_module):
    theta_pred = y_pred[:, :91]
    phi_pred = y_pred[:, 91:]

    theta_pred_soft = tf.reduce_sum(theta_pred * theta_angles, axis=1)
    phi_pred_soft = tf.reduce_sum(phi_pred * phi_angles, axis=1)

    theta_true = y_true[:, :91]
    phi_true = y_true[:, 91:]

    theta_true_soft = tf.reduce_sum(theta_true * theta_angles, axis=1)
    phi_true_soft = tf.reduce_sum(phi_true * phi_angles, axis=1)

    I_up_pred, I_low_pred = compute_physical_images_tf(theta_pred_soft, phi_pred_soft)
    I_up_ideal, I_low_ideal = compute_physical_images_tf(theta_true_soft, phi_true_soft)

    original_up = tf.squeeze(original_up_images, axis=-1)
    original_low = tf.squeeze(original_low_images, axis=-1)

    loss_data = tf.reduce_mean(tf.square(original_up - I_up_pred)) + \
                tf.reduce_mean(tf.square(original_low - I_low_pred))

    loss_angle = tf.reduce_mean(tf.square(I_up_pred - I_up_ideal)) + \
                 tf.reduce_mean(tf.square(I_low_pred - I_low_ideal))

    loss_cls = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

    physics_losses = [loss_data, loss_angle]
    weighted_physics, physics_weights = weight_module(physics_losses)

    total_loss = weighted_physics + 0.3 * loss_cls

    loss_dict = {
        "total": total_loss,
        "data": loss_data,
        "angle": loss_angle,
        "cls": loss_cls,
        "w_data": physics_weights[0],
        "w_angle": physics_weights[1],
    }

    return total_loss, loss_dict


# ============================================================================
# 6. Transformer model architecture
# ============================================================================
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(input_shape, num_classes, embed_dim=128, num_heads=8,
                            ff_dim=1024, transformer_layers=4, patch_size=4):
    inputs = Input(shape=input_shape)
    c = input_shape[-1]
    patch_dim = patch_size * patch_size * c

    def extract_patches_layer(images):
        return tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

    patches = Lambda(extract_patches_layer)(inputs)
    x = Reshape((-1, patch_dim))(patches)
    x = Dense(embed_dim)(x)

    for _ in range(transformer_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)

    return Model(inputs, outputs)


# ============================================================================
# 7. Main training pipeline
# ============================================================================
if __name__ == "__main__":
    epochs = 100
    init_lr = 1e-4
    batch_size = 400
    num_class = 452

    path = "/root/autodl-tmp"
    savepath = "/root/autodl-tmp"
    checkpoint_dir = os.path.join(savepath, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---------- Load data ----------
    with h5py.File(path + "/Meansub_train_data.h5", "r") as file:
        trainX = file["Meansub_train_data"][:]

    with h5py.File(path + "/Meansub_test_data.h5", "r") as file:
        testX = file["Meansub_test_data"][:]

    with h5py.File(path + "/Meansub_train_label.h5", "r") as file:
        trainY = file["Meansub_train_label"][:]

    with h5py.File(path + "/Meansub_test_label.h5", "r") as file:
        testY = file["Meansub_test_label"][:]

    print("Train data shape:", trainX.shape)
    print("Test data shape:", testX.shape)
    print("Train label shape:", trainY.shape)
    print("Test label shape:", testY.shape)

    trainX = np.expand_dims(trainX, axis=-1).astype(np.float32)
    testX = np.expand_dims(testX, axis=-1).astype(np.float32)

    print("Adjusted train data shape:", trainX.shape)
    print("Adjusted test data shape:", testX.shape)
    print("{} images for training, {} images for validation".format(len(trainX), len(testX)))

    img_h, img_w, img_c = trainX.shape[1], trainX.shape[2], trainX.shape[3]

    trainX_up = trainX[:, :14, :, :]
    trainX_low = trainX[:, 14:, :, :]
    testX_up = testX[:, :14, :, :]
    testX_low = testX[:, 14:, :, :]

    print("Train upper shape:", trainX_up.shape)
    print("Train lower shape:", trainX_low.shape)

    model = build_transformer_model((img_h, img_w, img_c), num_class, patch_size=4)
    weight_module = LearnableWeightModule(num_components=2)
    model.summary()

    optimizer = Adam(learning_rate=init_lr)

    def create_dataset(X, Y, X_up, X_low, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y, X_up, X_low))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(trainX, trainY, trainX_up, trainX_low, batch_size, shuffle=True)
    test_dataset = create_dataset(testX, testY, testX_up, testX_low, batch_size, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_losses = []
        train_accs = []
        epoch_w_data = []
        epoch_w_angle = []

        for x_batch, y_batch, x_up_batch, x_low_batch in train_dataset:
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss, loss_dict = custom_loss(y_batch, y_pred, x_up_batch, x_low_batch, weight_module)

                theta_pred = y_pred[:, :91]
                phi_pred = y_pred[:, 91:]
                theta_true = y_batch[:, :91]
                phi_true = y_batch[:, 91:]

                theta_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(theta_pred, axis=1), tf.argmax(theta_true, axis=1)), tf.float32)
                )
                phi_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(phi_pred, axis=1), tf.argmax(phi_true, axis=1)), tf.float32)
                )
                acc = (theta_acc + phi_acc) / 2.0

            all_vars = model.trainable_variables + weight_module.trainable_variables
            grads = tape.gradient(loss, all_vars)
            optimizer.apply_gradients(zip(grads, all_vars))

            train_losses.append(loss_dict["total"].numpy())
            train_accs.append(acc.numpy())
            epoch_w_data.append(loss_dict["w_data"].numpy())
            epoch_w_angle.append(loss_dict["w_angle"].numpy())

        val_losses = []
        val_accs = []

        for x_batch, y_batch, x_up_batch, x_low_batch in test_dataset:
            y_pred = model(x_batch, training=False)
            loss, loss_dict = custom_loss(y_batch, y_pred, x_up_batch, x_low_batch, weight_module)

            theta_pred = y_pred[:, :91]
            phi_pred = y_pred[:, 91:]
            theta_true = y_batch[:, :91]
            phi_true = y_batch[:, 91:]

            theta_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(theta_pred, axis=1), tf.argmax(theta_true, axis=1)), tf.float32)
            )
            phi_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(phi_pred, axis=1), tf.argmax(phi_true, axis=1)), tf.float32)
            )
            acc = (theta_acc + phi_acc) / 2.0

            val_losses.append(loss_dict["total"].numpy())
            val_accs.append(acc.numpy())

        train_loss_mean = np.mean(train_losses)
        train_acc_mean = np.mean(train_accs)
        val_loss_mean = np.mean(val_losses)
        val_acc_mean = np.mean(val_accs)

        print(f"Train Loss: {train_loss_mean:.4f}, Train Acc: {train_acc_mean:.4f}")
        print(f"Val Loss: {val_loss_mean:.4f}, Val Acc: {val_acc_mean:.4f}")
        print(
            f"Weights: w_data={np.mean(epoch_w_data):.4f}, "
            f"w_angle={np.mean(epoch_w_angle):.4f}"
        )

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            model.save_weights(os.path.join(checkpoint_dir, "checkmodel.weights.h5"))
            print("Saved best model!")

        if epoch > 0 and (epoch + 1) % 10 == 0:
            current_lr = optimizer.learning_rate.numpy()
            optimizer.learning_rate.assign(current_lr * 0.1)
            print(f"Learning rate reduced to {optimizer.learning_rate.numpy():.2e}")

    model.save(os.path.join(savepath, "model.h5"))
    model.save_weights(os.path.join(savepath, "weights.h5"))

    print("======================== Training Completed ===========================")