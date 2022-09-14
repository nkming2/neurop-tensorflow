import argparse

import imageio
import numpy as np
import tensorflow as tf


class Operator(tf.keras.Model):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.encoder = tf.keras.layers.Conv2D(64, (1, 1), name="encoder")
		self.mid_conv = tf.keras.layers.Conv2D(64, (1, 1), name="mid_conv")
		self.decoder = tf.keras.layers.Conv2D(3, (1, 1), name="decoder")
		self.act = tf.keras.layers.LeakyReLU(alpha=1e-2)

	def call(self, inputs, val):
		x_code = self.encoder(inputs)
		y_code = x_code + val
		y_code = self.act(self.mid_conv(y_code))
		y = self.decoder(y_code)
		return y


class Renderer(tf.keras.Model):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.ex_block = Operator(name="ex_renderer")
		self.bc_block = Operator(name="bc_renderer")
		self.vb_block = Operator(name="vb_renderer")

	def call(self, x_ex, x_bc, x_vb, v_ex, v_bc, v_vb):
		rec_ex = self.ex_block(x_ex, 0)
		rec_bc = self.bc_block(x_bc, 0)
		rec_vb = self.vb_block(x_vb, 0)

		map_ex = self.ex_block(x_ex, v_ex)
		map_bc = self.bc_block(x_bc, v_bc)
		map_vb = self.vb_block(x_vb, v_vb)

		return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb


class Encoder(tf.keras.Model):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.conv1 = tf.keras.layers.Conv2D(32, (7, 7),
		                                    strides=(2, 2),
		                                    activation="relu",
		                                    use_bias=True,
		                                    name="conv1")
		self.conv2 = tf.keras.layers.Conv2D(32, (3, 3),
		                                    strides=(2, 2),
		                                    activation="relu",
		                                    use_bias=True,
		                                    name="conv2")
		self.pad = tf.keras.layers.ZeroPadding2D()

	def call(self, inputs: tf.Tensor):
		x1 = self.conv1(self.pad(inputs))
		x2 = self.conv2(self.pad(x1))
		std = tf.math.reduce_std(x2, [1, 2])
		mean = tf.math.reduce_mean(x2, [1, 2])
		max = tf.math.reduce_max(x2, [1, 2])
		out = tf.concat([std, mean, max], 1)
		return out


class Predictor(tf.keras.Model):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.fc3 = tf.keras.layers.Dense(1, activation="tanh", name="fc3")

	def call(self, inputs):
		out = self.fc3(inputs)
		return out


class NeurOP(tf.keras.Model):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.image_encoder = Encoder(name="image_encoder")
		renderer = Renderer()

		self.bc_renderer = renderer.bc_block
		self.bc_predictor = Predictor(name="bc_predictor")

		self.ex_renderer = renderer.ex_block
		self.ex_predictor = Predictor(name="ex_predictor")

		self.vb_renderer = renderer.vb_block
		self.vb_predictor = Predictor(name="vb_predictor")

		self.renderers = [self.bc_renderer, self.ex_renderer, self.vb_renderer]
		self.predict_heads = [
		    self.bc_predictor, self.ex_predictor, self.vb_predictor
		]

	@tf.function(input_signature=[
	    tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32)
	])
	def call(self, img):
		vals = []
		for nop, predict_head in zip(self.renderers, self.predict_heads):
			img_resized = tf.image.resize(img, (256, 256),
			                              preserve_aspect_ratio=True)
			feat = self.image_encoder(img_resized)
			scalar = predict_head(feat)
			vals.append(scalar)
			img = nop(img, scalar)
		img = tf.clip_by_value(img, 0.0, 1.0)
		return img


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("img", type=str)
	args = ap.parse_args()
	input_img = args.img

	model = NeurOP()
	model.load_weights("checkpoints/neurop_fivek_lite")

	img = np.array(imageio.imread(input_img)) / (2**8 - 1)
	img = np.expand_dims(img, 0)
	input = tf.convert_to_tensor(img)
	output = model.predict(input)

	out_img = output
	out_img = np.squeeze(out_img, 0)
	out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
	imageio.imsave("result.jpg", out_img)
