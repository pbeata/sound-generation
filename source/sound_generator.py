
from audio_preprocessing import MinMaxNormalizer
import librosa

class SoundGenerator:
	"""
	SoundGenerator is responsible for generating audio from spectrograms.
	"""

	def __init__(self, vae, hop_length):
		self.vae = vae
		self.hop_length = hop_length
		self._min_max_normalizer = MinMaxNormalizer(0, 1)

	def generate(self, spectrograms, min_mav_values):
		generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
		signals = self.convert_spectrograms_to_audio(generated_spectrograms,
													 min_mav_values)
		return signals, latent_representations

	def convert_spectrograms_to_audio(self, spectrograms, min_mav_values):
		signals = []
		for spectrogram, min_mav_value in zip(spectrograms, min_mav_values):
			# reshape the log spectrogram (drop 3rd dimension channel)
			log_spectrogram = spectrogram[:, :, 0]
			# apply de-normalization
			denorm_log_spec = self._min_max_normalizer.denormalize(log_spectrogram,
																   min_mav_value["min"],
																   min_mav_value["max"])
			# log spectrogram --> linear spectrogram (dB to amplitude)
			spec = librosa.db_to_amplitude(denorm_log_spec)
			# apply Griffin-Lin algorithm
			signal = librosa.istft(spec, hop_length=self.hop_length)
			# append signal to the "signals" list
			signals.append(signal)
		return signals