from SpeakerNet import SpeakerNet, WrappedModel, ModelTrainer
from DatasetLoader import loadWAV
from tools.wrapper_tools import extract_defaults_from_trainSpeakerNet, update_args_with_config_file
from tools.AudioPreprocessingTool import AudioPreprocessingTool
import torch.nn.functional as F
import torch
import os

class VoxWrapper:
    def __init__(self, model_type, separation_threshold):

        trainSpeakerNet_path = './trainSpeakerNet.py'
        self.args = extract_defaults_from_trainSpeakerNet(trainSpeakerNet_path)
        self.args.eval = True
        self.separation_threshold = separation_threshold

        if model_type == "baseline_lite_ap":
            self.args.initial_model = './models/weights/RawNetSE34L/baseline_lite_ap.model'
            self.args.model = "ResNetSE34L"
            self.args.log_input = True
            self.args.trainfunc = 'angleproto'
            self.eval_frame = 400

        if model_type == "model500":
            self.args.initial_model = './models/weights/RawNetSE34L/model000000500.model'
            self.args.model = "ResNetSE34L"
            self.args.log_input = True
            self.args.trainfunc = 'angleproto'
            self.eval_frame = 400

        if model_type == "rawnet3":
            self.args.initial_model = './models/weights/RawNet3/model.pt'
            config_path = './configs/RawNet3_AAM.yaml'
            self.args = update_args_with_config_file(self.args, config_path)

        # Check if CUDA is available
        if torch.cuda.is_available():
            # Get the index of the current CUDA device
            self.args.gpu = torch.cuda.current_device()
            print(f"Current CUDA Device Number: {self.args.gpu}")
        else:
            # Raise an exception if CUDA is not available
            raise RuntimeError("CUDA is not available. You cannot run the model.")

        s = SpeakerNet(**vars(self.args))
        s = WrappedModel(s).cuda(self.args.gpu)
        self._trainer = ModelTrainer(s, **vars(self.args))
        self._trainer.loadParameters(self.args.initial_model)
        print(f"Model parameters loaded from: {self.args.initial_model}")

        self.separation_threshold = separation_threshold
        print(f"Separation threshold set to: {self.separation_threshold}")

    @staticmethod
    def check_if_16k_mono(file_path):
        """
        Validates if the specified audio file is 16 kHz mono.

        Args:
            file_path (str): Path to the audio file.

        Raises:
            ValueError: If the audio file is not 16 kHz mono.
        """
        audio = AudioPreprocessingTool.load_audio_file(file_path)

        if audio.frame_rate != 16000 or len(audio.split_to_mono()) != 1:
            raise ValueError("The audio file must be 16 kHz mono.")

    def generate_embeddings(self, audio_path):
        self.check_if_16k_mono(audio_path)
        data = torch.FloatTensor(loadWAV(audio_path, self.args.max_frames)).cuda()
        with torch.no_grad():
            ref_feat = self._trainer.__model__(data).detach().cpu()
        return ref_feat

    def calculate_similarity(self, embeddings1, embeddings2):
        """
        Calculates the mean cosine similarity between two sets of embeddings and checks it against a separation threshold.

        Args:
            embeddings1 (torch.Tensor): A tensor of shape (m, 512) containing m embeddings.
            embeddings2 (torch.Tensor): A tensor of shape (n, 512) containing n embeddings.

        Returns:
            float: The mean cosine similarity between all pairs of embeddings.
            bool: True if the mean similarity is greater than the separation threshold, else False.
        """
        # Ensure embeddings are on the same device
        embeddings1 = embeddings1.cuda()
        embeddings2 = embeddings2.cuda()

        # Normalize embeddings along the feature dimension
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Calculate the cosine similarity between all pairs
        similarity_matrix = torch.matmul(embeddings1, embeddings2.transpose(0, 1))

        # Calculate the mean of the cosine similarities
        mean_similarity = torch.mean(similarity_matrix)

        # Check against the separation threshold
        is_above_threshold = mean_similarity > self.separation_threshold

        return mean_similarity.item(), is_above_threshold.item()

