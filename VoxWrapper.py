from SpeakerNet import SpeakerNet, WrappedModel, ModelTrainer
from DatasetLoader import loadWAV
from tools.wrapper_tools import generate_args_with_config
from tools.AudioPreprocessingTool import AudioPreprocessingTool
import torch

class VoxWrapper:
    def __init__(self, initial_model, model_type):
        trainSpeakerNet_path = './trainSpeakerNet.py'
        self.args = None

        if model_type == "rawnet3":
            config_path = './configs/RawNet3_AAM.yaml'
            self.args = generate_args_with_config(trainSpeakerNet_path, config_path)
            self.args.initial_model = initial_model

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