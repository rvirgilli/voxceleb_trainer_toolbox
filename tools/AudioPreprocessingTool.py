from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import warnings

class AudioPreprocessingTool:
    class AudioSegmentHandler:
        def __init__(self, audio_segment):
            self.audio = audio_segment
            self.sampling_rate = audio_segment.frame_rate
            self.bit_depth = audio_segment.sample_width * 8
            self.duration_seconds = len(audio_segment) / 1000.0

    def __init__(self, file_path, channel_index):
        """
        Initializes the AudioPreprocessingTool with an audio file and a specific channel.

        Args:
            file_path (str): Path to the audio file.
            channel_index (int): Index of the specific channel to focus on.
        """
        self.original_audio = None
        self.processed_audio = None
        self.load_audio(file_path, channel_index)
        self._speech_segments = None
        self._audio_chunks = None

    @staticmethod
    def load_audio_file(file_path):
        """
        Loads an audio file based on its extension.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            AudioSegment: Loaded audio segment.

        Raises:
            ValueError: If the file format is unsupported.
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.wav':
            return AudioSegment.from_wav(file_path)
        elif file_extension == '.mp3':
            return AudioSegment.from_mp3(file_path)
        elif file_extension == '.ogg' or file_extension == '.opus':
            return AudioSegment.from_ogg(file_path)
        elif file_extension == '.flv':
            return AudioSegment.from_flv(file_path)
        else:
            raise ValueError("Unsupported file format")

    def load_audio(self, file_path, channel_index):
        """
        Loads an audio file and sets up the selected channel.

        Args:
            file_path (str): Path to the audio file.
            channel_index (int): Index of the specific channel to focus on.
        """
        audio = self.load_audio_file(file_path)
        selected_channel = audio.split_to_mono()[channel_index]
        self.original_audio = self.AudioSegmentHandler(selected_channel)
        self.processed_audio = self.AudioSegmentHandler(selected_channel)

    def resample_audio(self, new_sample_rate):
        self.processed_audio.audio = self.processed_audio.audio.set_frame_rate(new_sample_rate)
        self.processed_audio.sampling_rate = new_sample_rate

    def detect_speech_segments(self, silence_thresh=-30, min_silence_len=2000, min_segment_len=500, force=False):
        """
        Detects speech segments in the processed audio channel.

        Args:
            silence_thresh (int, optional): Silence threshold in dB. Defaults to -30.
            min_silence_len (int, optional): Minimum silence length in milliseconds. Defaults to 2000.
            min_segment_len (int, optional): Minimum length of a segment to be included in milliseconds. Defaults to 500.
            force (bool, optional): If True, forces a recalculation of speech segments. Defaults to False.

        Returns:
            list: A list of dictionaries, each containing 'start', 'end', and 'segment'.
        """
        if not force and self._speech_segments is not None:
            warnings.warn("Speech segments have already been calculated. Returning stored segments.")
            return self._speech_segments

        nonsilent_chunks = detect_nonsilent(
            self.processed_audio.audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        speech_segments = []
        for start, end in nonsilent_chunks:
            if end - start >= min_segment_len:
                segment = self.processed_audio.audio[start:end]
                segment_info = {
                    'start': start,
                    'end': end,
                    'segment': segment
                }
                speech_segments.append(segment_info)

        self._speech_segments = speech_segments
        return speech_segments


    def join_segments(self, fade_duration=50, silence_duration=500):
        """
        Joins stored speech segments with fade-in, fade-out, and silence between them.

        Args:
            fade_duration (int, optional): Duration of fade in and fade out in milliseconds. Defaults to 50.
            silence_duration (int, optional): Duration of silence between segments in milliseconds. Defaults to 500.

        Modifies:
            self.processed_audio.audio: Overwrites with the joined AudioSegment containing the desired segments.
        """
        if self._speech_segments is None:
            warnings.warn("No speech segments have been detected. Cannot join segments.")
            return

        result = AudioSegment.empty()
        silence = AudioSegment.silent(duration=silence_duration)

        for i, segment_info in enumerate(self._speech_segments):
            segment = segment_info['segment']

            # Apply fade-in and fade-out to every segment
            segment = segment.fade_in(duration=fade_duration).fade_out(duration=fade_duration)

            if i == 0:
                result += segment
            else:
                result += (silence + segment)

        # Update processed_audio with the joined segments
        self.processed_audio.audio = result

        return result

    def break_into_chunks(self, chunk_size=5000, fade_duration=50):
        """
        Breaks the processed audio into equal parts, and from each part,
        extracts a segment of duration chunk_size milliseconds that is
        centered within that part. Applies fade-in and fade-out to each chunk,
        and stores the list of these chunks in self._audio_chunks.

        Args:
            chunk_size (int, optional): Size of each chunk in milliseconds. Defaults to 5000.
            fade_duration (int, optional): Duration of fade in and fade out in milliseconds. Defaults to 50.
        """
        chunks = []
        audio = self.processed_audio.audio
        n_chunks = len(audio) // chunk_size
        if n_chunks == 0:
            self._audio_chunks = chunks
            return []

        part_duration = len(audio) / n_chunks

        for i in range(n_chunks):
            part_start_time = int(i * part_duration)
            part_end_time = int((i + 1) * part_duration)

            chunk_start_time = part_start_time + int((part_duration - chunk_size) / 2)
            chunk_end_time = chunk_start_time + chunk_size

            chunk = audio[chunk_start_time:chunk_end_time]
            faded_chunk = chunk.fade_in(fade_duration).fade_out(fade_duration)

            chunks.append(faded_chunk)

        self._audio_chunks = chunks

        return chunks

    def trim_audio(self, duration_ms, trim_from_start=True):
        """
        Trims the specified duration from either the start or end of the processed audio.
        Returns an empty AudioSegment if the trim duration exceeds the audio length.

        Args:
            duration_ms (int): The duration in milliseconds to trim.
            trim_from_start (bool, optional): Determines if trimming is from the start.
                                              If False, trims from the end. Defaults to True.
        """
        if duration_ms < 0:
            raise ValueError("Duration must be a non-negative value")

        audio_length_ms = len(self.processed_audio.audio)

        if duration_ms > audio_length_ms:
            # Return an empty AudioSegment if the duration to trim exceeds the audio length
            self.processed_audio.audio = AudioSegment.empty()
        elif trim_from_start:
            # Trim the start of the audio
            self.processed_audio.audio = self.processed_audio.audio[duration_ms:]
        else:
            # Trim the end of the audio
            self.processed_audio.audio = self.processed_audio.audio[:audio_length_ms - duration_ms]

        return self.processed_audio.audio