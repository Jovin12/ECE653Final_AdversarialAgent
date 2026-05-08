"""
audio_target.py - Speech Commands with SimpleCNN, ResNet18, and VGGish models
Uses local .wav files without torchaudio.load dependency
"""

import os
import wave
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchaudio.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from art.estimators.classification import PyTorchClassifier

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------
def _get_device() -> tuple[torch.device, str]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), "cpu"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    else:
        return torch.device("cpu"), "cpu"

TORCH_DEVICE, ART_DEVICE = _get_device()
print(f"[AudioTarget] Device: {TORCH_DEVICE}  (ART sees: '{ART_DEVICE}')")

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
SPEECH_COMMANDS_CONFIG = {
    'sample_rate': 16000,
    'duration_sec': 1.0,
    'n_mels': 64,
    'hop_length': 160,
    'n_fft': 400,
    'num_samples': 16000,
    'time_frames': 101,
}

# For VGGish: 96x64 input (time_frames x n_mels)
VGGISH_CONFIG = {
    'n_mels': 64,
    'time_frames': 96,  # VGGish expects 96x64 inputs
}

# ---------------------------------------------------------
# Simple CNN for Audio
# ---------------------------------------------------------
class SimpleAudioCNN(nn.Module):
    """
    Simple CNN for audio classification.
    Input: (batch, 1, 64, 101) - 64 mel bands x 101 time frames
    """
    def __init__(self, n_classes: int = 35):
        super(SimpleAudioCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1: 64x101 -> 32x50
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32x50 -> 16x25
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 16x25 -> 8x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 8x12 -> 4x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 101)
            dummy = self.conv_layers(dummy)
            self.flattened_size = dummy.view(1, -1).size(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# ResNet18 for Audio
# ---------------------------------------------------------
class ResNet18Audio(nn.Module):
    """ResNet18 adapted for audio mel-spectrograms"""
    def __init__(self, n_classes: int = 35, pretrained: bool = True):
        super(ResNet18Audio, self).__init__()
        
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet18(weights=None)
        
        # Modify first conv for mono audio (1 channel instead of 3)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final FC layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, n_classes)
        
        # Initialize new layers
        nn.init.kaiming_normal_(self.resnet.conv1.weight)
        nn.init.normal_(self.resnet.fc.weight, std=0.01)
        
    def forward(self, x):
        return self.resnet(x)


# ---------------------------------------------------------
# VGGish Model (Google's VGGish for Audio)
# ---------------------------------------------------------
class VGGish(nn.Module):
    """
    VGGish model for audio classification.
    Based on Google's VGGish architecture.
    Input: (batch, 1, 96, 64) - 96 time frames x 64 mel bands
    """
    def __init__(self, n_classes: int = 35):
        super(VGGish, self).__init__()
        
        # VGGish frontend (convolutional layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only pool time dimension
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 96, 64)
            dummy = self.features(dummy)
            self.flattened_size = dummy.view(1, -1).size(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes),
        )
    
    def forward(self, x):
        # x shape: (batch, 1, time_frames, n_mels)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# Wave File Reader
# ---------------------------------------------------------
def read_wav_file(filepath, target_sr=16000, target_samples=16000):
    """Read a WAV file using Python's wave module"""
    try:
        with wave.open(filepath, 'rb') as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            
            frames = wav.readframes(n_frames)
            
            if sampwidth == 1:
                waveform = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                waveform = (waveform - 128) / 128.0
            elif sampwidth == 2:
                waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                waveform = waveform / 32768.0
            elif sampwidth == 4:
                waveform = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
                waveform = waveform / 2147483648.0
            else:
                return None
            
            if n_channels > 1:
                waveform = waveform.reshape(-1, n_channels)
                waveform = waveform.mean(axis=1)
            
            if framerate != target_sr:
                waveform_tensor = torch.from_numpy(waveform).float()
                resampler = T.Resample(framerate, target_sr)
                waveform = resampler(waveform_tensor).numpy()
            
            if len(waveform) < target_samples:
                waveform = np.pad(waveform, (0, target_samples - len(waveform)))
            else:
                waveform = waveform[:target_samples]
            
            return waveform.astype(np.float32)
    except Exception as e:
        return None


# ---------------------------------------------------------
# Audio Preprocessor
# ---------------------------------------------------------
class AudioPreprocessor:
    def __init__(self, n_mels=64, time_frames=101, device=TORCH_DEVICE):
        self.sample_rate = SPEECH_COMMANDS_CONFIG['sample_rate']
        self.num_samples = SPEECH_COMMANDS_CONFIG['num_samples']
        self.n_mels = n_mels
        self.time_frames = time_frames
        self.hop_length = SPEECH_COMMANDS_CONFIG['hop_length']
        self.n_fft = SPEECH_COMMANDS_CONFIG['n_fft']
        self.device = device
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            normalized=True,
        ).to(device)
    
    def waveform_to_spectrogram(self, waveform):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-10)
        
        # Resize time dimension if needed
        if mel_spec.size(2) != self.time_frames:
            mel_spec = F.interpolate(
                mel_spec.unsqueeze(0), 
                size=(self.n_mels, self.time_frames), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Normalize
        mean = mel_spec.mean()
        std = mel_spec.std()
        if std > 0:
            mel_spec = (mel_spec - mean) / std
        
        return mel_spec.cpu()
    
    def __call__(self, waveform):
        return self.waveform_to_spectrogram(waveform)


# ---------------------------------------------------------
# Find Speech Commands Directory
# ---------------------------------------------------------
def find_speech_commands_directory():
    search_paths = [
        "./data/SpeechCommands/speech_commands_v0.02",
        "./data/speech_commands_v0.02",
        "./data/SpeechCommands",
        "./data",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                items = os.listdir(path)
                class_dirs = [d for d in items 
                            if os.path.isdir(os.path.join(path, d)) 
                            and not d.startswith('_') 
                            and not d.startswith('.')]
                if class_dirs:
                    return path
            except:
                continue
    return None


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
def load_speech_commands_dataset(
    sample_limit: int = 500,
    test_split: float = 0.3,
    n_mels: int = 64,
    time_frames: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Speech Commands dataset"""
    
    print(f"[AudioTarget] Loading Speech Commands...")
    print(f"  Target spectrogram size: ({n_mels}, {time_frames})")
    
    data_dir = find_speech_commands_directory()
    if data_dir is None:
        raise RuntimeError("Speech Commands dataset not found!")
    
    # Get class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) 
                  and not d.startswith('_') 
                  and not d.startswith('.')]
    class_dirs.sort()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
    n_classes = len(class_dirs)
    
    print(f"  Found {n_classes} classes")
    
    # Collect files
    max_per_class = max(1, sample_limit // n_classes) if sample_limit else 100
    
    file_paths = []
    labels = []
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        wav_files = wav_files[:max_per_class]
        
        for wav_file in wav_files:
            file_paths.append(os.path.join(class_path, wav_file))
            labels.append(class_to_idx[class_name])
    
    print(f"  Collected {len(file_paths)} files ({max_per_class} per class max)")
    
    # Process files
    preprocessor = AudioPreprocessor(n_mels=n_mels, time_frames=time_frames)
    x_data = []
    y_data = []
    
    print(f"  Processing audio files...")
    
    for idx, (filepath, label) in enumerate(zip(file_paths, labels)):
        waveform = read_wav_file(filepath)
        if waveform is not None:
            try:
                spectrogram = preprocessor(waveform)
                x_data.append(spectrogram.numpy())
                y_data.append(label)
            except:
                pass
        
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(file_paths)} files")
    
    if len(x_data) == 0:
        raise RuntimeError("No audio files could be processed!")
    
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int64)
    
    print(f"  Successfully processed: {len(x_data)} samples")
    print(f"  Input shape: {x_data.shape}")
    
    # Train/test split
    n_samples = len(x_data)
    n_test = int(n_samples * test_split)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    x_train, x_test = x_data[train_idx], x_data[test_idx]
    y_train, y_test = y_data[train_idx], y_data[test_idx]
    
    print(f"  Train: {len(x_train)}, Test: {len(x_test)}")
    
    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train_audio_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> nn.Module:
    """Train the model"""
    
    model.train()
    model.to(TORCH_DEVICE)
    
    x_tensor = torch.FloatTensor(x_train).to(TORCH_DEVICE)
    y_tensor = torch.LongTensor(y_train).to(TORCH_DEVICE)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"[AudioTarget] Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        scheduler.step()
        accuracy = 100 * correct / total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss: {total_loss/len(dataloader):.4f}, Acc: {accuracy:.1f}%")
    
    model.eval()
    return model


# ---------------------------------------------------------
# Model Factory
# ---------------------------------------------------------
def create_model(model_name: str, n_classes: int) -> nn.Module:
    """Create model by name"""
    
    if model_name == 'simple_cnn':
        model = SimpleAudioCNN(n_classes=n_classes)
    elif model_name == 'resnet18_audio':
        model = ResNet18Audio(n_classes=n_classes, pretrained=True)
    elif model_name == 'vggish':
        model = VGGish(n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: simple_cnn, resnet18_audio, vggish")
    
    return model


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def load_audio_target(
    model_name: str = 'simple_cnn',
    dataset_name: str = 'speech_commands',
    sample_limit: int = 500,
    train_model: bool = True,
    test_split: float = 0.3,
    epochs: int = 15,
) -> tuple[PyTorchClassifier, np.ndarray, np.ndarray]:
    """Load audio target with specified model on Speech Commands"""
    
    print(f"\n[AudioTarget] Loading {dataset_name} with {model_name}")
    
    # Configure input size based on model
    if model_name == 'vggish':
        n_mels = VGGISH_CONFIG['n_mels']
        time_frames = VGGISH_CONFIG['time_frames']
    else:  # simple_cnn or resnet18_audio
        n_mels = SPEECH_COMMANDS_CONFIG['n_mels']
        time_frames = SPEECH_COMMANDS_CONFIG['time_frames']
    
    # Load dataset
    x_train, x_test, y_train, y_test = load_speech_commands_dataset(
        sample_limit=sample_limit,
        test_split=test_split,
        n_mels=n_mels,
        time_frames=time_frames,
    )
    
    input_shape = x_train.shape[1:]
    n_classes = len(np.unique(y_train))
    
    print(f"\n[AudioTarget] Input shape: {input_shape}")
    print(f"[AudioTarget] Number of classes: {n_classes}")
    
    # Create model
    model = create_model(model_name, n_classes)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[AudioTarget] Model parameters: {n_params:,}")
    
    # Train
    if train_model:
        model = train_audio_model(model, x_train, y_train, epochs=epochs)
    
    # Test accuracy
    model.eval()
    with torch.no_grad():
        batch_size = 32
        correct = 0
        total = 0
        for i in range(0, len(x_test), batch_size):
            batch_x = torch.FloatTensor(x_test[i:i+batch_size]).to(TORCH_DEVICE)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted.cpu().numpy() == y_test[i:i+batch_size]).sum()
            total += len(batch_x)
        
        clean_acc = 100 * correct / total
    
    print(f"\n[AudioTarget] Clean test accuracy: {clean_acc:.1f}%")
    
    # Clip values
    clip_min = float(x_train.min()) - 0.5
    clip_max = float(x_train.max()) + 0.5
    
    # Wrap with ART
    art_classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=n_classes,
        clip_values=(clip_min, clip_max),
        device_type=ART_DEVICE,
    )
    
    print(f"[AudioTarget] Ready for attacks! x_test shape: {x_test.shape}")
    
    return art_classifier, x_test.astype(np.float32), y_test.astype(np.int64)


# ---------------------------------------------------------
# Quick Test
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING AUDIO MODELS")
    print("="*60)
    
    for model_name in ['simple_cnn', 'resnet18_audio', 'vggish']:
        print(f"\n--- Testing {model_name} ---")
        try:
            classifier, x_test, y_test = load_audio_target(
                model_name=model_name,
                sample_limit=200,
                train_model=True,
                epochs=10,
            )
            print(f"✓ {model_name} loaded successfully!")
            print(f"  Test samples: {len(x_test)}")
            print(f"  Input shape: {x_test.shape}")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")