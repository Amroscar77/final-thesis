from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
from django.conf import settings
from .forms import VideoUploadForm, ImageUploadForm

# ─── Try importing new dependencies (graceful fallback) ───
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

try:
    from .model_v2 import (
        DeepfakeDetectorV2, LegacyModel,
        load_model_v2, load_legacy_model,
        get_v2_transforms, IM_SIZE_V2, MEAN_V2, STD_V2
    )
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

# ─── Template names ───
index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

# ─── V1 Legacy config ───
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(
    mean=-1 * np.divide(mean, std),
    std=np.divide([1, 1, 1], std)
)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'

# On CPU-only machines cap frames to avoid RAM exhaustion / segfault
CPU_MAX_FRAMES = 20

# V1 transforms
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# V2 transforms
if HAS_V2:
    v2_transforms = get_v2_transforms(im_size=IM_SIZE_V2, is_training=False)
else:
    v2_transforms = train_transforms

# ─── MTCNN face detector (singleton) ───
mtcnn_detector = None

def get_mtcnn():
    global mtcnn_detector
    if mtcnn_detector is None and HAS_MTCNN:
        mtcnn_detector = MTCNN(
            image_size=300, margin=40, keep_all=False,
            device=device, post_process=False,
        )
    return mtcnn_detector


# ═══════════════════════════════════════════════
# V1 Legacy Model (backward compatibility)
# ═══════════════════════════════════════════════
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def build_image_model(num_classes=2):
    """
    Build image-only ResNeXt-50 model matching the inline training script.
    Keys: conv1.*, bn1.*, layer1-4.*, fc.1.* (Linear 2048->512), fc.4.* (Linear 512->num_classes)
    """
    m = models.resnext50_32x4d(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(512, num_classes),
    )
    return m


# Image transforms for 224×224 (matches training augmentation normalisation)
image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ═══════════════════════════════════════════════
# Face Detection (MTCNN or fallback)
# ═══════════════════════════════════════════════
def detect_face(rgb_frame, padding=40):
    """Detect and crop face from an RGB frame. Returns cropped face or None."""
    # Try MTCNN first
    mtcnn = get_mtcnn()
    if mtcnn is not None:
        try:
            from PIL import Image
            pil_img = Image.fromarray(rgb_frame)
            boxes, probs = mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                h, w = rgb_frame.shape[:2]
                y1 = max(0, y1 - padding)
                y2 = min(h, y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(w, x2 + padding)
                face = rgb_frame[y1:y2, x1:x2]
                if face.size > 0:
                    return face, (x1, y1, x2, y2)
        except Exception as e:
            print(f"MTCNN error: {e}")

    # Fallback to face_recognition
    if HAS_FACE_RECOGNITION:
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            h, w = rgb_frame.shape[:2]
            top = max(0, top - padding)
            bottom = min(h, bottom + padding)
            left = max(0, left - padding)
            right = min(w, right + padding)
            face = rgb_frame[top:bottom, left:right]
            if face.size > 0:
                return face, (left, top, right, bottom)

    # Last fallback: OpenCV Haar cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img_h, img_w = rgb_frame.shape[:2]
        y1 = max(0, y - padding)
        y2 = min(img_h, y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(img_w, x + w + padding)
        face = rgb_frame[y1:y2, x1:x2]
        if face.size > 0:
            return face, (x1, y1, x2, y2)

    return None, None


# ═══════════════════════════════════════════════
# Video Dataset
# ═══════════════════════════════════════════════
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []

        # Use evenly-spaced sampling so the model sees the full video, not just the first second
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(self.count, total) if total > 0 else self.count
        indices = [int(i) for i in np.linspace(0, max(total - 1, 0), n)] if total > 0 else list(range(n))
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face, _ = detect_face(rgb_frame)
            frame_to_use = face if face is not None else rgb_frame
            frames.append(self.transform(frame_to_use))
        cap.release()

        if len(frames) == 0:
            dummy = torch.zeros(3, 112, 112) if self.transform == train_transforms else torch.zeros(3, 300, 300)
            frames = [dummy] * self.count
        while len(frames) < self.count:
            frames.append(frames[-1].clone())
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)


# ═══════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════
def im_convert(tensor, video_file_name):
    """Display a tensor as an image."""
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image


def predict_v1(model, img, path='./', video_file_name=""):
    """Predict using V1 legacy model."""
    fmap, logits = model(img.to(device))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('confidence of prediction:', confidence)
    return [int(prediction.item()), confidence]


def predict_v2(model, frames_tensor, mode='video'):
    """Predict using V2 model."""
    with torch.no_grad():
        if mode == 'image':
            logits, extras = model(frames_tensor.to(device), mode='image')
        else:
            logits, extras = model(frames_tensor.to(device), mode='video')

    probs = torch.softmax(logits, dim=1)
    _, prediction = torch.max(probs, 1)
    confidence = probs[:, int(prediction.item())].item() * 100

    result = {
        'prediction': int(prediction.item()),
        'confidence': confidence,
        'attention_weights': extras.get('attention_weights', None),
    }
    return result


def generate_gradcam(model, input_tensor, mode='video'):
    """Generate Grad-CAM heatmap for explainability."""
    try:
        model.eval()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        if mode == 'image':
            logits, _ = model(input_tensor, mode='image')
        else:
            logits, _ = model(input_tensor, mode='video')

        # Get the predicted class
        pred_class = logits.argmax(dim=1).item()
        logits[0, pred_class].backward()

        # Get gradients from the spatial backbone's last conv layer
        # This is a simplified Grad-CAM
        return None  # Placeholder — full implementation requires hooks
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


# ═══════════════════════════════════════════════
# Model Selection
# ═══════════════════════════════════════════════
def get_accurate_model(sequence_length, model_version='v1'):
    """Find the best model file for given sequence length and version."""
    model_name = []
    sequence_model = []
    final_model = ""

    if model_version == 'v2':
        list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*v2*.pt"))
        if list_models:
            best_acc = 0
            for mp in list_models:
                basename = os.path.basename(mp)
                try:
                    acc = int(basename.split("_")[1])
                    if acc > best_acc:
                        best_acc = acc
                        final_model = mp
                except (IndexError, ValueError):
                    pass
            if final_model:
                return final_model
        return None  # No V2 weights found — caller must fall back to V1

    if model_version == 'image':
        # Prefer dedicated image models (model_XX_acc_image_*.pt)
        list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*image*.pt"))
        if list_models:
            best_acc = 0
            for mp in list_models:
                basename = os.path.basename(mp)
                try:
                    acc = int(basename.split("_")[1])
                    if acc > best_acc:
                        best_acc = acc
                        final_model = mp
                except (IndexError, ValueError):
                    pass
            if final_model:
                print(f"[INFO] Using image-specific model: {os.path.basename(final_model)}")
                return final_model
        # Fall through to V1 models if no image model exists

    # V1 models — score every video model and pick the best
    # Scoring: celeb-trained data is worth more (+5 acc points) because
    # it was trained on diverse celebrity faces, not just lab deepfakes.
    # LSTM handles any sequence length, so frame count is only a tiebreaker.
    all_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))
    video_models = [os.path.basename(m) for m in all_models
                    if "image" not in os.path.basename(m).lower()
                    and "v2" not in os.path.basename(m).lower()]

    best_score = -1
    for filename in video_models:
        try:
            acc = int(filename.split("_")[1])
        except (IndexError, ValueError):
            acc = 0
        # Bonus for models trained on celebrity / diverse data
        diversity_bonus = 5 if "celeb" in filename.lower() else 0
        score = acc + diversity_bonus
        if score > best_score:
            best_score = score
            final_model = os.path.join(settings.PROJECT_DIR, "models", filename)

    if final_model:
        print(f"[INFO] Selected model: {os.path.basename(final_model)}")
    return final_model


ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'])
ALLOWED_IMAGE_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'])


def allowed_video_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_image_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# ═══════════════════════════════════════════════
# Views
# ═══════════════════════════════════════════════

def index(request):
    """Home page with video and image upload."""
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        image_upload_form = ImageUploadForm()
        # Clear session
        for key in ['file_name', 'preprocessed_images', 'faces_cropped_images',
                     'upload_type', 'model_version']:
            if key in request.session:
                del request.session[key]

        context = {
            "video_form": video_upload_form,
            "image_form": image_upload_form,
            "has_v2": HAS_V2,
            "has_mtcnn": HAS_MTCNN,
        }
        return render(request, index_template_name, context)
    else:
        upload_type = request.POST.get('upload_type', 'video')
        model_version = request.POST.get('model_version', 'v1')

        if upload_type == 'image':
            return _handle_image_upload(request, model_version)
        else:
            return _handle_video_upload(request, model_version)


def _handle_video_upload(request, model_version):
    """Handle video file upload."""
    video_upload_form = VideoUploadForm(request.POST, request.FILES)
    image_upload_form = ImageUploadForm()

    if video_upload_form.is_valid():
        video_file = video_upload_form.cleaned_data['upload_video_file']
        video_file_ext = video_file.name.split('.')[-1]
        sequence_length = video_upload_form.cleaned_data.get('sequence_length') or 20

        video_content_type = video_file.content_type.split('/')[0]
        if video_content_type in settings.CONTENT_TYPES:
            if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                return render(request, index_template_name, {
                    "video_form": video_upload_form,
                    "image_form": image_upload_form,
                    "has_v2": HAS_V2,
                })

        if not allowed_video_file(video_file.name):
            video_upload_form.add_error("upload_video_file", "Only video files are allowed")
            return render(request, index_template_name, {
                "video_form": video_upload_form,
                "image_form": image_upload_form,
                "has_v2": HAS_V2,
            })

        saved_video_file = 'uploaded_file_' + str(int(time.time())) + "." + video_file_ext
        save_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
        with open(save_path, 'wb') as vFile:
            shutil.copyfileobj(video_file, vFile)

        request.session['file_name'] = save_path
        request.session['sequence_length'] = sequence_length
        request.session['upload_type'] = 'video'
        request.session['model_version'] = model_version
        return redirect('ml_app:predict')
    else:
        return render(request, index_template_name, {
            "video_form": video_upload_form,
            "image_form": image_upload_form,
            "has_v2": HAS_V2,
        })


def _handle_image_upload(request, model_version):
    """Handle image file upload."""
    image_upload_form = ImageUploadForm(request.POST, request.FILES)
    video_upload_form = VideoUploadForm()

    if image_upload_form.is_valid():
        image_file = image_upload_form.cleaned_data['upload_image_file']
        image_file_ext = image_file.name.split('.')[-1]

        if not allowed_image_file(image_file.name):
            image_upload_form.add_error("upload_image_file", "Only image files are allowed")
            return render(request, index_template_name, {
                "video_form": video_upload_form,
                "image_form": image_upload_form,
                "has_v2": HAS_V2,
            })

        saved_image_file = 'uploaded_image_' + str(int(time.time())) + "." + image_file_ext
        save_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', saved_image_file)
        with open(save_path, 'wb') as iFile:
            shutil.copyfileobj(image_file, iFile)

        request.session['file_name'] = save_path
        request.session['upload_type'] = 'image'
        request.session['model_version'] = model_version
        return redirect('ml_app:predict')
    else:
        return render(request, index_template_name, {
            "video_form": video_upload_form,
            "image_form": image_upload_form,
            "has_v2": HAS_V2,
        })


def predict_page(request):
    """Prediction results page — handles both video and image."""
    if request.method != "GET":
        return redirect("ml_app:home")

    if 'file_name' not in request.session:
        return redirect("ml_app:home")

    file_path = request.session['file_name']
    upload_type = request.session.get('upload_type', 'video')
    model_version = request.session.get('model_version', 'v1')
    sequence_length = request.session.get('sequence_length', 20)

    file_name = os.path.basename(file_path)
    file_name_only = os.path.splitext(file_name)[0]

    start_time = time.time()

    if upload_type == 'image':
        return _predict_image(request, file_path, file_name_only, model_version, start_time)
    else:
        return _predict_video(request, file_path, file_name_only, model_version,
                              sequence_length, start_time)


def _predict_image(request, file_path, file_name_only, model_version, start_time, _force_device=None):
    """Run deepfake detection on a single image."""
    global device
    _device = _force_device if _force_device else device
    # Free any cached GPU memory before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            return render(request, predict_template_name, {"error": "Could not read image file"})

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect and crop face
        face, bbox = detect_face(rgb_image)
        if face is None:
            return render(request, predict_template_name, {"no_faces": True, "upload_type": "image"})

        # Save the original and face-cropped images for display
        original_image_name = f"{file_name_only}_original.png"
        original_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', original_image_name)
        pImage.fromarray(rgb_image).save(original_path)

        cropped_face_name = f"{file_name_only}_face.png"
        cropped_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', cropped_face_name)
        pImage.fromarray(face).save(cropped_path)

        # Choose model and predict
        use_image_model = False  # set True when image-specific model is loaded
        if model_version == 'v2' and HAS_V2:
            transform = v2_transforms
            face_tensor = transform(face).unsqueeze(0)  # (1, C, H, W)

            path_to_model = get_accurate_model(20, model_version='v2')
            if not path_to_model:
                path_to_model = get_accurate_model(20, model_version='v1')
                model_version = 'v1'  # fallback

            if model_version == 'v2':
                model = load_model_v2(path_to_model, device=_device)
                result = predict_v2(model, face_tensor, mode='image')
                prediction = result['prediction']
                confidence = round(result['confidence'], 1)
            else:
                # Fallback to v1
                model_version = 'v1'

        if model_version == 'v1':
            import gc

            # Prefer image-specific model (trained on Kaggle diverse dataset, 96% val acc)
            path_to_model = get_accurate_model(20, model_version='image')
            use_image_model = path_to_model is not None
            if not path_to_model:
                path_to_model = get_accurate_model(20, model_version='v1')
            if not path_to_model:
                return render(request, predict_template_name, {"error": "No model file found"})

            if use_image_model:
                # Image model: ResNeXt-50 + custom fc, input (1, C, 224, 224)
                # Keys: conv1.*, layer1-4.*, fc.1.*, fc.4.*
                transform = image_transforms
                face_tensor = transform(face).unsqueeze(0)
                model = build_image_model(2).to(_device)
                model.load_state_dict(torch.load(path_to_model, map_location=torch.device(_device), weights_only=False))
                model.eval()
                with torch.no_grad():
                    logits = model(face_tensor.to(_device))
                sm_out = nn.Softmax(dim=1)(logits)
                _, pred_t = torch.max(sm_out, 1)
                prediction = int(pred_t.item())
                confidence = round(float(sm_out[0, prediction].item()) * 100, 1)
            else:
                # Fallback: V1 video model with single frame as sequence of length 1
                transform = train_transforms
                face_tensor = transform(face).unsqueeze(0).unsqueeze(0)
                model = Model(2).to(_device)
                model.load_state_dict(torch.load(path_to_model, map_location=torch.device(_device), weights_only=False))
                model.eval()
                result = predict_v1(model, face_tensor)
                prediction = result[0]
                confidence = round(result[1], 1)

            del model
            gc.collect()

        output = "REAL" if prediction == 1 else "FAKE"
        elapsed = round(time.time() - start_time, 2)

        display_version = 'IMAGE' if use_image_model else model_version.upper()
        context = {
            'upload_type': 'image',
            'original_image': original_image_name,
            'face_image': cropped_face_name,
            'output': output,
            'confidence': confidence,
            'model_version': display_version,
            'elapsed_time': elapsed,
        }
        return render(request, predict_template_name, context)

    except Exception as e:
        err_str = str(e).lower()
        _oom_types = (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, 'OutOfMemoryError') else ()
        is_gpu_oom = isinstance(e, _oom_types) or 'out of memory' in err_str or 'cuda out of memory' in err_str
        if is_gpu_oom and _force_device != 'cpu':
            print(f"[WARN] CUDA OOM on image, retrying on CPU: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()
            return _predict_image(request, file_path, file_name_only, model_version, start_time, _force_device='cpu')
        print(f"Image prediction error: {e}")
        import traceback
        traceback.print_exc()
        return render(request, predict_template_name,
                      {'error': f'Prediction failed: {e}', 'upload_type': 'image'})


def _predict_video(request, video_file, file_name_only, model_version,
                    sequence_length, start_time, _force_device=None):
    """Run deepfake detection on a video."""
    import gc
    global device
    _device = _force_device if _force_device else device

    # On CPU-only machines cap frames to prevent RAM exhaustion / segfault
    if _device == 'cpu' and sequence_length > CPU_MAX_FRAMES:
        print(f"[INFO] CPU mode: capping frames {sequence_length} -> {CPU_MAX_FRAMES}")
        sequence_length = CPU_MAX_FRAMES

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    try:
        video_file_name = os.path.basename(video_file)
        path_to_videos = [video_file]
        production_video_name = video_file_name

        # Choose transforms based on model version
        if model_version == 'v2' and HAS_V2:
            current_transforms = v2_transforms
        else:
            current_transforms = train_transforms
            model_version = 'v1'

        # Load validation dataset
        video_dataset = validation_dataset(
            path_to_videos, sequence_length=sequence_length,
            transform=current_transforms
        )

        # ── Extract preview frames (evenly spaced — never load all frames!) ──
        print("<== | Started Videos Splitting | ==>")
        preprocessed_images = []
        faces_cropped_images = []

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")

        # Sample only `sequence_length` frames evenly across the video
        num_preview = min(sequence_length, total_frames)
        if total_frames <= num_preview:
            sample_indices = list(range(total_frames))
        else:
            sample_indices = [int(i) for i in
                              np.linspace(0, total_frames - 1, num_preview)]

        # Read only the sampled frames — never store the whole video in RAM
        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        print(f"Using {len(frames)} sampled frames (out of {total_frames} total)")

        padding = 40
        faces_found = 0
        num_preview = min(sequence_length, len(frames))

        for i in range(num_preview):
            frame = frames[i]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image
            image_name = f"{file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            pImage.fromarray(rgb_frame, 'RGB').save(image_path)
            preprocessed_images.append(image_name)

            # Face detection
            face, bbox = detect_face(rgb_frame, padding=padding)
            if face is not None:
                face_rgb = pImage.fromarray(face, 'RGB')
                image_name = f"{file_name_only}_cropped_faces_{i+1}.png"
                image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
                face_rgb.save(image_path)
                faces_found += 1
                faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")

        if faces_found == 0:
            return render(request, predict_template_name, {"no_faces": True, "upload_type": "video"})

        # ── Prediction ──
        print("<=== | Started Prediction | ===>")
        output = ""
        confidence = 0.0
        attention_data = None

        if model_version == 'v2':
            path_to_model = get_accurate_model(sequence_length, model_version='v2')
            if not path_to_model:
                path_to_model = get_accurate_model(sequence_length, model_version='v1')
                model_version = 'v1'

        if model_version == 'v2':
            model = load_model_v2(path_to_model, device=_device)
            frames_tensor = video_dataset[0]  # (1, T, C, H, W)
            result = predict_v2(model, frames_tensor, mode='video')
            prediction = result['prediction']
            confidence = round(result['confidence'], 1)
            output = "REAL" if prediction == 1 else "FAKE"
            attention_data = result.get('attention_weights', None)
        else:
            path_to_model = get_accurate_model(sequence_length, model_version='v1')
            if not path_to_model:
                return render(request, predict_template_name, {"error": "No model file found"})

            model = Model(2).to(_device)
            model.load_state_dict(torch.load(path_to_model, map_location=torch.device(_device), weights_only=False))
            model.eval()

            result = predict_v1(model, video_dataset[0])
            prediction = result[0]
            confidence = round(result[1], 1)
            output = "REAL" if prediction == 1 else "FAKE"

            # Free model memory immediately after inference
            del model
            gc.collect()

        elapsed = round(time.time() - start_time, 2)
        print(f"Prediction: {prediction} == {output} Confidence: {confidence}")
        print(f"<=== | Prediction Done in {elapsed}s | ===>")

        # Prepare attention weights for template
        attention_list = []
        if attention_data is not None:
            try:
                attn = attention_data.squeeze().cpu().numpy().tolist()
                if isinstance(attn, list):
                    attention_list = [round(float(a), 4) for a in attn]
            except Exception:
                pass

        context = {
            'upload_type': 'video',
            'preprocessed_images': preprocessed_images,
            'faces_cropped_images': faces_cropped_images,
            'original_video': production_video_name,
            'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
            'output': output,
            'confidence': confidence,
            'model_version': model_version.upper(),
            'elapsed_time': elapsed,
            'total_frames': total_frames,
            'analyzed_frames': sequence_length,
            'faces_found': faces_found,
            'attention_weights': json.dumps(attention_list),
        }
        return render(request, predict_template_name, context)

    except Exception as e:
        err_str = str(e).lower()
        _oom_types = (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, 'OutOfMemoryError') else ()
        is_gpu_oom = isinstance(e, _oom_types) or 'out of memory' in err_str or 'cuda out of memory' in err_str
        if is_gpu_oom and _force_device != 'cpu':
            print(f"[WARN] CUDA OOM on video, retrying on CPU: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()
            return _predict_video(request, video_file, file_name_only,
                                  model_version, sequence_length, start_time, _force_device='cpu')
        if _force_device == 'cpu' and ('memory' in err_str or 'cannot allocate' in err_str or 'out of memory' in err_str):
            return render(request, predict_template_name, {
                'error': 'Not enough RAM to process this video on CPU. Try uploading a shorter video or reducing the frame count.'
            })
        print(f"Exception occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return render(request, 'cuda_full.html')


def about(request):
    return render(request, about_template_name)


def handler404(request, exception):
    return render(request, '404.html', status=404)


def cuda_full(request):
    return render(request, 'cuda_full.html')
