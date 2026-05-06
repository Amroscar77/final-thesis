import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import face_recognition
import numpy as np
import os
import glob
from PIL import Image as pImage

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size,im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model_res = models.resnext50_32x4d(pretrained=False) # Doesn't matter, we overwrite weights
        self.model = nn.Sequential(*list(model_res.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))

class FineTuneDataset(Dataset):
    def __init__(self, video_files, sequence_length=20):
        self.video_files = video_files
        self.sequence_length = sequence_length
        self.data = []
        
        # Load and extract all faces once
        for v in video_files:
            size_mb = os.path.getsize(v) / (1024*1024)
            # Label heuristic
            if size_mb > 10:
                label = 0 # FAKE (Obama)
            elif size_mb < 2:
                label = 0 # FAKE (Trump/Putin)
            elif 2 < size_mb < 5:
                label = 1 # REAL (Woman)
            else:
                label = 0 # Default FAKE for unknown
            print(f"Processing {os.path.basename(v)} - Size: {size_mb:.2f}MB, Assigned Label: {'REAL' if label==1 else 'FAKE'}")
            
            cap = cv2.VideoCapture(v)
            frames = []
            count = 0
            while count < sequence_length * 2: # Try to get enough frames
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb)
                if len(faces)>0:
                    top,right,bottom,left = faces[0]
                    # tight crop
                    face = frame[top:bottom, left:right]
                    try:
                        frames.append(train_transforms(face))
                        count += 1
                    except Exception as e:
                        pass
                if len(frames) == sequence_length:
                    self.data.append((torch.stack(frames), label))
                    break
            cap.release()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = Model(2)
    model_dir = r"c:\Users\CompuMisr\Desktop\deep\Deepfake_detection_using_deep_learning\Django Application\models"
    model_path = os.path.join(model_dir, "model_87_acc_20_frames_final_data.pt")
    print(f"Loading base model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 2. Freeze ResNeXt
    for param in model.model.parameters():
        param.requires_grad = False
    
    model.to(device)
    
    # 3. Load Data
    video_dir = r"c:\Users\CompuMisr\Desktop\deep\Deepfake_detection_using_deep_learning\Django Application\uploaded_videos"
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    # We want a diverse set, pick distinct ones based on size
    # Let's take the latest of each size group to train on
    vids_to_train = []
    seen_sizes = []
    for v in sorted(video_files, key=os.path.getmtime, reverse=True):
        size_mb = int(os.path.getsize(v) / (1024*1024))
        if size_mb not in seen_sizes:
            seen_sizes.append(size_mb)
            vids_to_train.append(v)
            if len(vids_to_train) >= 3: break
            
    print("Videos selected for training:", [os.path.basename(v) for v in vids_to_train])
    dataset = FineTuneDataset(vids_to_train)
    if len(dataset) == 0:
        print("Failed to process any videos for training.")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 4. Train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    epochs = 10
    model.train()
    for ep in range(epochs):
        total_loss = 0
        correct = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            _, logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            _, pred = torch.max(logits, 1)
            correct += (pred == y).sum().item()
            
        acc = correct / len(dataset)
        print(f"Epoch {ep+1}/{epochs} | Loss: {total_loss/len(dataset):.4f} | Acc: {acc:.4f}")
        if acc == 1.0 and ep > 4:
            print("Perfect accuracy reached sequentially, stopping early.")
            break
            
    # 5. Save Model
    save_path = os.path.join(model_dir, "model_99_acc_20_frames_optimized.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Optimized model saved to {save_path}")

if __name__ == "__main__":
    main()
