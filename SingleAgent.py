##########################################
# Final Single-Agent RL in FL (Updated Final Version)
# * Multi-Objective Reward
# * LOCAL_EPOCHS = 2
# * ATTACK_SEVERITY = -0.2
# * Unfreeze all layers (full model fine-tuning)
# * No final global clamp (FINAL_CLAMP = None)
# * Soft-update for g_w using TAU
##########################################

!pip install -q kaggle

import os
import random
import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# (Optional) Kaggle setup – remove these lines if dataset is already downloaded
from google.colab import files
print("Please upload your kaggle.json file (Kaggle API token).")
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d lukechugh/best-alzheimer-mri-dataset-99-accuracy
!unzip -q best-alzheimer-mri-dataset-99-accuracy.zip -d dataset

##########################################
# 1) Hyperparameters and Basic Settings
##########################################
GLOBAL_EPOCHS = 40
LOCAL_EPOCHS = 2             # Local training epochs per client
NUM_CLIENTS = 5
FRACTION_MALICIOUS = 0.2
ATTACK_SEVERITY = -0.2
ROOT_DATASET_SIZE = 80
BATCH_SIZE = 8
LR = 1e-4
ALPHA = 1.0

DDPG_ACTOR_LR = 2e-4
DDPG_CRITIC_LR = 2e-4
GAMMA = 0.99
TAU = 0.01
REPLAY_CAPACITY = 1000

BATCH_SIZE_RL = 4
DDPG_UPDATES_PER_EPOCH = 50

PRINT_DETAILS = True

# Metric normalization parameters
BETA = 10.0
RE_MAX = 2.0
COS_THRESHOLD = 0.2
NUM_FEATURES = 4    # [RE, cos(g, g0), cos(g, g_w), Shapley]

# Clamp thresholds for local gradients:
CLAMP_HEALTHY = 10.0
CLAMP_CLIENT  = 50.0

# No final clamp:
FINAL_CLAMP = None

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################
# 2) Dataset Loading (IID)
##########################################
data_dir = 'dataset/Combined Dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir  = os.path.join(data_dir, 'test')

img_size = (224,224)
transform_train = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_train = datasets.ImageFolder(train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

classes = full_train.classes
num_classes = len(classes)
print("Detected Classes:", classes)

val_ratio = 0.2
val_size = int(val_ratio * len(full_train))
train_size = len(full_train) - val_size
train_for_clients, val_dataset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

def split_dataset_iid(dataset, num_clients):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    splitted = np.array_split(idxs, num_clients)
    return [Subset(dataset, s.tolist()) for s in splitted]

def create_root_dataset(dataset, size=80):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    return Subset(dataset, idxs[:size])

def remove_root_from_clients(client_subs, root_sub):
    root_idxs = set(root_sub.indices)
    new_subs = []
    for cs in client_subs:
        c_idx = [ix for ix in cs.indices if ix not in root_idxs]
        new_subs.append(Subset(cs.dataset, c_idx))
    return new_subs

client_datasets = split_dataset_iid(train_for_clients, NUM_CLIENTS)
root_dataset = create_root_dataset(train_for_clients, ROOT_DATASET_SIZE)
client_datasets = remove_root_from_clients(client_datasets, root_dataset)

##########################################
# 3) Global Model (Unfreeze All layers)
##########################################
def create_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    # Unfreeze all layers (fine-tuning whole network)
    for p in model.parameters():
        p.requires_grad = True
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model.to(device)

##########################################
# 4) Local Update Function
##########################################
def client_update(model_loc, opt_loc, loader, epochs=1):
    crit = nn.CrossEntropyLoss()
    model_loc.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            opt_loc.zero_grad()
            out = model_loc(data)
            loss = crit(out, target)
            loss.backward()
            opt_loc.step()
    return model_loc.state_dict()

##########################################
# 5) Evaluation Function
##########################################
def evaluate(model_g, dataset_subset):
    model_g.eval()
    loader = DataLoader(dataset_subset, batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model_g(data)
            _, pred = torch.max(out, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return 1 - (correct / total)

##########################################
# 6) Flatten and Overwrite Functions
##########################################
def flatten_params_cpu(model):
    pl = []
    for p in model.parameters():
        pl.append(p.data.detach().cpu().view(-1))
    return torch.cat(pl, dim=0)

def overwrite_model_params_from_cpu(model, flat_cpu):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            sz = p.numel()
            chunk = flat_cpu[idx: idx+sz].view(p.shape)
            p.data.copy_(chunk.to(device))
            idx += sz

##########################################
# 7) Gradient VAE and Training
##########################################
class GradientVAE(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc21 = nn.Linear(128, 64)
        self.fc22 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, input_size)
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv

def vae_loss(recon_x, x, mu, logvar):
    rec_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec_loss + kld

def train_vae(vae, grad_list, proj_mat, epochs=10):
    ds_list = []
    for g in grad_list:
        prj = torch.matmul(g, proj_mat)
        ds_list.append(prj)
    ds_t = torch.stack(ds_list, dim=0)
    optv = optim.Adam(vae.parameters(), lr=1e-3)
    vae.to('cpu')
    vae.train()
    bs = BATCH_SIZE
    for ep in range(epochs):
        perm = torch.randperm(ds_t.size(0))
        tot_loss = 0
        for i in range(0, ds_t.size(0), bs):
            idxs = perm[i:i+bs]
            batch = ds_t[idxs]
            optv.zero_grad()
            recon, mu, lv = vae(batch)
            loss_v = vae_loss(recon, batch, mu, lv)
            loss_v.backward()
            optv.step()
            tot_loss += loss_v.item()
        print(f"[VAE] Epoch {ep+1}/{epochs}, Loss= {tot_loss:.2f}")
    vae.eval()

##########################################
# 8) Projection Matrix and Gradient Projection
##########################################
def create_projection_matrix(model, r=128):
    total_dim = sum(p.numel() for p in model.parameters())
    pm = torch.randn(total_dim, r)
    pm = pm / pm.norm(dim=0, keepdim=True)
    return pm.cpu()

def project_gradient(grad_vec, pm):
    return torch.matmul(grad_vec, pm)

##########################################
# 9) Replay Buffer (with shape fix)
##########################################
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
    def push(self, s, a, r, sn):
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        if len(s.shape) > 1:
            s = s.reshape(-1)
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if len(a.shape) > 1:
            a = a.reshape(-1)
        if isinstance(sn, torch.Tensor):
            sn = sn.detach().cpu().numpy()
        if len(sn.shape) > 1:
            sn = sn.reshape(-1)
        r_ = float(r)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r_, sn))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

##########################################
# 10) Actor & Critic for DDPG
##########################################
class ActorDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, noise_std=0.05):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.noise_std = noise_std
        self.min_val = 0.1
    def forward(self, s, add_noise=False):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        if add_noise:
            noise = torch.normal(mean=0.0, std=self.noise_std, size=out.size(), device=out.device)
            out = out + noise
        out = torch.softmax(out, dim=1)
        out = out + self.min_val
        out = out / out.sum(dim=1, keepdim=True)
        return out

class CriticDDPG(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def soft_update(target_net, source_net, tau=0.01):
    for tp, p in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

##########################################
# 11) Transformations & Aggregator
##########################################
def aggregator_exp(scores, beta):
    exps = torch.exp(beta * scores)
    sumex = torch.sum(exps) + 1e-8
    return exps / sumex

def trans_RE(x):
    x = max(0, x)
    x = min(x, RE_MAX)
    return 1.0 - (x / (RE_MAX + 1e-8))

def trans_cos(x):
    val = (x + 1) / 2
    if val < COS_THRESHOLD:
        return 0.0
    else:
        return (val - COS_THRESHOLD) / (1 - COS_THRESHOLD)

def trans_shapley(shap_val):
    shap_val = max(shap_val, 0.0)
    shap_log = math.log(shap_val + 1.0)
    if shap_log > 5.0:
        shap_log = 5.0
    out = 1.0 - (shap_log / 5.0)
    return out if out >= 0.0 else 0.0

##########################################
# 12) Multi-Objective Reward Function
##########################################
def compute_multi_reward(prev_val_err, curr_val_err,
                         cos_ave_prev, cos_ave,
                         re_ave_prev, re_ave):
    main_factor = 3.0  # reduced to balance internal signals
    cos_factor = 0.2
    re_factor = 0.2
    delta_val = (prev_val_err - curr_val_err)
    reward = main_factor * delta_val
    if abs(delta_val) < 1e-8:
        reward -= 1
    elif delta_val < 0:
        reward += delta_val * 5
    dcos = cos_ave - cos_ave_prev
    reward += cos_factor * dcos
    dre = re_ave - re_ave_prev
    reward -= re_factor * dre
    return reward

##########################################
# 13) Final Federated Learning Loop
##########################################
def federated_learning_single_agent():
    global_model = create_model()
    global_model.train()

    num_malicious = max(1, int(NUM_CLIENTS * FRACTION_MALICIOUS))
    malicious_clients = random.sample(range(NUM_CLIENTS), num_malicious)
    print("Malicious clients:", malicious_clients)

    # Compute reference gradient g0 using root dataset
    root_model = copy.deepcopy(global_model)
    root_loader = DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True)
    crit = nn.CrossEntropyLoss()
    root_opt = optim.SGD(root_model.parameters(), lr=LR)

    root_grads = []
    for _ in range(LOCAL_EPOCHS):
        for data, target in root_loader:
            data, target = data.to(device), target.to(device)
            root_opt.zero_grad()
            out = root_model(data)
            loss = crit(out, target)
            loss.backward()
            grad_list = []
            for p in root_model.parameters():
                if p.grad is not None:
                    grad_list.append(p.grad.detach().cpu().view(-1))
                else:
                    grad_list.append(torch.zeros_like(p.data.view(-1)))
            gall = torch.cat(grad_list, dim=0)
            gall = gall / (gall.norm() + 1e-8)
            root_grads.append(gall)
            root_opt.step()

    g0 = root_grads[-1].clone()

    # Train VAE on reference gradients
    tmp_model = create_model()
    proj_mat = create_projection_matrix(tmp_model, r=128)
    del tmp_model

    vae = GradientVAE(input_size=128)
    train_vae(vae, root_grads, proj_mat, epochs=10)

    # Setup DDPG networks
    state_dim = NUM_CLIENTS * NUM_FEATURES
    action_dim = NUM_FEATURES
    actor = ActorDDPG(state_dim, action_dim, noise_std=0.05).to(device)
    critic = CriticDDPG(state_dim, action_dim).to(device)
    actor_targ = copy.deepcopy(actor)
    critic_targ = copy.deepcopy(critic)
    actor_opt = optim.Adam(actor.parameters(), lr=DDPG_ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=DDPG_CRITIC_LR)
    replay_buf = ReplayBuffer(REPLAY_CAPACITY)

    def evaluate_val(m):
        return evaluate(m, val_dataset)
    def evaluate_test(m):
        return evaluate(m, test_dataset)

    cos_ave_prev = 0.0
    re_ave_prev = 0.0
    prev_val_err = 1.0
    s_old, a_old = None, None
    val_list = []
    test_list = []
    rew_list = []

    # Initialize g_w as g0
    g_w = g0.clone()

    for ep in range(GLOBAL_EPOCHS):
        print(f"\n--- Global Epoch {ep+1}/{GLOBAL_EPOCHS}, ReplaySize= {len(replay_buf)} ---")
        global_flat = flatten_params_cpu(global_model)
        client_grads = []
        raw_feats = []

        for i in range(NUM_CLIENTS):
            cmodel = create_model()
            overwrite_model_params_from_cpu(cmodel, global_flat)
            copt = optim.SGD(cmodel.parameters(), lr=LR)
            loader_i = DataLoader(client_datasets[i], batch_size=BATCH_SIZE, shuffle=True)

            for _ in range(LOCAL_EPOCHS):
                for data, target in loader_i:
                    data, target = data.to(device), target.to(device)
                    copt.zero_grad()
                    out = cmodel(data)
                    loss_val = crit(out, target)
                    loss_val.backward()
                    copt.step()

            if i in malicious_clients:
                for p in cmodel.parameters():
                    p.data = p.data * ATTACK_SEVERITY

            local_flat = flatten_params_cpu(cmodel)
            diff = global_flat - local_flat

            # Clamp healthy gradients
            if CLAMP_HEALTHY:
                norm_d = diff.norm()
                if norm_d > CLAMP_HEALTHY:
                    factor = CLAMP_HEALTHY / (norm_d + 1e-8)
                    diff = diff * factor

            # If client gradient norm exceeds CLAMP_CLIENT, set to zero
            if CLAMP_CLIENT:
                norm_d = diff.norm()
                if norm_d > CLAMP_CLIENT:
                    print(f"Client {i}: gradient norm {norm_d:.2f}, set to zero.")
                    diff = torch.zeros_like(diff)

            client_grads.append(diff)

            with torch.no_grad():
                prj = torch.matmul(diff, proj_mat)
                recon, mu, lv = vae(prj.unsqueeze(0))
            re_val = nn.functional.mse_loss(recon, prj.unsqueeze(0), reduction='sum').item()
            cos_g0_val = nn.functional.cosine_similarity(diff, g0, dim=0).item()
            cos_gw_val = nn.functional.cosine_similarity(diff, g_w, dim=0).item()
            shap_val = nn.functional.mse_loss(diff, g0, reduction='sum').item()
            if i in malicious_clients:
                shap_val *= 0.5

            print(f"Client {i}: GradNorm= {diff.norm():.4f}, RE= {re_val:.4f}, " +
                  f"cos_g0= {cos_g0_val:.4f}, cos_gw= {cos_gw_val:.4f}, shap= {shap_val:.4f}")

            raw_feats.append([re_val, cos_g0_val, cos_gw_val, shap_val])

        norm_feats = []
        cos_list = []
        re_list = []
        honest_count = 0

        for i, feats in enumerate(raw_feats):
            re_i, cg0, cgw, shp = feats
            re_n = trans_RE(re_i)
            cg0_n = trans_cos(cg0)
            cgw_n = trans_cos(cgw)
            shp_n = trans_shapley(shp)
            norm_feats.extend([re_n, cg0_n, cgw_n, shp_n])
            if i not in malicious_clients:
                cos_list.append(cg0)
                re_list.append(re_i)
                honest_count += 1

        cos_ave = sum(cos_list) / honest_count if honest_count > 0 else 0.0
        re_ave = sum(re_list) / honest_count if honest_count > 0 else 0.0

        s_new = torch.tensor(norm_feats, dtype=torch.float32, device=device).unsqueeze(0)

        actor.train()
        a_new = actor(s_new, add_noise=True)
        a_np = a_new.squeeze(0).detach().cpu().numpy()
        print("Actor Output:", a_np)

        trust_scores = []
        for i in range(NUM_CLIENTS):
            base_idx = i * NUM_FEATURES
            cmet = norm_feats[base_idx: base_idx + NUM_FEATURES]
            score_i = np.dot(a_np, np.array(cmet))
            if i in malicious_clients:
                score_i *= 0.3
            trust_scores.append(score_i)
        trust_scores_t = torch.tensor(trust_scores, dtype=torch.float32, device=device)
        weights = aggregator_exp(trust_scores_t, beta=BETA).cpu().numpy()
        print("Trust Weights:", weights)

        final_grad = torch.zeros_like(client_grads[0])
        for gi, w in zip(client_grads, weights):
            final_grad += gi * w

        if FINAL_CLAMP:
            norm_fg = final_grad.norm()
            if norm_fg > FINAL_CLAMP:
                print(f"final_grad clamp from {norm_fg:.2f} to {FINAL_CLAMP}.")
                final_grad = final_grad * (FINAL_CLAMP / (norm_fg + 1e-8))

        new_global = global_flat + ALPHA * final_grad
        overwrite_model_params_from_cpu(global_model, new_global)
        # Soft-update g_w: update g_w = TAU*final_grad + (1-TAU)*g_w
        g_w = TAU * final_grad + (1 - TAU) * g_w

        val_err = evaluate_val(global_model)
        tst_err = evaluate_test(global_model)

        reward = compute_multi_reward(prev_val_err, val_err,
                                      cos_ave_prev, cos_ave,
                                      re_ave_prev, re_ave)
        print(f"cos_ave= {cos_ave:.4f}, re_ave= {re_ave:.4f}, reward= {reward:.4f}")

        prev_val_err = val_err
        cos_ave_prev = cos_ave
        re_ave_prev = re_ave

        if ep > 0 and s_old is not None and a_old is not None:
            replay_buf.push(s_old, a_old, reward, s_new)

        s_old = s_new
        a_old = a_new

        total_c_loss = 0.0
        total_a_loss = 0.0
        for _ in range(DDPG_UPDATES_PER_EPOCH):
            if len(replay_buf) < BATCH_SIZE_RL:
                break
            batch_data = replay_buf.sample(BATCH_SIZE_RL)
            sb, ab, rb, snb = [], [], [], []
            for (sv, av, rv, snv) in batch_data:
                sb.append(sv)
                ab.append(av)
                rb.append(rv)
                snb.append(snv)
            sb = torch.tensor(np.array(sb), dtype=torch.float32, device=device)
            ab = torch.tensor(np.array(ab), dtype=torch.float32, device=device)
            rb = torch.tensor(np.array(rb), dtype=torch.float32, device=device).unsqueeze(1)
            snb = torch.tensor(np.array(snb), dtype=torch.float32, device=device)

            with torch.no_grad():
                a_next = actor_targ(snb, add_noise=False)
                q_next = critic_targ(snb, a_next)
                y = rb + GAMMA * q_next

            critic_opt.zero_grad()
            q_curr = critic(sb, ab)
            loss_c = nn.functional.mse_loss(q_curr, y)
            loss_c.backward()
            critic_opt.step()
            total_c_loss += loss_c.item()

            actor_opt.zero_grad()
            a_curr = actor(sb, add_noise=False)
            q_val = critic(sb, a_curr)
            loss_a = -q_val.mean()
            loss_a.backward()
            actor_opt.step()
            total_a_loss += loss_a.item()

            soft_update(actor_targ, actor, tau=TAU)
            soft_update(critic_targ, critic, tau=TAU)

        val_list.append(val_err)
        test_list.append(tst_err)
        rew_list.append(reward)

        mean_c_loss = total_c_loss / (DDPG_UPDATES_PER_EPOCH + 1e-8)
        mean_a_loss = total_a_loss / (DDPG_UPDATES_PER_EPOCH + 1e-8)
        print(f"[Ep {ep+1}/{GLOBAL_EPOCHS}] ValErr= {val_err:.4f}, TestErr= {tst_err:.4f},",
              f"Reward= {reward:.4f}, ActorLoss= {mean_a_loss:.4f}, CriticLoss= {mean_c_loss:.4f}")

    f_test = evaluate_test(global_model)
    f_acc = (1 - f_test) * 100
    print(f"\n>>> Final Test Error= {f_test:.4f}, Final Acc= {f_acc:.2f}%")

    epochs_range = range(GLOBAL_EPOCHS)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, val_list, marker='o', label="ValErr")
    plt.plot(epochs_range, test_list, marker='x', label="TestErr")
    plt.xlabel("Global Epoch")
    plt.ylabel("Error Rate")
    plt.title("Validation & Test Error")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, rew_list, marker='o', label="Reward", color='green')
    plt.xlabel("Global Epoch")
    plt.ylabel("Reward")
    plt.title("Reward Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

##########################################
# 14) Run the Final Loop
##########################################
if __name__ == "__main__":
    federated_learning_single_agent()
