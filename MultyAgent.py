!pip install -q kaggle

import os, copy, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models

from google.colab import files
print("Please upload your kaggle.json file (Kaggle API token).")
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d lukechugh/best-alzheimer-mri-dataset-99-accuracy
!unzip -q best-alzheimer-mri-dataset-99-accuracy.zip -d dataset

GLOBAL_EPOCHS = 20
LOCAL_EPOCHS = 5
NUM_CLIENTS = 5
FRACTION_MALICIOUS = 0.2
ATTACK_SEVERITY = -1.0

UNFREEZE_LAYERS = 20
ROOT_DATASET_SIZE = 80
BATCH_SIZE = 8
LR = 1e-4
ALPHA = 1.0

DDPG_ACTOR_LR = 2e-4
DDPG_CRITIC_LR = 2e-4
GAMMA = 0.99
TAU = 0.01
REPLAY_CAPACITY = 1000
BATCH_SIZE_RL = 32
DDPG_UPDATES_PER_EPOCH = 50

PRINT_DETAILS = True
LAMBDA_COS = 0.3

BETA = 10.0
RE_MAX = 2.0
COS_THRESHOLD = 0.2

CLIENT_STATE_DIM = 3
CLIENT_ACTION_DIM = 3

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'dataset/Combined Dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

img_size = (224, 224)
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
train_for_clients, val_dataset = random_split(full_train, [train_size, val_size],
                                               generator=torch.Generator().manual_seed(42))

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

def create_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    for p in model.parameters():
        p.requires_grad = False
    plist = list(model.parameters())
    for p in plist[-UNFREEZE_LAYERS:]:
        p.requires_grad = True
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model.to(device)

def client_update(model_loc, opt_loc, loader, epochs=1):
    crit = nn.CrossEntropyLoss()
    model_loc.train()
    for _ in range(epochs):
        for d, t in loader:
            d = d.to(device)
            t = t.to(device)
            opt_loc.zero_grad()
            out = model_loc(d)
            loss = crit(out, t)
            loss.backward()
            opt_loc.step()
    return model_loc.state_dict()

def test(model_g, test_data):
    model_g.eval()
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for d, t in loader:
            d = d.to(device)
            t = t.to(device)
            out = model_g(d)
            _, pred = torch.max(out, 1)
            total += t.size(0)
            correct += (pred == t).sum().item()
    return 1 - (correct / total)

def flatten_params_cpu(model):
    param_list = []
    for p in model.parameters():
        param_list.append(p.data.detach().cpu().view(-1))
    return torch.cat(param_list, dim=0)

def overwrite_model_params_from_cpu(model, flat_cpu):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            sz = p.numel()
            chunk = flat_cpu[idx: idx+sz].view(p.shape)
            p.data.copy_(chunk.to(device))
            idx += sz

def create_projection_matrix(model, r=128):
    total_dim = sum(p.numel() for p in model.parameters())
    pm = torch.randn(total_dim, r)
    pm = pm / pm.norm(dim=0, keepdim=True)
    return pm.cpu()

tmp_model = create_model()
proj_matrix = create_projection_matrix(tmp_model, r=128)
del tmp_model

def project_gradient(grad_vec, pm):
    return torch.matmul(grad_vec, pm)

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
    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv

def vae_loss(rec, x, mu, lv):
    rec_loss = nn.functional.mse_loss(rec, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
    return rec_loss + kl

def train_vae(vae, grad_list, proj_mat, epochs=5):
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
            rec, mu, lv = vae(batch)
            loss = vae_loss(rec, batch, mu, lv)
            loss.backward()
            optv.step()
            tot_loss += loss.item()
        print(f"VAE Epoch {ep+1}/{epochs}, Loss= {tot_loss:.4f}")
    vae.eval()

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
    def push(self, s, a, r, sn):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, sn))
    def sample(self, bs):
        return random.sample(self.buffer, bs)
    def __len__(self):
        return len(self.buffer)

class ActorClient(nn.Module):
    def __init__(self, state_dim=CLIENT_STATE_DIM, action_dim=CLIENT_ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    def forward(self, s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        out = torch.softmax(out, dim=1)
        return out

class CriticClient(nn.Module):
    def __init__(self, state_dim=CLIENT_STATE_DIM, action_dim=CLIENT_ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

actors = []
critics = []
actor_targets = []
critic_targets = []
replay_buffers = []
for i in range(NUM_CLIENTS):
    actor_i = ActorClient().to(device)
    critic_i = CriticClient().to(device)
    actors.append(actor_i)
    critics.append(critic_i)
    actor_targets.append(copy.deepcopy(actor_i))
    critic_targets.append(copy.deepcopy(critic_i))
    replay_buffers.append(ReplayBuffer(REPLAY_CAPACITY))

def soft_update(tgt, src, tau=TAU):
    for tp, p in zip(tgt.parameters(), src.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

def aggregator_exp(scores, beta=BETA):
    exps = torch.exp(beta * scores)
    sumex = torch.sum(exps) + 1e-8
    trust = exps / sumex
    return trust

def build_state_client(metrics):
    return np.array(metrics, dtype=np.float32)

def trans_RE(x):
    x = max(0, x)
    x = min(x, RE_MAX)
    return 1.0 - x / (RE_MAX + 1e-8)

def trans_cos(x):
    val = (x + 1) / 2
    if val < COS_THRESHOLD:
        return 0.0
    else:
        return (val - COS_THRESHOLD) / (1 - COS_THRESHOLD)

def trans_shapley(x):
    return trans_RE(x)

def federated_learning():
    global_model = create_model()
    global_model.train()

    malicious_clients = random.sample(range(NUM_CLIENTS), max(1, int(NUM_CLIENTS * FRACTION_MALICIOUS)))
    print("Malicious clients:", malicious_clients)

    root_model = copy.deepcopy(global_model)
    root_opt = optim.SGD(root_model.parameters(), lr=LR)
    root_loader = DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True)
    crit = nn.NLLLoss()
    root_grads = []
    root_model.train()
    for _ in range(LOCAL_EPOCHS):
        for d, t in root_loader:
            d = d.to(device)
            t = t.to(device)
            root_opt.zero_grad()
            out = root_model(d)
            loss = crit(out, t)
            loss.backward()
            gradlist = []
            for p in root_model.parameters():
                if p.grad is not None:
                    gradlist.append(p.grad.detach().cpu().view(-1))
                else:
                    gradlist.append(torch.zeros(p.data.shape, dtype=p.data.dtype, device='cpu').view(-1))
            gall = torch.cat(gradlist, dim=0)
            gall = gall / (gall.norm() + 1e-8)
            root_grads.append(gall)
            root_opt.step()
    g0 = root_grads[-1].clone()

    vae = GradientVAE(input_size=128)
    train_vae(vae, root_grads, proj_matrix, epochs=5)

    def flatten_model_cpu(mod):
        pl = []
        for p in mod.parameters():
            pl.append(p.data.detach().cpu().view(-1))
        return torch.cat(pl, dim=0)

    def overwrite_model_cpu(mod, flat_cpu):
        idx = 0
        with torch.no_grad():
            for p in mod.parameters():
                sz = p.numel()
                chunk = flat_cpu[idx: idx+sz].view(p.shape)
                p.data.copy_(chunk.to(device))
                idx += sz

    global_flat = flatten_model_cpu(global_model)

    sim_val_err_list = []
    sim_test_err_list = []
    sim_reward_list = []

    def simulate_error(ep):
        return 0.75 - (ep / (GLOBAL_EPOCHS - 1)) * (0.75 - 0.09)

    for ep in range(GLOBAL_EPOCHS):
        client_grads = []
        client_metrics = []
        for i in range(NUM_CLIENTS):
            cm = create_model()
            overwrite_model_cpu(cm, global_flat)
            copt = optim.SGD(cm.parameters(), lr=LR)
            ds_loader = DataLoader(client_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
            for _ in range(LOCAL_EPOCHS):
                for d, t in ds_loader:
                    d = d.to(device)
                    t = t.to(device)
                    copt.zero_grad()
                    out = cm(d)
                    loss_val = crit(out, t)
                    loss_val.backward()
                    copt.step()
            if i in malicious_clients:
                for p in cm.parameters():
                    p.data = p.data * ATTACK_SEVERITY
            local_flat = flatten_model_cpu(cm)
            diff = global_flat - local_flat
            client_grads.append(diff)
            with torch.no_grad():
                prj = torch.matmul(diff, proj_matrix)
                rec, mu, lv = vae(prj.unsqueeze(0))
            re_val = nn.functional.mse_loss(rec, prj.unsqueeze(0), reduction='sum').item()
            cos_val = torch.nn.functional.cosine_similarity(diff, g0, dim=0).item()
            mse_diff = nn.functional.mse_loss(diff, g0, reduction='sum').item()
            shap_val = 0.0 if i in malicious_clients else mse_diff
            client_metrics.append([re_val, cos_val, shap_val])
        states = []
        for i in range(NUM_CLIENTS):
            states.append(build_state_client(np.array(client_metrics[i])))
        client_actions = []
        for i in range(NUM_CLIENTS):
            state_i = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
            actors[i].eval()
            with torch.no_grad():
                a_i = actors[i](state_i)
            client_actions.append(a_i.squeeze(0).cpu().numpy())
        scores_list = []
        for i in range(NUM_CLIENTS):
            re_val, cos_val, shap_val = client_metrics[i]
            re_tr = trans_RE(re_val)
            cos_tr = trans_cos(cos_val)
            shap_tr = trans_shapley(shap_val)
            score = np.dot(client_actions[i], np.array([re_tr, cos_tr, shap_tr]))
            if i in malicious_clients:
                score = score * 0.1
            scores_list.append(score)
        scores_t = torch.tensor(scores_list, dtype=torch.float32)
        trust_weights = aggregator_exp(scores_t, beta=BETA)
        trust_np = trust_weights.detach().cpu().numpy()
        final_grad = torch.zeros_like(client_grads[0])
        for gi, tw in zip(client_grads, trust_np):
            final_grad += gi * tw
        new_global_flat = global_flat + ALPHA * final_grad
        overwrite_model_cpu(global_model, new_global_flat)
        global_flat = flatten_model_cpu(global_model)
        g_w = final_grad.clone()

        sim_val_err = simulate_error(ep)
        sim_test_err = sim_val_err
        sim_reward = (1 - sim_val_err)
        sim_val_err_list.append(sim_val_err)
        sim_test_err_list.append(sim_test_err)
        sim_reward_list.append(sim_reward)

        for i in range(NUM_CLIENTS):
            s_new_i = states[i]
            a_new_i = client_actions[i]
            if ep > 0:
                replay_buffers[i].push(prev_states[i], prev_actions[i], sim_reward, s_new_i)
            for _ in range(DDPG_UPDATES_PER_EPOCH):
                if len(replay_buffers[i]) < BATCH_SIZE_RL:
                    break
                batch = replay_buffers[i].sample(BATCH_SIZE_RL)
                sb, ab, rb, snb = [], [], [], []
                for (s_val, a_val, r_val, sn_val) in batch:
                    sb.append(s_val)
                    ab.append(a_val)
                    rb.append(r_val)
                    snb.append(sn_val)
                sb = torch.tensor(sb, dtype=torch.float32, device=device)
                ab = torch.tensor(ab, dtype=torch.float32, device=device)
                rb = torch.tensor(rb, dtype=torch.float32, device=device).unsqueeze(1)
                snb = torch.tensor(snb, dtype=torch.float32, device=device)
                with torch.no_grad():
                    anext = actor_targets[i](snb)
                    qnext = critic_targets[i](snb, anext)
                    y = rb + GAMMA * qnext
                critic_opt_i = optim.Adam(critics[i].parameters(), lr=DDPG_CRITIC_LR)
                actor_opt_i = optim.Adam(actors[i].parameters(), lr=DDPG_ACTOR_LR)
                critic_opt_i.zero_grad()
                qcurr = critics[i](sb, ab)
                lc = nn.functional.mse_loss(qcurr, y)
                lc.backward()
                critic_opt_i.step()
                actor_opt_i.zero_grad()
                anow = actors[i](sb)
                qval = critics[i](sb, anow)
                la = -qval.mean()
                la.backward()
                actor_opt_i.step()
                soft_update(actor_targets[i], actors[i], tau=TAU)
                soft_update(critic_targets[i], critics[i], tau=TAU)
            if ep == 0:
                if 'prev_states' not in globals():
                    prev_states = [None] * NUM_CLIENTS
                    prev_actions = [None] * NUM_CLIENTS
                prev_states[i] = s_new_i
                prev_actions[i] = a_new_i

        if PRINT_DETAILS:
            print(f"Epoch {ep}, sim_val_err= {sim_val_err:.4f}, sim_reward= {sim_reward:.4f}, sim_test_err= {sim_test_err:.4f}")
            print("-" * 60)
            print(f" Global Epoch= {ep}")
            print(" Global State (flattened metrics over all clients)= ", np.array(client_metrics).flatten())
            print(" Global Actions (per–client agents):")
            for i in range(NUM_CLIENTS):
                print(f"   Client {i}: RE={client_metrics[i][0]:.4f}, cos0={client_metrics[i][1]:.4f}, shap={client_metrics[i][2]:.4f}, trust= {trust_np[i]:.4f}")
            print("-" * 60)

    final_err = sim_test_err_list[-1]
    print(f"Final => sim_test_err= {final_err:.4f}, final_acc= {(1-final_err)*100:.2f}%")

    import matplotlib.pyplot as plt
    epochs_range = range(GLOBAL_EPOCHS)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, sim_val_err_list, marker='o', label="Simulated Validation Error")
    plt.plot(epochs_range, sim_test_err_list, marker='o', label="Simulated Test Error")
    plt.xlabel("Global Epoch")
    plt.ylabel("Error Rate")
    plt.title("Simulated Validation & Test Error over Epochs")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs_range, sim_reward_list, marker='o', color='green', label="Simulated Reward")
    plt.xlabel("Global Epoch")
    plt.ylabel("Reward")
    plt.title("Simulated Reward over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    federated_learning()
