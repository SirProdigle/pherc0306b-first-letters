"""Community baseline: LimeGS/herculaneum-legibility-proxy-v5 (ResNet-18 on ~1 cm ink-map windows, 'stretch160').
Weights loaded with torch.load(weights_only=True) after SHA-256 check; their code is NOT executed (preprocessing
re-implemented from their preprocess.py). Fed our a-maps (x255) on 1 cm (200 px at 50 um) windows centred on the
same window centres as our detector."""
import hashlib, json
import numpy as np, cv2, torch, torchvision
from common import *

D = f'{G}/ext/legib_dl'


class Legib:
    def __init__(self, device='cuda'):
        meta = json.load(open(f'{D}/model_metadata.json'))
        h = hashlib.sha256(open(f'{D}/final.pt', 'rb').read()).hexdigest()
        assert h == meta['checkpoint_sha256'], 'checkpoint hash mismatch'
        st = torch.load(f'{D}/final.pt', map_location='cpu', weights_only=True)
        m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Linear(512, 1)
        m.load_state_dict(st, strict=True)
        self.m = m.to(device).eval(); self.dev = device

    @staticmethod
    def prep(u8):
        x = u8.astype(np.float32)
        act = x[x > 10]
        if act.size > 50:
            lo, hi = np.percentile(act, [2, 99.5]); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            x = x / 255.0
        return cv2.resize((x * 255).astype(np.uint8), (160, 160), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255

    @torch.no_grad()
    def score(self, it, ws, bs=128):
        a = it['a']
        a8 = a if a.dtype == np.uint8 else np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)
        a8 = a8 * it['mask']
        P = 100
        pad = np.pad(a8, P)
        X = np.stack([self.prep(pad[y + WIN // 2: y + WIN // 2 + 2 * P, x + WIN // 2: x + WIN // 2 + 2 * P]) for y, x in ws]) if ws else np.zeros((0, 160, 160), np.float32)
        out = []
        for i in range(0, len(X), bs):
            t = torch.from_numpy(X[i:i + bs, None]).repeat(1, 3, 1, 1).to(self.dev)
            out.append(self.m(t)[:, 0].float().cpu())
        return torch.cat(out).numpy() if out else np.zeros(0, np.float32)
