# -*- coding: utf-8 -*-
"""CarotKcat Web UI — 给不熟悉命令行的用户用的可视化预测页面。
启动: python app.py   然后浏览器打开 http://127.0.0.1:7860
"""
import re, tempfile
import numpy as np, pandas as pd, torch, torch.nn as nn, gradio as gr, esm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

print("加载 ESM-2 模型（首次运行会自动下载约2.5GB权重）...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.cuda().half().eval()
bc = alphabet.get_batch_converter()

class DualNet(nn.Module):
    def __init__(self, d_seq=1280, d_sub=2048):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(d_seq, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.GELU())
        self.sub = nn.Sequential(nn.Linear(d_sub, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.head(torch.cat([self.seq(x[:, :1280]), self.sub(x[:, 1280:])], 1)).squeeze(-1)

mlp = DualNet().cuda()
mlp.load_state_dict(torch.load("models/kcat_dual_randomsplit.pt", weights_only=False)["state_dict"])
mlp.eval()
print("CarotKcat 模型加载完成！")

def seq_emb(seq):
    _, _, toks = bc([("q", seq[:1022])])
    with torch.no_grad():
        out = esm_model(toks.cuda(), repr_layers=[33])["representations"][33]
    L = min(len(seq), 1022)
    return out[0, 1:L+1].float().mean(0).cpu().numpy()

def parse_fasta(text):
    recs, name, seq = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name: recs.append((name, "".join(seq)))
            name, seq = line[1:], []
        elif line:
            seq.append(re.sub(r"[^A-Z]", "", line.upper()))
    if name: recs.append((name, "".join(seq)))
    return recs

def predict(fasta_file, seq_text, smiles):
    if not smiles or Chem.MolFromSmiles(smiles.strip()) is None:
        return None, "❌ 请输入合法的底物 SMILES", None
    mol_img = Draw.MolToImage(Chem.MolFromSmiles(smiles.strip()))

    recs = []
    if fasta_file is not None:
        recs += parse_fasta(open(fasta_file.name, encoding="utf-8", errors="ignore").read())
    if seq_text.strip():
        s = seq_text.strip()
        recs += parse_fasta(s) if s.startswith(">") else [("手动输入序列", re.sub(r"[^A-Z]", "", s.upper()))]
    if not recs:
        return None, "❌ 请上传 FASTA 文件或粘贴序列", mol_img

    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles.strip()), 2, nBits=2048), dtype=np.float32)
    rows = []
    for name, seq in recs:
        if len(seq) < 20:
            rows.append([name, len(seq), None, None, "序列过短"])
            continue
        X = np.concatenate([seq_emb(seq), fp]).astype(np.float32)
        with torch.no_grad():
            logk = mlp(torch.tensor(X, device="cuda").unsqueeze(0)).item()
        rows.append([name[:60], len(seq), round(logk, 3), round(10**logk, 4), "OK"])
    df = pd.DataFrame(rows, columns=["酶", "序列长度", "log10(kcat)", "预测 kcat (s⁻¹)", "状态"])
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df, f"✅ 完成 {len(df)} 条预测，可下载 CSV", mol_img, out

with gr.Blocks(title="CarotKcat") as demo:
    gr.Markdown("# 🧬 CarotKcat — 酶催化常数 kcat 预测\n"
                "输入**酶序列**（FASTA 文件或直接粘贴）+ **底物 SMILES**，输出预测 kcat (s⁻¹)。\n"
                "模型：ESM-2 序列编码 + 底物分子指纹双塔网络，17,051 条通用酶数据训练，类胡萝卜素酶强化。")
    with gr.Row():
        with gr.Column():
            fasta = gr.File(label="上传 FASTA 文件（可选）", file_types=[".fasta", ".fa", ".txt"])
            seq = gr.Textbox(label="或直接粘贴序列 / FASTA 文本", lines=6,
                             placeholder=">my_enzyme\nMKTAYIAKQRQISFVKSHFSRQ...")
            smi = gr.Textbox(label="底物 SMILES", lines=1,
                             placeholder="例：β-胡萝卜素 CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/...")
            btn = gr.Button("🔬 开始预测", variant="primary")
            status = gr.Markdown()
        with gr.Column():
            mol_img = gr.Image(label="底物结构", height=220)
            table = gr.Dataframe(label="预测结果")
            dl = gr.File(label="下载结果 CSV")
    btn.click(predict, [fasta, seq, smi], [table, status, mol_img, dl])

demo.launch(server_name="127.0.0.1", server_port=7860)
