import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
model_fpath = "/data1/qxwang/checkpoints/simcse-large-gsm8k"
tokenizer = AutoTokenizer.from_pretrained(model_fpath)
model = AutoModel.from_pretrained(model_fpath)
def compute_embeds_fn(texts, model, tokenizer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy()




import numpy as np
from collections import defaultdict
from typing import List, Dict, Union

class StepHashDict:
    def __init__(
        self,
        threshold: float = 0.7,
        rep_mode: str = "medoid",              # "first" | "centroid" | "medoid"
    ):
        self.dicts: Dict[int, Dict[int, dict]] = defaultdict(dict)
        self.threshold = threshold
        self.rep_mode = rep_mode.lower()
        assert self.rep_mode in {"first", "centroid", "medoid"}
        # first: use the first embedding inserted as the representative
        # centroid: use the mean of all embeddings in the cluster as the representative
        # medoid: use the most similar embedding to the centroid as the representative

    # ------------ 私有辅助函数 ------------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _build_rep_matrix(self, clusters: Dict[int, dict]) -> np.ndarray:
        """把当前所有 rep_embedding 拼成 (K, D) 矩阵；K=0 返回 None"""
        if not clusters:
            return None
        reps = [info["rep_embedding"] for info in clusters.values()]
        reps = np.vstack(reps).copy()
        reps.setflags(write=True)
        return reps             # (K, D)

    # ------------ 对外接口 ------------
    def update_sample_step_hash_dict(
        self,
        sample_id: int,
        embeddings: np.ndarray,   # shape (N, D) 已 L2 归一化
        texts: List[str],
        lead_correct_list: List[bool] = None
    ) -> Dict[int, dict]:
        """
        给定某个问题 (sample_id) 的多条 (embedding, text)，
        按余弦阈值进行在线聚类并返回该问题的 clusters。
        """
        assert len(embeddings) == len(texts), "embeddings 和 texts 数量不一致"
        #breakpoint()
        clusters = self.dicts[sample_id]            # 取引用
        rep_matrix = self._build_rep_matrix(clusters)

        for idx, (emb, txt) in enumerate(zip(embeddings, texts)):
            lead_to_correct = lead_correct_list[idx] if lead_correct_list else None
            # ---------- build a new rep_matrix is there's no existing clusters ----------
            
            if rep_matrix is None:                  # 第 1 次
                clusters[0] = dict(
                    rep_embedding=emb,
                    rep_text=txt,
                    members_texts=[txt],
                    members_idx=[idx],
                    member_embeddings=[emb],        # 仅为 centroid/medoid 时用
                    lead_to_correct=lead_to_correct
                )
                rep_matrix = emb[None, :].copy()         # (1, D)
                rep_matrix.setflags(write=True)
                continue

            # ---------- 计算与所有簇代表的余弦相似度 ----------
            sims = np.dot(rep_matrix, emb)          # (K,)
            best_row = int(np.argmax(sims))
            best_sim = float(sims[best_row])
            best_cid = best_row

            # ---------- insert into an existing cluster ----------
            if best_sim > self.threshold:
                cinfo = clusters[best_cid]
                cinfo["members_texts"].append(txt)
                cinfo["members_idx"].append(idx)
                cinfo["lead_to_correct"] = cinfo["lead_to_correct"] or lead_to_correct
                if self.rep_mode in {"centroid", "medoid"}:
                    cinfo["member_embeddings"].append(emb)

                # -------- 根据 rep_mode 动态更新代表向量 --------
                if self.rep_mode == "centroid":
                    new_rep = self._normalize(
                        np.mean(cinfo["member_embeddings"], axis=0)
                    )
                    cinfo["rep_embedding"] = new_rep
                elif self.rep_mode == "medoid":
                    centroid = np.mean(cinfo["member_embeddings"], axis=0)
                    sims_to_centroid = np.dot(
                        cinfo["member_embeddings"], centroid
                    )
                    best_idx = int(np.argmax(sims_to_centroid))
                    new_rep = cinfo["member_embeddings"][best_idx]
                    cinfo["rep_embedding"] = new_rep
                    cinfo["rep_text"] = cinfo["members_texts"][best_idx]

                # 同步更新 rep_matrix 中该行
                rep_matrix[best_row] = cinfo["rep_embedding"]

            else:  # --------- build a new cluster ---------
                new_cid = len(clusters)
                #print(new_cid)
                clusters[new_cid] = dict(
                    rep_embedding=emb,
                    rep_text=txt,
                    members_texts=[txt],
                    members_idx=[idx],
                    member_embeddings=[emb],
                    lead_to_correct=lead_to_correct
                )
                rep_matrix = np.vstack([rep_matrix, emb[None, :]]).copy()
                rep_matrix.setflags(write=True)


        return clusters
    
    def look_up_step_correctness(
        self,
        sample_id: int,
        texts: Union[str, List[str]]
    ) -> List[bool]:
        """
        按 *字符串* 精确匹配 members_texts：
        - 输入可以是单个 str，也可以是 str 列表。
        - 对于每个待查字符串，遍历该 sample 的所有簇，
          若在某簇 cinfo["members_texts"] 中找到完全一致的项，
          返回该簇的 lead_to_correct。
        - 若找不到，则抛 ValueError。
        """
        # 统一成列表
        if isinstance(texts, str):
            texts = [texts]

        clusters = self.dicts.get(sample_id, {})
        if not clusters:
            raise KeyError(f"No clusters found for sample_id {sample_id}")

        results: List[bool] = []

        for query in texts:
            found = False
            for cinfo in clusters.values():
                if query in cinfo["members_texts"]:
                    results.append(cinfo["lead_to_correct"])
                    found = True
                    break

            if not found:
                raise ValueError(
                    f'Text "{query}" not found in any cluster for sample_id {sample_id}'
                )

        return results
    
    
    def print_dict_info(self):
        """
        打印当前字典的统计信息。
        """
        print(f"Total samples: {len(self.dicts)}")
        for sample_id, clusters in self.dicts.items():
            print(f"Sample ID: {sample_id}, Clusters: {len(clusters)}")
            for cid, cinfo in clusters.items():
                print(f"  Cluster ID: {cid}, Members: {len(cinfo['members_texts'])}, Lead Correct: {cinfo['lead_to_correct']}")
                
texts = [" Properties of a Rhombus\n- Diagonals of a rhombus bisect each other at right angles. Therefore, \\(K\\) is the midpoint of both diagonals \\(FG\\) and \\(HJ\\) and diagonals \\(FK\\) and \\(KJ\\) are perpendicular to each other.\n- The diagonals of a rhombus also bisect the angles of the rhombus. So, \\( \\angle FKH = \\angle KHF = 41^\\circ \\) and similarly for the other angles.", "Properties of a Rhombus\nA rhombus has several important properties:\n- All sides are of equal length.\n- The diagonals bisect each other at right angles (\\(90^\\circ\\)).\n- The diagonals also bisect the angles of the rhombus.\n\n"]     
d = StepHashDict(threshold=0.7, rep_mode="medoid")
embeds = compute_embeds_fn(
    texts,
    model, tokenizer
)

d.update_sample_step_hash_dict(sample_id=1, embeddings=embeds, texts=texts, lead_correct_list=[True, False])
d.print_dict_info()