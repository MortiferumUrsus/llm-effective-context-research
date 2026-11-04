# Third-party notices

Original project code and documentation are provided under the MIT license in `LICENSE`. This grant does not replace upstream terms for datasets, prompts, dependencies, or other third-party material.

| Material | Source and attribution | Terms and treatment |
|---|---|---|
| BABILong | Yuri Kuratov and coauthors; [project](https://github.com/booydar/babilong), [paper](https://arxiv.org/abs/2406.10149) | Upstream project Apache-2.0; license retained in `licenses/BABILong-Apache-2.0.txt`. The release contains derived observation records and source IDs, not the complete source corpus. |
| Banking77 | Inigo Casanueva and coauthors, PolyAI; [dataset card](https://huggingface.co/datasets/PolyAI/banking77), [paper](https://arxiv.org/abs/2003.04807) | CC BY 4.0. Source IDs and labels are used in derived evaluation records. The complete corpus is not redistributed. The observation export adds predictions, scores, grouping metadata, and analysis fields. |
| ReAct prompt examples | Shunyu Yao and coauthors; [repository](https://github.com/ysymyth/ReAct), [paper](https://arxiv.org/abs/2210.03629) | MIT; upstream copyright and permission notice retained in `licenses/ReAct-MIT.txt`. The project uses a fixed-document tool backend and task adaptations. |
| MemGPT document prompt | Charles Packer and coauthors; [tag 0.2.7](https://github.com/letta-ai/letta/tree/033d9d61f4e646193abe4f7699b1bf76c7f759f7), [paper](https://arxiv.org/abs/2310.08560) | Apache-2.0 at the pinned revision; license retained in `licenses/MemGPT-Apache-2.0.txt`. The local system is a simplified archival-search loop with additional task instructions. |
| BGE-small-en-v1.5 | BAAI; [model card](https://huggingface.co/BAAI/bge-small-en-v1.5) | Used by the experiment implementation. Weights and tokenizer assets are not bundled. Consult the model card for their terms. |

BABILong itself combines bAbI tasks and background text. The release does not claim ownership of those underlying materials. Recovering the complete benchmark requires obtaining the upstream data under its applicable terms. A project software license does not relicense that data.

The observation export preserves recorded model answer strings and adds a documented field allowlist and recovered task identities. Provider error bodies, account information, and full prompt/source corpora are excluded. API output strings are supplied as research observations; the software license does not assert rights beyond those available in the underlying materials.

Dependency packages are obtained separately and retain their own licenses. No model weights, third-party papers, course templates, private credentials, or unrelated documents are included in this release.
