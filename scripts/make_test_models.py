"""Generate small *random-weight* GGUF models for pipeline testing.

The weights are random, so the text these models produce is meaningless.
They exist so the whole PocketLLM pipeline (GGUF parsing, planning, MoE
placement, llama-bench tuning, llama-server chat) can be tested on machines
where real model weights cannot be downloaded. The tokenizer is a real one
copied from llama.cpp's models/ggml-vocab-qwen2.gguf.

usage: python scripts/make_test_models.py <llama.cpp dir> <out dir>
"""
import os
import sys

import numpy as np


def main(llama_dir, out_dir):
    sys.path.insert(0, os.path.join(llama_dir, "gguf-py"))
    import gguf

    vocab = gguf.GGUFReader(os.path.join(llama_dir, "models", "ggml-vocab-qwen2.gguf"))
    n_vocab = len(vocab.fields["tokenizer.ggml.tokens"].data)
    rng = np.random.default_rng(0)

    def rnd(*shape):  # numpy shape is reversed relative to ggml's ne[]
        return (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(np.float16)

    def copy_tokenizer(w):
        for name, field in vocab.fields.items():
            if not name.startswith("tokenizer."):
                continue
            vtype = field.types[0]
            if vtype == gguf.GGUFValueType.ARRAY:
                sub = field.types[1]
                vals = [field.parts[i] for i in field.data]
                if sub == gguf.GGUFValueType.STRING:
                    w.add_array(name, [bytes(v) for v in vals])
                else:
                    w.add_array(name, [v.tolist()[0] for v in vals])
            elif vtype == gguf.GGUFValueType.STRING:
                w.add_string(name, bytes(field.parts[field.data[0]]).decode())
            else:
                w.add_key_value(name, field.parts[field.data[0]].tolist()[0], vtype)

    def build(arch, path, n_layer, n_embd, n_head, n_head_kv, n_ff, moe=None):
        w = gguf.GGUFWriter(path, arch)
        head_dim = n_embd // n_head
        w.add_name(f"pocketllm-test-{arch}-random")
        w.add_block_count(n_layer)
        w.add_context_length(4096)
        w.add_embedding_length(n_embd)
        w.add_feed_forward_length(n_ff)
        w.add_head_count(n_head)
        w.add_head_count_kv(n_head_kv)
        w.add_key_length(head_dim)
        w.add_value_length(head_dim)
        w.add_rope_freq_base(1e6)
        w.add_layer_norm_rms_eps(1e-6)
        w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
        if moe:
            n_expert, n_used, n_ff_exp = moe
            w.add_expert_count(n_expert)
            w.add_expert_used_count(n_used)
            w.add_expert_feed_forward_length(n_ff_exp)
        copy_tokenizer(w)
        w.add_tensor("token_embd.weight", rnd(n_vocab, n_embd))
        w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
        w.add_tensor("output.weight", rnd(n_vocab, n_embd))
        kv_dim = head_dim * n_head_kv
        for i in range(n_layer):
            p = f"blk.{i}."
            w.add_tensor(p + "attn_norm.weight", np.ones(n_embd, dtype=np.float32))
            w.add_tensor(p + "attn_q.weight", rnd(n_embd, n_embd))
            w.add_tensor(p + "attn_k.weight", rnd(kv_dim, n_embd))
            w.add_tensor(p + "attn_v.weight", rnd(kv_dim, n_embd))
            w.add_tensor(p + "attn_output.weight", rnd(n_embd, n_embd))
            w.add_tensor(p + "attn_q_norm.weight", np.ones(head_dim, dtype=np.float32))
            w.add_tensor(p + "attn_k_norm.weight", np.ones(head_dim, dtype=np.float32))
            w.add_tensor(p + "ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
            if moe:
                n_expert, _, n_ff_exp = moe
                w.add_tensor(p + "ffn_gate_inp.weight", rnd(n_expert, n_embd).astype(np.float32))
                w.add_tensor(p + "ffn_gate_exps.weight", rnd(n_expert, n_ff_exp, n_embd))
                w.add_tensor(p + "ffn_up_exps.weight", rnd(n_expert, n_ff_exp, n_embd))
                w.add_tensor(p + "ffn_down_exps.weight", rnd(n_expert, n_embd, n_ff_exp))
            else:
                w.add_tensor(p + "ffn_gate.weight", rnd(n_ff, n_embd))
                w.add_tensor(p + "ffn_up.weight", rnd(n_ff, n_embd))
                w.add_tensor(p + "ffn_down.weight", rnd(n_embd, n_ff))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)")

    os.makedirs(out_dir, exist_ok=True)
    build("qwen3", os.path.join(out_dir, "test-dense-f16.gguf"), n_layer=12, n_embd=1024, n_head=16, n_head_kv=4, n_ff=3072)
    build("qwen3moe", os.path.join(out_dir, "test-moe-f16.gguf"), n_layer=12, n_embd=1024, n_head=16, n_head_kv=4,
          n_ff=3072, moe=(32, 4, 512))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
