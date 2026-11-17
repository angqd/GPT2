from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math 

class CausalSelfAttention(nn.Module):
    ## multi head attention 
    ## leverage batch dimensions where it applies all operations to all batches in parallel 
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key query vector projections for all heads in a abatch 
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        #output projection 
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        # regularisation 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C = x.size() # batch size , sequence length , embedding dimensionality 
        # C is number of channels  nh : num heads , hs : head size , C = hs * nh 
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        # attention 
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        att = F.softmax(att,dim=-1)
        y = att @ v # (B,nh,T,T) X (B,nh,T,hs) -> B,nh,T,hs
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module): 
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') 
        # we use the gelu approx to be able to load the original weights 
        self.c_proj = nn.Linear(config.n_embd*4,config.n_embd)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass 
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        # define transformer 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

    def forward(self,idx):
        # idx is shape (B,T)
        B,T = idx.size()
        assert T <= self.config.block_size
        # forward pos and token embeddings 
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the tranformer 
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) 
        return logits

    # ---------------- generation (single prompt idx of shape (1,T) -> (1,T+new)) ----------------
    # no KV cache: every step re-runs the full forward pass over the sequence so far

    def _next_logits(self, x):
        # logits for the next token of every row in x (B,T) -> (B,vocab_size)
        # crop to the last block_size tokens so long generations don't overflow wpe
        return self(x[:, -self.config.block_size:])[:, -1, :]

    @torch.no_grad()
    def greedy_search(self, idx, max_new_tokens=50, eos_token_id=None):
        # always take the single most likely next token (deterministic)
        assert idx.size(0) == 1
        for _ in range(max_new_tokens):
            next_tok = self._next_logits(idx).argmax(dim=-1, keepdim=True)  # (1,1)
            idx = torch.cat((idx, next_tok), dim=1)
            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break
        return idx

    @torch.no_grad()
    def top_k_sampling(self, idx, max_new_tokens=50, k=50, temperature=1.0, eos_token_id=None):
        # sample only among the k most likely tokens
        assert idx.size(0) == 1
        for _ in range(max_new_tokens):
            probs = F.softmax(self._next_logits(idx) / temperature, dim=-1)  # (1,vocab)
            topk_probs, topk_idx = torch.topk(probs, k, dim=-1)              # (1,k)
            # multinomial renormalises the k probabilities for us
            next_tok = torch.gather(topk_idx, -1, torch.multinomial(topk_probs, 1))
            idx = torch.cat((idx, next_tok), dim=1)
            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break
        return idx

    @torch.no_grad()
    def top_p_sampling(self, idx, max_new_tokens=50, p=0.9, temperature=1.0, eos_token_id=None):
        # nucleus sampling: sample from the smallest set of tokens whose cumulative prob >= p
        assert idx.size(0) == 1
        for _ in range(max_new_tokens):
            probs = F.softmax(self._next_logits(idx) / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            # drop a token if the tokens ranked above it already cover p
            # (cum - own prob), so the first token that crosses p is kept and rank 0 is always kept
            sorted_probs[(cum - sorted_probs) >= p] = 0.0
            next_tok = torch.gather(sorted_idx, -1, torch.multinomial(sorted_probs, 1))
            idx = torch.cat((idx, next_tok), dim=1)
            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break
        return idx

    @torch.no_grad()
    def beam_search(self, idx, max_new_tokens=50, num_beams=4, length_penalty=1.0, eos_token_id=None):
        # keep the num_beams highest total-log-prob sequences at each step, return the best one
        assert idx.size(0) == 1
        V = self.config.vocab_size
        prompt_len = idx.size(1)
        beams = idx.repeat(num_beams, 1)                       # (nb,T)
        scores = torch.zeros(num_beams, device=idx.device)     # summed log probs of each beam
        scores[1:] = -1e9      # all beams start identical; only let beam 0 expand at step 0
        finished = []          # (length-normalised score, tokens) for beams that hit EOS

        def normalise(score, length):
            return score / (length ** length_penalty)

        for step in range(max_new_tokens):
            logp = F.log_softmax(self._next_logits(beams), dim=-1)      # (nb,V)
            cand = (scores[:, None] + logp).view(-1)                    # (nb*V,) every beam x token
            # take 2*nb candidates so that nb non-EOS ones survive even if some hit EOS
            top_scores, top_ids = torch.topk(cand, 2 * num_beams)
            beam_ids, tok_ids = top_ids // V, top_ids % V
            new_beams, new_scores = [], []
            for rank, (s, b, t) in enumerate(zip(top_scores, beam_ids, tok_ids)):
                seq = torch.cat((beams[b], t.view(1)))
                if eos_token_id is not None and t.item() == eos_token_id:
                    # only count it if it ranks within the top num_beams (otherwise it can't win a slot)
                    if rank < num_beams:
                        finished.append((normalise(s.item(), seq.size(0) - prompt_len), seq))
                else:
                    new_beams.append(seq); new_scores.append(s)
                if len(new_beams) == num_beams:
                    break
            beams, scores = torch.stack(new_beams), torch.stack(new_scores)
            # stop early once we have num_beams finished and no running beam can still beat the worst
            if len(finished) >= num_beams:
                best_running = normalise(scores.max().item(), step + 1)
                if best_running <= min(f[0] for f in finished):
                    break

        # sequences still running at the end compete with the finished ones
        gen_len = beams.size(1) - prompt_len
        finished += [(normalise(s.item(), gen_len), seq) for s, seq in zip(scores, beams)]
        return max(finished, key=lambda f: f[0])[1].unsqueeze(0)

    def generate(self, idx, max_new_tokens=50, method=None, do_sample=False, **kwargs):
        # method: 'greedy' | 'beam' | 'top_k' | 'top_p'; if omitted, do_sample picks top_k or greedy
        method = method or ('top_k' if do_sample else 'greedy')
        fn = {'greedy': self.greedy_search, 'beam': self.beam_search,
              'top_k': self.top_k_sampling, 'top_p': self.top_p_sampling}[method]
        return fn(idx, max_new_tokens=max_new_tokens, **kwargs)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        # the causal mask buffer is not a learnable weight, so we don't copy it
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        # the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



################### SAMPLING
# usage: python gpt2.py --strategy {greedy,beam,top_k,top_p}   (then answer the prompts)
import argparse, time

def ask(label, default, cast=str):
    # prompt for a value, Enter keeps the default
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        try:
            return cast(raw) if raw else default
        except ValueError:
            print(f"  expected {cast.__name__}, try again")

def check_on_device(model, device):
    # every parameter AND buffer (e.g. the causal mask "bias") must live on the target device
    off = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
           if t.device.type != device.type]
    assert not off, f"tensors not on {device}: {off[:5]}"
    n = sum(1 for _ in model.parameters()) + sum(1 for _ in model.buffers())
    print(f"all {n} parameter/buffer tensors are on {device}")

class GenStats:
    """--stats: per-forward-pass timing and MPS memory, collected with forward hooks (no model changes)."""
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.step_times, self.peak_mem = [], 0           # seconds per forward; sampled peak allocated bytes
        self.weights_mem = self.mem()                    # baseline: memory held by the loaded model
        self.hooks = [model.register_forward_pre_hook(self._pre), model.register_forward_hook(self._post)]

    def sync(self):
        # MPS is asynchronous: without a sync the timers would only measure kernel *launch* time
        if self.device.type == "mps":
            torch.mps.synchronize()

    def mem(self):
        return torch.mps.current_allocated_memory() if self.device.type == "mps" else 0

    def _pre(self, module, args):
        self.sync(); self._t0 = time.perf_counter()

    def _post(self, module, args, out):
        self.sync()
        self.step_times.append(time.perf_counter() - self._t0)
        self.peak_mem = max(self.peak_mem, self.mem())   # sampled after each forward, so approximate

    def warmup(self, ids):
        # first MPS run compiles Metal shaders; do it once untimed so time-to-first-token is honest
        with torch.no_grad():
            self.model(ids[:, -self.model.config.block_size:])
        self.sync()
        self.step_times, self.peak_mem = [], 0           # the hooks recorded the warmup pass; drop it

    def report(self, n_new, total_s):
        for h in self.hooks: h.remove()
        t = self.step_times
        mb = lambda b: f"{b / 2**20:,.0f} MB"
        print("\n---- stats ----")
        print(f"total {total_s:.2f}s | {n_new} new tokens | {n_new / total_s:.1f} tokens/s | {total_s / n_new * 1000:.1f} ms/token")
        print(f"forward passes: {len(t)} | first (prompt, time-to-first-token): {t[0] * 1000:.1f} ms")
        if len(t) > 1:
            later = t[1:]
            print(f"later passes: mean {sum(later) / len(later) * 1000:.1f} ms | last {t[-1] * 1000:.1f} ms "
                  f"(no KV cache, so each pass re-processes the whole sequence)")
        if self.device.type == "mps":
            print(f"memory: weights {mb(self.weights_mem)} | peak during generation ~{mb(self.peak_mem)} | "
                  f"driver total {mb(torch.mps.driver_allocated_memory())} / recommended max {mb(torch.mps.recommended_max_memory())}")
        else:
            print("memory: only reported for --device mps")

def main():
    parser = argparse.ArgumentParser(description="Generate text with the hand-written GPT-2")
    parser.add_argument("-s", "--strategy", required=True, choices=["greedy", "beam", "top_k", "top_p"])
    parser.add_argument("--device", default="mps", help="mps (default) or cpu")
    parser.add_argument("--stats", action="store_true", help="print tokens/sec, latency and MPS memory after generating")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine (use --device cpu)")

    model = GPT.from_pretrained('gpt2').eval().to(device)
    check_on_device(model, device)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')

    # ---- ask for every parameter the chosen strategy needs ----
    prompt = ""
    while not prompt:
        prompt = input("Prompt: ").strip()
    max_new_tokens = ask("max_new_tokens", 50, int)
    kwargs = {}
    if args.strategy == "beam":
        kwargs["num_beams"] = ask("num_beams", 4, int)
        kwargs["length_penalty"] = ask("length_penalty", 1.0, float)
    elif args.strategy == "top_k":
        kwargs["k"] = ask("k", 50, int)
        kwargs["temperature"] = ask("temperature", 1.0, float)
    elif args.strategy == "top_p":
        kwargs["p"] = ask("p", 0.9, float)
        kwargs["temperature"] = ask("temperature", 1.0, float)
    if args.strategy in ("top_k", "top_p"):      # greedy / beam are deterministic
        seed = ask("seed", 42, int)
        torch.manual_seed(seed)
        if device.type == "mps":
            torch.mps.manual_seed(seed)
    if ask("stop at <|endoftext|> (y/n)", "y") == "y":
        kwargs["eos_token_id"] = enc.eot_token

    # ---- generate ----
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)   # (1,T)
    stats = GenStats(model, device) if args.stats else None
    if stats:
        stats.warmup(ids)
        stats.sync(); t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, method=args.strategy, **kwargs)
    if stats:
        stats.sync(); total_s = time.perf_counter() - t0
    assert out.device.type == device.type
    n_new = out.size(1) - ids.size(1)
    print(f"\n[{args.strategy}] {n_new} new tokens\n")
    print(enc.decode(out[0].tolist()))
    if stats:
        stats.report(n_new, total_s)

if __name__ == "__main__":
    main()
