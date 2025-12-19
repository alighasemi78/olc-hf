import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent_loop import run_turn_based_agents
from olc_hooks import OLCController, attach_projection_hooks

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Infer d_model from the embedding matrix
    d_model = model.get_input_embeddings().weight.shape[1]
    controller = OLCController(d_model=d_model, num_channels=2, device=model.device)
    handles = attach_projection_hooks(model, controller)

    def step_fn(text: str, agent_idx: int) -> str:
        controller.set_channel(agent_idx)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )
        gen = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return gen.strip()

    def step_fn_wrapped(text: str) -> str:
        # agent_loop alternates A,B; infer idx from last line length is messy; simplest:
        # run_turn_based_agents calls this without agent_idx; so use closure state
        raise RuntimeError("Use run_turn_based_agents_custom below")

    def run_turn_based_agents_custom(prompt: str, n_steps: int = 4):
        shared = []
        for t in range(n_steps):
            agent_idx = t % 2
            agent_name = "A" if agent_idx == 0 else "B"
            context = (
                f"Task:\n{prompt}\n\n"
                f"Shared scratchpad (may be incomplete):\n"
                + "\n".join(shared[-10:])
                + "\n\n"
                f"Agent {agent_name}, provide the next step. Keep it short."
            )
            out = step_fn(context, agent_idx)
            shared.append(f"{agent_name}: {out}")
        return shared

    prompt = "You have two agents. Solve: If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?"
    shared = run_turn_based_agents_custom(prompt, n_steps=4)

    for h in handles:
        h.remove()

    print("\n".join(shared))


if __name__ == "__main__":
    main()
