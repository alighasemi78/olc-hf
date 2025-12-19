import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent_loop import run_turn_based_agents

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small + permissive


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    def step_fn(text: str) -> str:
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

    prompt = "You have two agents. Solve: If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?"
    res = run_turn_based_agents(step_fn, prompt, n_steps=4)

    print("\n".join(res["shared"]))


if __name__ == "__main__":
    main()
