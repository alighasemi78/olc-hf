from dataclasses import dataclass

@dataclass
class AgentState:
    name: str
    history: list  # list[str]

def run_turn_based_agents(step_fn, prompt: str, n_steps: int = 4):
    agents = [AgentState("A", []), AgentState("B", [])]
    shared = []

    for t in range(n_steps):
        a = agents[t % 2]
        context = (
            f"Task:\n{prompt}\n\n"
            f"Shared scratchpad (may be incomplete):\n" + "\n".join(shared[-10:]) + "\n\n"
            f"Agent {a.name}, provide the next step. Keep it short."
        )
        out = step_fn(context)
        a.history.append(out)
        shared.append(f"{a.name}: {out}")

    return {"shared": shared, "agents": agents}
