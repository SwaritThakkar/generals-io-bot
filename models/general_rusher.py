"""Siege stress-test opponent: charge at a producer, then all-in the general."""

def act(state, player):
    from strategies.wastar import search
    plan = search(state, player, epsilon=1.0)
    # Wait while charging. Once the general is affordable, follow the shortest
    # feasible route one all-in step at a time and replan after every response.
    if plan is not None and plan.target_kind == "enemy general":
        return int(plan.action)
    return 0
