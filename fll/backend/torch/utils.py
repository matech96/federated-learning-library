import torch as th


def state_dict_eq(state1, state2):
    return all(
        [
            (sk == ok) and th.all(sv == ov)
            for (sk, sv), (ok, ov) in zip(state1.items(), state2.items())
        ]
    )
