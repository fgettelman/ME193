"""
Copy walks between the board(s) and the laptop, so both partners' walks end
up in one place (the navigator trains on both of your data).

    python sync.py                      # board <-> laptop (data/): both directions
    python sync.py --pull               # only board -> laptop
    python sync.py --push               # only laptop -> board (e.g. your partner's walks)
    python sync.py --host A --host B    # two boards: everything ends up on both
    python sync.py --models             # also push the laptop's models (from train.py)

Walks are matched by id; a newer review (labels) wins either way.
"""

import argparse
import os

import common
import deploy

store = common.st.Store(common.DATA)


def walk_ids_on(host):
    return {w["id"]: w for w in common.get(host, "/api/state")["walks"]}


def labels_time(lab):
    return (lab or {}).get("reviewed", 0)


def pull(host):
    from server import unbundle
    n = 0
    for wid, w in walk_ids_on(host).items():
        have = os.path.exists(os.path.join(store._wdir(wid), "meta.json"))
        if not have or w.get("reviewed_at", 0) > labels_time(store.labels(wid)):
            unbundle(store, common.get(host, "/api/bundle?id=" + wid, timeout=60))
            n += 1
    for rec in common.get(host, "/api/state")["stairs"]["recordings"]:
        if not os.path.exists(os.path.join(store.root, "stairs", rec["id"], "meta.json")):
            from server import unbundle_stairs
            unbundle_stairs(store, common.get(host, "/api/stairs_bundle?id=" + rec["id"], timeout=60))
            n += 1
    print(f"{host}: pulled {n} walks/recordings -> data/")


def push(host, models=False):
    from server import bundle, stairs_bundle
    on = walk_ids_on(host)
    n = 0
    for wid in store.walk_ids():
        if wid not in on or labels_time(store.labels(wid)) > on[wid].get("reviewed_at", 0):
            common.post(host, "/api/bundle", bundle(store, wid), timeout=120)
            n += 1
    on_st = {r["id"] for r in common.get(host, "/api/state")["stairs"]["recordings"]}
    for rec in store.stairs_recordings():
        if rec["id"] not in on_st:
            common.post(host, "/api/stairs_bundle", stairs_bundle(store, rec["id"]), timeout=120)
            n += 1
    for name, p in store.people().items():
        common.post(host, "/api/person", {"name": name, "step_length_m": p["step_length_m"]})
    if models:
        for name in ("policy", "stairs"):
            m = common.st._read(os.path.join(store.root, "models", name + ".json"))
            if m:
                common.post(host, f"/api/model?name={name}", m, timeout=60)
                print(f"{host}: pushed {name} model")
    print(f"{host}: pushed {n} walks/recordings")


def main():
    ap = argparse.ArgumentParser(description="Sync walks between board(s) and laptop")
    ap.add_argument("--host", action="append", help=f"board IP (default {deploy.BOARD_HOST}); repeat for two boards")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--models", action="store_true", help="also push the laptop's trained models")
    args = ap.parse_args()
    hosts = args.host or [deploy.BOARD_HOST]
    both = not (args.pull or args.push)
    if args.pull or both:
        for h in hosts:
            pull(h)
    if args.push or both or args.models:
        for h in hosts:
            push(h, args.models)
    print(f"Laptop now has {len(store.walk_ids())} walks in data/walks/.")


if __name__ == "__main__":
    main()
