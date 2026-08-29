import csv
import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.opensdi import GENERATORS, GENERATOR_LABELS


@hydra.main(version_base=None, config_path="../configs/eval", config_name="ablation")
def main(cfg: DictConfig):
    results_dir = Path(cfg.results_dir)
    rows = []
    for arm in cfg.arms:
        path = results_dir / f"{arm['name']}.json"
        if not path.exists():
            print(f"missing {path} — skipping {arm['label']}")
            continue
        data = json.load(open(path))
        per_gen = data["per_generator"]
        rows.append({
            "label": arm["label"],
            "loc": [per_gen[g]["loc_f1"] for g in GENERATORS],
            "det": [per_gen[g]["det_f1"] for g in GENERATORS],
            "loc_avg": data["average"]["loc_f1"],
            "det_avg": data["average"]["det_f1"],
        })

    if not rows:
        print("No result files found.")
        return

    labels = [GENERATOR_LABELS[g] for g in GENERATORS]
    header = (f"{'Model':22s}" + "".join(f"{l:>8s}" for l in labels) + f"{'Avg.':>8s}"
              + "  |" + "".join(f"{l:>8s}" for l in labels) + f"{'Avg.':>8s}")
    print("\n" + " " * 22 + f"{'Localization: pixel F1':^48s}" + "  |"
          + f"{'Detection: F1':^48s}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['label']:22s}"
              + "".join(f"{v * 100:8.1f}" for v in row["loc"])
              + f"{row['loc_avg'] * 100:8.1f}" + "  |"
              + "".join(f"{v * 100:8.1f}" for v in row["det"])
              + f"{row['det_avg'] * 100:8.1f}")

    out_path = results_dir / "ablation_table.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model"] + [f"loc_{g}" for g in GENERATORS] + ["loc_avg"]
                        + [f"det_{g}" for g in GENERATORS] + ["det_avg"])
        for row in rows:
            writer.writerow([row["label"]] + row["loc"] + [row["loc_avg"]]
                            + row["det"] + [row["det_avg"]])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
