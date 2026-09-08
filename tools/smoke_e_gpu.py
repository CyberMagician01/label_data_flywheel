"""在实际E网络上验证双域前后向和优化器更新；随机权重测试不用于准确率。"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy/route_e/ecdetseg"))
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from engine.core import YAMLConfig
    from label_data_flywheel.io import write_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.manual_seed(3407)
    torch.set_num_threads(4)
    t = time.time()
    config = YAMLConfig(
        str(ROOT / "legacy/route_e/ecdetseg/configs/bee_e/e2_formal_joint_1280.yml")
    )
    config.yaml_cfg["eval_spatial_size"] = [128, 128]
    config.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    config.yaml_cfg["ViTAdapter"]["enable_dual_domain_norm"] = True
    config.yaml_cfg["ViTAdapter"]["domain_adapter_rank"] = 4
    config.yaml_cfg["ECDet"]["enable_explicit_domain"] = True
    model = config.model.cuda().train()
    criterion = config.criterion.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    targets = [
        {
            "labels": torch.tensor([0], device="cuda"),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device="cuda"),
            "keypoints": torch.tensor(
                [[[0.46, 0.5, 2.0], [0.54, 0.5, 2.0]]], device="cuda"
            ),
            "pose_state": torch.tensor([2], device="cuda"),
            "pose_mask": torch.tensor([1.0], device="cuda"),
            "domain_id": torch.tensor([i], device="cuda"),
        }
        for i in (0, 1)
    ]
    x = torch.rand(2, 3, 128, 128, device="cuda")
    output = model(x, targets)
    losses = criterion(output, targets)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    assert parameters and all(torch.isfinite(p.grad).all() for p in parameters)
    before = parameters[-1].detach().clone()
    optimizer.step()
    changed = not torch.equal(before, parameters[-1].detach())
    assert changed
    report = {
        "passed": True,
        "device": torch.cuda.get_device_name(0),
        "domains_in_update": ["RGB", "IR"],
        "loss_terms": len(losses),
        "finite_loss": float(loss),
        "parameters_with_gradient": len(parameters),
        "optimizer_changed_parameters": changed,
        "output_shape": list(output["pred_boxes"].shape),
        "elapsed_seconds": time.time() - t,
        "weights": "random_initialization_for_execution_test",
        "accuracy_measurement": False,
    }
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
