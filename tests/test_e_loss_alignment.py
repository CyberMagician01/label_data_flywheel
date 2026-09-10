"""直接执行 E 源码中的损失函数，验证加权对梯度及不完整图像的作用。"""

import ast
from pathlib import Path
import types

import pytest
torch = pytest.importorskip("torch")
from torch.nn import functional as F
from torchvision.ops import box_iou as tv_box_iou, box_convert


def criterion():
    path=Path(__file__).resolve().parents[1]/"legacy/route_e/ecdetseg/engine/edgecrafter/criterion.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ECCriterion")
    names={"_valid_query_weight","_background_query_weight","_matched_instance_weight",
           "_get_src_permutation_idx","loss_labels_mal","loss_labels_vfl","loss_density",
           "_build_adaptive_density_target", "loss_ir_hard_negative"}
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    cls.bases=[]; cls.decorator_list=[]
    namespace={"torch":torch,"F":F,
               "box_iou":lambda a,b:(tv_box_iou(a,b),None),
               "box_cxcywh_to_xyxy":lambda x:box_convert(x,"cxcywh","xyxy")}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),"exec"),namespace)
    c=namespace["ECCriterion"]()
    c.num_classes=1; c.alpha=.75; c.gamma=2.; c.mal_alpha=None; c.ignore_query_iou_threshold=.5
    c.use_adaptive_density_target=False
    return c


def test_quality_scales_real_classification_gradient_instead_of_changing_target():
    c=criterion()
    gradients=[]
    for q in (1.,.1):
        logits=torch.tensor([[[.2]]],requires_grad=True)
        box=torch.tensor([[[.5,.5,.2,.2]]])
        target={"boxes":box[0],"labels":torch.tensor([0]),"hierarchy_quality":torch.tensor([q])}
        loss=c.loss_labels_mal({"pred_logits":logits,"pred_boxes":box},[target],
                               [(torch.tensor([0]),torch.tensor([0]))],1)["loss_mal"]
        loss.backward(); gradients.append(logits.grad.item())
    assert gradients[1] == pytest.approx(.1*gradients[0],rel=1e-5)
    assert gradients[0] < 0  # 都推动已标正例的预测概率提高


def test_partial_image_keeps_positive_and_verified_negative_but_not_unknown_background():
    c=criterion()
    boxes=torch.tensor([[[.2,.2,.1,.1],[.5,.5,.1,.1],[.8,.8,.1,.1]]])
    logits=torch.zeros(1,3,1,requires_grad=True)
    target={"boxes":boxes[0,:1],"labels":torch.tensor([0]),"annotation_complete":False,
            "verified_background_boxes":boxes[0,2:]}
    loss=c.loss_labels_mal({"pred_logits":logits,"pred_boxes":boxes},[target],
                           [(torch.tensor([0]),torch.tensor([0]))],1)["loss_mal"]
    loss.backward()
    assert logits.grad[0,0,0] < 0
    assert logits.grad[0,1,0] == 0
    assert logits.grad[0,2,0] > 0


def test_partial_count_does_not_teach_whole_image_density():
    c=criterion(); prediction=torch.ones(1,1,4,4,requires_grad=True)
    losses=c.loss_density({"pred_density":prediction},[{"annotation_complete":False}],[],1)
    sum(losses.values()).backward()
    assert prediction.grad.abs().sum()==0


def test_hard_negative_prior_does_not_override_partial_background_mask():
    c=criterion()
    boxes=torch.tensor([[[.2,.2,.1,.1],[.5,.5,.1,.1],[.8,.8,.1,.1]]])
    logits=torch.zeros(1,3,1,requires_grad=True)
    target={"boxes":boxes[0,:1],"labels":torch.tensor([0]),"annotation_complete":False,
            "verified_background_boxes":boxes[0,2:]}
    outputs={"pred_logits":logits,"pred_boxes":boxes,"query_initial_references":boxes,
             "pred_ir_hard_negative_prior":torch.ones(1,1,4,4)}
    losses=c.loss_ir_hard_negative(outputs,[target],
                                   [(torch.tensor([0]),torch.tensor([0]))],1)
    sum(losses.values()).backward()
    assert logits.grad[0,0,0]==0 and logits.grad[0,1,0]==0
    assert logits.grad[0,2,0]>0


def test_background_regions_follow_real_flip_and_letterbox():
    from torchvision import tv_tensors
    from torchvision.transforms.v2 import functional as image_F
    root=Path(__file__).resolve().parents[1]
    path=root/"legacy/route_e/ecdetseg/engine/data/transforms/_transforms.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef)
             and n.name in {"RandomHorizontalFlipWithKeypoints","LetterBox"}]
    for cls in classes: cls.decorator_list=[]
    def tv_box(data, key, box_format, spatial_size):
        return tv_tensors.BoundingBoxes(data,format=box_format,canvas_size=spatial_size)
    namespace={"torch":torch,"F":image_F,"convert_to_tv_tensor":tv_box}
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),"exec"),namespace)
    boxes=tv_tensors.BoundingBoxes([[10.,5.,30.,15.]],format="XYXY",canvas_size=(50,100))
    target={"boxes":boxes,"ignore_boxes":boxes.clone(),"verified_background_boxes":boxes.clone()}
    image,target=namespace["RandomHorizontalFlipWithKeypoints"](1)((torch.zeros(3,50,100),target))
    image,target=namespace["LetterBox"](200)((image,target))
    expected=torch.tensor([[140.,60.,180.,80.]])
    assert image.shape[-2:]==(200,200)
    for name in ("boxes","ignore_boxes","verified_background_boxes"):
        assert torch.equal(target[name].as_subclass(torch.Tensor),expected)
