"""沿用PoseVerifier的裁剪、批次和推理设置，仅分离CPU准备与GPU计算。"""
import numpy as np
import torch

def prepare(pose,image,boxes):
    prepared=[]
    for bbox_id,box in enumerate(boxes):
        x1,y1,x2,y2=box[:4]
        bbox=np.asarray([x1,y1,x2-x1+1,y2-y1+1,box[4]],dtype=np.float32)
        center,scale=pose.box2cs(pose.model.cfg,bbox)
        data=dict(center=center,scale=scale,bbox_score=float(box[4]),bbox_id=bbox_id,
                  dataset=pose.dataset_info.dataset_name,
                  joints_3d=np.zeros((pose.model.cfg.data_cfg.num_joints,3),dtype=np.float32),
                  joints_3d_visible=np.zeros((pose.model.cfg.data_cfg.num_joints,3),dtype=np.float32),
                  rotation=0,ann_info=dict(image_size=np.asarray(pose.model.cfg.data_cfg.image_size),
                  num_joints=pose.model.cfg.data_cfg.num_joints,flip_pairs=pose.dataset_info.flip_pairs),img=image)
        prepared.append(pose.pipeline(data))
    return prepared

def predict(pose,prepared,batch_size=256):
    if not prepared:return np.empty((0,2,3),dtype=np.float32)
    outputs=[]
    for start in range(0,len(prepared),batch_size):
        batch=pose.collate(prepared[start:start+batch_size],samples_per_gpu=batch_size)
        batch=pose.scatter(batch,[next(pose.model.parameters()).device])[0]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            result=pose.model(img=batch['img'],img_metas=batch['img_metas'],return_loss=False,return_heatmap=False)
        outputs.append(np.asarray(result['preds'],dtype=np.float32))
    predictions=np.concatenate(outputs,axis=0)
    assert predictions.shape==(len(prepared),2,3) and np.isfinite(predictions).all()
    return predictions
