import os
import torch
import argparse
import torch.nn as nn

from utils import load_model, load_dataloader
from utils import PREDICTION_RES_DIR, WORK_DIR, SPLIT_SYM
from utils import load_DLCL, init_bd_trigger
from src.model.utils import MyModel
from src.dlcl import TargetDevice
from src.abst_cl_model import TorchModel
from src.attack.utils import collect_model_pred


def remove_non_module_attributes(obj):
    # Loop over all attributes of the object
    for attr_name in dir(obj):
        attr_value = getattr(obj, attr_name)

        # If the attribute is not an instance of nn.Module, delete it
        if not isinstance(attr_value, nn.Module) and not attr_name.startswith("__"):
            delattr(obj, attr_name)

def predict_two_inputs(model, test_loader, device, bd_trigger):

    bd_preds = collect_model_pred(model, test_loader, device, bd_trigger, return_label=False)
    cl_preds, y = collect_model_pred(model, test_loader, device, None, return_label=True)
    bd_y = torch.full_like(y, bd_trigger.target_label, dtype=torch.long)
    bd_pred_labels = bd_preds.max(1)[1]
    cl_pred_labels = cl_preds.max(1)[1]

    return {
        "bd_preds": bd_preds,
        "cl_preds": cl_preds,
        "bd_pred_labels": bd_pred_labels,
        "cl_pred_labels": cl_pred_labels,
        "y": y,
        "bd_y": bd_y,
    }
def compute_acc_asr(pred_dict):
    y = pred_dict["y"]
    cl = pred_dict["cl_pred_labels"]
    bd = pred_dict["bd_pred_labels"]
    bd_y = pred_dict["bd_y"]

    clean_acc = (cl == y).float().mean().item()
    asr = (bd == bd_y).float().mean().item()

    return clean_acc, asr

def load_our_model(save_dir):
    # main.py 攻击完成后一般会在这里落 best.tar
    possible = ["best.tar"] + [f"{5*i+4}.tar" for i in reversed(range(20))]

    # fallback：如果当前目录没找到，就去同名的 cpu/gpu 目录找
    alt_dirs = [save_dir]
    if save_dir.endswith("cpu"):
        alt_dirs.append(save_dir[:-3] + "gpu")
    elif save_dir.endswith("gpu"):
        alt_dirs.append(save_dir[:-3] + "cpu")

    for d in alt_dirs:
        for name in possible:
            p = os.path.join(d, name)
            if os.path.exists(p):
                bd_trigger, save_model, _ = torch.load(p, map_location="cpu", weights_only=False)
                return True, bd_trigger, save_model

    return False, None, None



def get_prediction(approach_name):
    batch_size = 100
    current_work_dir = f"{approach_name}_work_dir"
    if not os.path.isdir(current_work_dir):
        os.mkdir(current_work_dir)
    approach_dir = os.path.join(PREDICTION_RES_DIR, approach_name)
    if not os.path.isdir(approach_dir):
        os.mkdir(approach_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_types = ['float32']
    output_num = 1

    model = load_model(load_pretrained=True)
    model_data_name = model.model_data_name
    train_loader, valid_loader, test_loader = load_dataloader(False, batch_size, batch_size)
    for hardware_id in [-1, 0]:
        hardware_target = TargetDevice(hardware_id)
        
        task_name = model_data_name + SPLIT_SYM + "torchcompile" + SPLIT_SYM + str(hardware_target)

        if approach_name == "clean":
            is_load = True
            bd_trigger = init_bd_trigger(8, "left_up", device)
        elif approach_name == "ours":
            save_dir = os.path.join(WORK_DIR, task_name)
            is_load, bd_trigger, loaded_model = load_our_model(save_dir)
            if is_load:
                model = loaded_model  # 只有成功加载才覆盖 model
            output_num = 10  # CIFAR10

        elif approach_name == "belt":
            is_load = True
            belt_save_dir = os.path.join("model_weight", "belt")
            [bd_trigger, state_dict] = torch.load(os.path.join(belt_save_dir, f"{model_data_name}."))
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError
        if not is_load:
            print("ERROR:", approach_name, task_name, "Not load the model")
            continue

        cl_func = load_DLCL()
        bd_trigger.trigger = bd_trigger.trigger.to(device)
        abst_model = TorchModel(
            model, batch_size, model.input_sizes, input_types,
            output_num, current_work_dir,
            model_name=task_name,
            target_device=hardware_target,
        )
        compiled_model = cl_func(abst_model)
        model = model.eval().to(device)

        ori_pred_path = os.path.join(approach_dir, task_name + ".ori")
        
        compiled_pred_path = os.path.join(approach_dir, task_name + ".compiled")
        if not os.path.isfile(compiled_pred_path):
            try:
                compiled_model_prediction = predict_two_inputs(compiled_model, test_loader, device, bd_trigger)
                torch.save(compiled_model_prediction, compiled_pred_path)

                clean_acc, asr = compute_acc_asr(compiled_model_prediction)
                print("Compiled Model -> clean_acc: %.4f  ASR: %.4f" % (clean_acc, asr))

            except:
                pass


        if not os.path.isfile(ori_pred_path):
            try:
                ori_model_prediction = predict_two_inputs(model, test_loader, device, bd_trigger)
                torch.save(ori_model_prediction, ori_pred_path)
                clean_acc, asr = compute_acc_asr(ori_model_prediction)
                print("Original Model -> clean_acc: %.4f  ASR: %.4f" % (clean_acc, asr))
            except:
                pass

        print(approach_name, task_name, "Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach_name', type=str, default="ours")
    args = parser.parse_args()
    get_prediction(args.approach_name)

