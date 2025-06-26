import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from load_OP import *
import torchmetrics
from tqdm import tqdm
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "Total DescriptionOP-CostGlobal-based Top-1 Accuracy"},
    "parameters": {
        "n_samples": {"values": [90,60,70,80]},
        "alpha": {"values": [0.9,0.95,0.8,0.85]},
        "alpha_crop": {"values": [0.2,0.3,0.4,0.5]}

    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='clip')
#sweep_id = "zi71byp1"
def main():
    wandb.init(project='clip')
    config = wandb.config
    hparams['n_samples'] = config.n_samples
    # for mix
    hparams['alpha'] = config.alpha
    # for crop
    hparams['alpha_crop'] = config.alpha_crop
    seed_everything(hparams['seed'])

    bs = hparams['batch_size']
    dataloader = DataLoader(mydataset, bs, shuffle=True, num_workers=16, pin_memory=True)

    print("Loading model...")

    device = torch.device(hparams['device'])
    # load model
    model = load_clip_to_cpu()
    model.to(device)
    # model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)
    # op=OP(max_iter=hparams['max_iter'],M=49,N=5,n_cls=hparams['class_num'],b=bs)
    op_d = OP_d(max_iter=hparams['max_iter'], gama=hparams['gama'], constrain_type=hparams['constrain_type'],
                alpha=hparams['alpha'])

    print("Encoding descriptions...")

    description_encodings = compute_description_encodings(model)

    label_encodings = compute_label_encodings(model)

    print(
        "n_samples: %d \nalpha: %f \nalpha_crop: %f" % (hparams['n_samples'], hparams['alpha'], hparams['alpha_crop']))
    print("constrain_type: %s " % (hparams['constrain_type']))

    print("Evaluating...")

    langOP_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
    langOP_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],
                                                        top_k=5).to(device)

    langOPG_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
    langOPG_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],
                                                         top_k=5).to(device)

    langOPD_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
    langOPD_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],
                                                         top_k=5).to(device)

    langOPGD_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
    langOPGD_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],
                                                          top_k=5).to(device)

    lang_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
    lang_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'], top_k=5).to(
        device)

    # clip_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
    # clip_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],top_k=5).to(device)

    for batch_number, batch in enumerate(tqdm(dataloader)):
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        # images_global=images[:,0]
        # images_region=images[:,1:]

        bs, n_region, n_chaneel, h, w = images.shape

        images = torch.reshape(images, (bs * n_region, n_chaneel, h, w))

        image_encodings = model.encode_image(images)

        image_encodings = torch.reshape(image_encodings, (bs, n_region, -1))
        image_encodings = torch.permute(image_encodings, (1, 0, 2))

        image_encodings = F.normalize(image_encodings, dim=2)

        image_encodings_global = image_encodings[0]

        image_encodings_region = image_encodings[1:]
        sim_rg = torch.einsum('nbd,bd->bn', image_encodings_region, image_encodings_global)

        # image_encodings_region = F.normalize(image_encodings_region, dim=2)
        # image_encodings_global = F.normalize(image_encodings_global, dim=1)

        # image_labels_similarity = image_encodings_pool @ label_encodings.T
        # clip_predictions = image_labels_similarity.argmax(dim=1)

        # clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
        # clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)

        image_description_similarity = [None] * n_classes

        image_description_similarity_cumulative_OP = [None] * n_classes
        image_description_similarity_cumulative_OPG = [None] * n_classes
        image_description_similarity_cumulative_OP_c = [None] * n_classes
        image_description_similarity_cumulative_OPG_c = [None] * n_classes
        image_description_similarity_cumulative = [None] * n_classes

        for i, (k, v) in enumerate(
                description_encodings.items()):  # You can also vectorize this; it wasn't much faster for me

            # dot_product_matrix = image_encodings @ v.T

            # image_description_similarity[i] = dot_product_matrix

            """#only region"""
            image_description_similarity_cumulative_OP[i] = op_d.get_OP_distence(image_features=image_encodings_region,
                                                                                 text_features=v, sim_rg=sim_rg,
                                                                                 is_constrain=False)
            # image_description_similarity_cumulative_OPG[i] = op_d.get_OP_distence(image_features=image_encodings_region,text_features=v, is_constrain=False)
            """#region+global"""
            image_description_similarity_cumulative_OPG[i] = op_d.get_OP_distence(image_features=image_encodings,
                                                                                  text_features=v, sim_rg=sim_rg,
                                                                                  is_constrain=False)

            """#regionl+cost_global"""
            image_description_similarity_cumulative_OPG_c[i] = op_d.get_OP_distence(image_features=image_encodings,
                                                                                    text_features=v, sim_rg=sim_rg,
                                                                                    is_constrain=False,
                                                                                    is_cost_global=True)
            """#regionl +cost&sim_global"""
            image_description_similarity_cumulative_OP_c[i] = op_d.get_OP_distence(image_features=image_encodings,
                                                                                   text_features=v, sim_rg=sim_rg,
                                                                                   is_cost_global=True,
                                                                                   is_sim_global=True)

            dot_product_matrix = image_encodings_global @ v.T

            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])

        # create tensor of similarity means
        # only global
        cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)
        # only region
        cumulative_tensor_OP = torch.stack(image_description_similarity_cumulative_OP, dim=1)
        # all
        cumulative_tensor_OPG = torch.stack(image_description_similarity_cumulative_OPG, dim=1)
        cumulative_tensor_OP_C = torch.stack(image_description_similarity_cumulative_OP_c, dim=1)
        cumulative_tensor_OPG_c = torch.stack(image_description_similarity_cumulative_OPG_c, dim=1)

        # mix global and region
        # global + all
        # cumulative_tensor_OPGD = hparams['alpha']*cumulative_tensor_OPG + (1-hparams['alpha'])* cumulative_tensor
        # global + region
        # cumulative_tensor_OPD = hparams['alpha'] * cumulative_tensor_OP + (1 - hparams['alpha']) * cumulative_tensor

        # descr_predictions = cumulative_tensor.argmax(dim=1)
        # descr_predictions_OP = cumulative_tensor_OP.argmax(dim=1)

        lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
        # lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)

        langOP_acc = langOP_accuracy_metric(cumulative_tensor_OP.softmax(dim=-1), labels)
        # langOP_acc_top5 = langOP_accuracy_metric_top5(cumulative_tensor_OP.softmax(dim=-1), labels)

        langOPG_acc = langOPG_accuracy_metric(cumulative_tensor_OPG.softmax(dim=-1), labels)
        # langOPG_acc_top5 = langOPG_accuracy_metric_top5(cumulative_tensor_OPG.softmax(dim=-1), labels)

        langOPC_acc = langOPD_accuracy_metric(cumulative_tensor_OP_C.softmax(dim=-1), labels)
        # langOPD_acc_top5 = langOPD_accuracy_metric_top5(cumulative_tensor_OPD.softmax(dim=-1), labels)

        langOPGC_acc = langOPGD_accuracy_metric(cumulative_tensor_OPG_c.softmax(dim=-1), labels)
        # langOPGD_acc_top5 = langOPGD_accuracy_metric_top5(cumulative_tensor_OPGD.softmax(dim=-1), labels)

        print("\n###ACC####")

        print(
            "\nbaseline: %.2f\nbaseline+OP: %.2f\nbaseline+op+global: %.2f\nbaseline+op+costglobal: %.2f\nbaseline+op+cost&sim_global: %.2f"
            % (lang_acc, langOP_acc, langOPG_acc, langOPGC_acc, langOPC_acc))

    print("\n")

    accuracy_logs = {}

    accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100 * lang_accuracy_metric.compute().item()
    # accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

    accuracy_logs["Total DescriptionOP-based Top-1 Accuracy: "] = 100 * langOP_accuracy_metric.compute().item()
    # accuracy_logs["Total DescriptionOPC-based Top-5 Accuracy: "] = 100*langOP_accuracy_metric_top5.compute().item()

    accuracy_logs["Total DescriptionOP-Global-based Top-1 Accuracy: "] = 100 * langOPG_accuracy_metric.compute().item()
    # accuracy_logs["Total DescriptionOP-based Top-5 Accuracy: "] = 100*langOPG_accuracy_metric_top5.compute().item()

    accuracy_logs[
        "Total DescriptionOP-CostGlobal-based Top-1 Accuracy: "] = 100 * langOPGD_accuracy_metric.compute().item()
    # accuracy_logs["Total DescriptionOPD-based Top-5 Accuracy: "] = 100*langOPGD_accuracy_metric_top5.compute().item()

    accuracy_logs[
        "Total DescriptionOP-Cost&SimGlobal-based Top-1 Accuracy: "] = 100 * langOPD_accuracy_metric.compute().item()
    # accuracy_logs["Total DescriptionOPCD-based Top-5 Accuracy: "] = 100*langOPD_accuracy_metric_top5.compute().item()

    # accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
    # accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

    # print the dictionary
    print("\n")
    dataset = hparams['dataset']
    encoder = hparams['model_size']
    alpha = hparams['alpha']
    alpha_crop = hparams['alpha_crop']
    n_samples = hparams['n_samples']

    for key, value in accuracy_logs.items():
        print(key, value)

    wandb.log({
        'Total Description-based Top-1 Accuracy': 100 * lang_accuracy_metric.compute().item(),
        'Total DescriptionOP-based Top-1 Accuracy': 100 * langOP_accuracy_metric.compute().item(),
        'Total DescriptionOP-Global-based Top-1 Accuracy': 100 * langOPG_accuracy_metric.compute().item(),
        'Total DescriptionOP-CostGlobal-based Top-1 Accuracy': 100 * langOPGD_accuracy_metric.compute().item(),
        'Total DescriptionOP-Cost&SimGlobal-based Top-1 Accuracy': 100 * langOPD_accuracy_metric.compute().item()

    })


# Start sweep job.
wandb.agent(sweep_id, project='clip', function=main, count=2000)