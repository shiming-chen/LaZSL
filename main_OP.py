import torch

from load_OP import *
import torchmetrics
from tqdm import tqdm




seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(mydataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")

device = torch.device(hparams['device'])
# load model
model=load_clip_to_cpu()
model.to(device)
#model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)
#op=OP(max_iter=hparams['max_iter'],M=49,N=5,n_cls=hparams['class_num'],b=bs)
op_d=OP_d(max_iter=hparams['max_iter'], gama=hparams['gama'],constrain_type=hparams['constrain_type'],alpha=hparams['alpha'])

print("Encoding descriptions...")

description_encodings = compute_description_encodings(model)

label_encodings = compute_label_encodings(model)

print("n_samples: %d \nalpha: %f \nalpha_crop: %f" %(hparams['n_samples'],hparams['alpha'],hparams['alpha_crop']))
print("constrain_type: %s " %(hparams['constrain_type']))

print("Evaluating...")





langOPD_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
#langOPD_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],top_k=5).to(device)



lang_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
#lang_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],top_k=5).to(device)

#clip_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num']).to(device)
#clip_accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=hparams['class_num'],top_k=5).to(device)

for batch_number, batch in enumerate(tqdm(dataloader)):
    images, labels = batch
    
    images = images.to(device)
    labels = labels.to(device)

    #images_global=images[:,0]
    #images_region=images[:,1:]
    bs,n_region,n_chaneel,h,w=images.shape

    images=torch.reshape(images,(bs*n_region,n_chaneel,h,w))
    
    image_encodings = model.encode_image(images)

    image_encodings = torch.reshape(image_encodings,(bs,n_region,-1))
    image_encodings = torch.permute(image_encodings,(1,0,2))

    image_encodings = F.normalize(image_encodings, dim=2)


    image_encodings_global=image_encodings[0]

    image_encodings_region=image_encodings[1:]
    sim_rg = torch.einsum('nbd,bd->bn', image_encodings_region, image_encodings_global)





    
    #image_labels_similarity = image_encodings_pool @ label_encodings.T
    #clip_predictions = image_labels_similarity.argmax(dim=1)
    
    
    #clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
    #clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
    
    #
    image_description_similarity = [None] * n_classes

    image_description_similarity_cumulative_OP = [None] * n_classes
    image_description_similarity_cumulative_OPG = [None] * n_classes
    image_description_similarity_cumulative_OP_c = [None] * n_classes
    image_description_similarity_cumulative_OPG_c = [None] * n_classes
    image_description_similarity_cumulative = [None] * n_classes



    
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me

        # dot_product_matrix = image_encodings @ v.T

        # image_description_similarity[i] = dot_product_matrix


        """#region +cost&sim_global"""
        image_description_similarity_cumulative_OP_c[i] = op_d.get_OP_distence(image_features=image_encodings,
                                                                               text_features=v, sim_rg=sim_rg,
                                                                               is_cost_global=True,
                                                                               is_sim_global=True)

        dot_product_matrix = image_encodings_global @ v.T

        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])

    # create tensor of similarity means

    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)

    cumulative_tensor_OP_C = torch.stack(image_description_similarity_cumulative_OP_c, dim=1)


    if hparams['dataset'] == 'imagenet-a':
        cumulative_tensor = cumulative_tensor[:, imagenet_a_lt]
        cumulative_tensor_OP_C = cumulative_tensor_OP_C[:, imagenet_a_lt]

    if hparams['dataset'] == 'imagenet-r':
        cumulative_tensor = cumulative_tensor[:, imagenet_r_lt]
        cumulative_tensor_OP_C = cumulative_tensor_OP_C[:, imagenet_a_lt]





    
    
    lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    langOPC_acc = langOPD_accuracy_metric(cumulative_tensor_OP_C.softmax(dim=-1), labels)




    print("\n###ACC####")

    print("\nbaseline: %.2f\nbaseline+op+cost&sim_global: %.2f"
        % (lang_acc, langOPC_acc))
    
    

print("\n")

accuracy_logs = {}

accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
#accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()



accuracy_logs["Total DescriptionOP-Cost&SimGlobal-based Top-1 Accuracy: "] = 100*langOPD_accuracy_metric.compute().item()
#accuracy_logs["Total DescriptionOPCD-based Top-5 Accuracy: "] = 100*langOPD_accuracy_metric_top5.compute().item()







