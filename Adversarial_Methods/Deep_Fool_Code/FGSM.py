
def fgsm(model, loss, device, images, labels, eps,is_bayes=False):
    import torch
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
    if(is_bayes):
        outputs,_ = model(images)
    else:
        outputs = model(images)
    
    model.zero_grad()
    # print(outputs.shape,labels.shape)
    cost = loss(outputs, labels).to(device)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

# TODO: Attack with marginalization
# def bayes_fgsm(model, loss, device, images, labels, eps, J=5):
#     import torch
#     images = images.to(device)
#     labels = labels.to(device)
#     images.requires_grad = True
    
#     # train should be true for marginalization
#     model.train(True)
#     outputs = torch.zeros(images.shape[0], model.num_c, J).to(device)
#     for j in range(J):
#     outputs,_ = model(images)
    
def fgsm_graybox(models, loss, images, labels, eps) :
    # todo: feed in cross-entropy loss for graybox scenario
    
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
            
    outputs = [model(images) for model in models]
    for model in models:
        model.zero_grad()
    
    costs = [loss(output, labels).to(device) for output in outputs]
    cost = torch.stack(costs, dim=0).mean(dim=0)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images