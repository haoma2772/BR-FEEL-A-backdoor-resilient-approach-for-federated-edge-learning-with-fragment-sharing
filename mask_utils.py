import torch
import pandas as pd
import base_config
from torchvision import transforms
from utility import get_poison_transform
from test_model import test_model
from model import Net
import copy
import wandb
import math



def get_layer_name(model_name, backdoor_rate, fragment_ratio):
    all_layer_name = {}
    layer_name = []
    backdoor_train_layer_name = []
    clinet_layer_idx = []
    if model_name == 'resnet34':
        all_layer_name = {'conv1', 'bn1','layer1', 'layer2', 'layer3', 'layer4', 'fc'}
        clinet_layer_idx = [4,5,0,1,2,3,0,1,2,3]
        if fragment_ratio == 'small':
            layer_name = [{'conv1','layer1', },
                            {'bn1',  'layer3', },
                            { 'layer2', 'layer4', },
                            { 'layer4', 'fc'},
                            { 'layer3', 'fc'},
                            {'conv1', 'layer4',},
                            ]
        elif fragment_ratio == 'middle':
            layer_name = [{'conv1','layer1', 'layer3'},
                            {'bn1', 'layer2', 'layer3', },
                            {'conv1', 'layer1', 'layer4', },
                            {'layer2', 'layer4', 'fc'},
                            {'layer1', 'layer3', 'fc'},
                            {'conv1','layer3', 'layer4',},
                            ]
        elif fragment_ratio == 'large':
            layer_name = [{'conv1','layer1', 'layer3','fc'},
                            {'bn1', 'layer2', 'layer3','fc' },
                            {'conv1', 'layer1', 'layer4','fc' },
                            {'layer2', 'layer3','layer4', 'fc'},
                            { 'layer1','layer2', 'layer3','fc'},
                            {'conv1','bn1','layer3', 'layer4',},
                            ]


        if backdoor_rate == 0:
            pass
        
        elif backdoor_rate == 0.1:
            
            backdoor_train_layer_name = {'layer1', 'layer3', 'fc'}
            
            
        elif backdoor_rate == 0.2:
                    
            if fragment_ratio == 'small':
                backdoor_train_layer_name = {'conv1', 'layer3', 'layer4', 'fc'}
            elif fragment_ratio == 'middle':
                backdoor_train_layer_name = {'conv1', 'layer1','layer3', 'layer4', 'fc'}   
            elif fragment_ratio == 'large':
                backdoor_train_layer_name = {'conv1', 'bn1','layer1','layer2','layer3', 'layer4', 'fc'}
            

        elif backdoor_rate == 0.3:

            backdoor_train_layer_name = {'conv1', 'layer1', 'layer2','layer3', 'layer4', 'fc'}
            

        elif backdoor_rate == 0.9:

            backdoor_train_layer_name = {'conv1', 'bn1','layer1', 'layer2','layer3', 'layer4', 'fc'}
    elif model_name == 'mobilenetv2':

        all_layer_name = {'features.0.', 'features.1.', 'features.2.', 'features.3.','features.4.', 'features.5.',
                          'features.6.', 'features.7.', 'features.8.', 'features.9.', 'features.10.', 'features.11.',
                          'features.12.', 'features.13.', 'features.14.', 'features.15.', 'features.16.', 'features.17.',
                          'features.18.', 'classifier.1'}
        clinet_layer_idx = [4,5,0,1,2,3,0,1,2,3]

        layer_name = [{'features.0.', 'features.1.', 'features.2.', 'features.3.','features.4.','features.5.',
                        'features.6.', 'features.7.', 'features.8.', 'features.17.', 'classifier.1',},

                      {'features.5.', 'features.6.', 'features.7.', 'features.8.', 'features.9.', 'features.10.',
                       'features.12.', 'features.13.', 'features.14.', 'features.15.', 'classifier.1',},

                      {'features.12.', 'features.13.', 'features.14.', 'features.15.', 'features.16.', 'features.17.',
                       'features.18.','features.0.', 'features.1.', 'features.2.','classifier.1',},

                      { 'features.1.', 'features.3.','features.5.','features.6.','features.9.', 'features.10.',
                       'features.11.',  'features.13.','features.15.', 'features.17.', 'classifier.1'},

                      {'features.0.',  'features.2.', 'features.4.', 'features.6.','features.8.', 'features.10.',
                        'features.12.',  'features.14.','features.16.',  'features.18.','classifier.1'},

                      {'features.1.', 'features.3.','features.5.','features.6.','features.9.', 'features.10.',
                       'features.11.',  'features.13.','features.15.', 'features.17.', 'classifier.1'}
                        ]
        
        if backdoor_rate == 0:
            pass
        
        elif backdoor_rate == 0.1:
            
            backdoor_train_layer_name = {'layer1', 'layer3', 'fc'}
            
            
        elif backdoor_rate == 0.2:
                    
            backdoor_train_layer_name = {'features.0.',  'features.1.', 'features.2.', 
                                'features.3.','features.4.','features.5.','features.6.', 
                                'features.8.', 'features.9.','features.10.','features.11.','features.12.', 
                                'features.13.', 'features.14.','features.15.','features.16.',
                                'features.17.','features.18.' 'classifier.1'}
            

        elif backdoor_rate == 0.3:

            backdoor_train_layer_name = {'conv1', 'layer1', 'layer2','layer3', 'layer4', 'fc'}
            

        elif backdoor_rate == 0.9:

            backdoor_train_layer_name = {'conv1', 'bn1','layer1', 'layer2','layer3', 'layer4', 'fc'}




    elif model_name == 'MLP':
        all_layer_name = {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias','fc3.weight', 'fc3.bias',
                          'fc4.weight', 'fc4.bias','fc5.weight', 'fc5.bias',}
        clinet_layer_idx = [4,5,0,1,2,3,0,1,2,3]
        layer_name = [{'fc1.weight', 'fc1.bias','fc2.weight', 'fc2.bias','fc4.weight', 'fc4.bias',},
                            {'fc1.weight', 'fc1.bias','fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias',},
                            {'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias','fc4.weight', 'fc4.bias',},
                            {'fc3.weight', 'fc3.bias','fc4.weight', 'fc4.bias','fc5.weight', 'fc5.bias',},
                            {'fc1.weight', 'fc1.bias','fc3.weight', 'fc3.bias','fc5.weight', 'fc5.bias',},
                            {'fc2.weight', 'fc2.bias','fc4.weight', 'fc4.bias','fc5.weight', 'fc5.bias',},
                            ]


        if backdoor_rate == 0:
            pass
        
        elif backdoor_rate == 0.1:
            
            backdoor_train_layer_name = {'layer1', 'layer3', 'fc'}
            
            
        elif backdoor_rate == 0.2:
                    
            backdoor_train_layer_name = {'fc1.weight', 'fc1.bias','fc2.weight', 'fc2.bias',
                                         'fc3.weight', 'fc3.bias','fc4.weight', 'fc4.bias','fc5.weight', 'fc5.bias',}
            

        elif backdoor_rate == 0.3:

            backdoor_train_layer_name = {'conv1', 'layer1', 'layer2','layer3', 'layer4', 'fc'}
            

        elif backdoor_rate == 0.9:

            backdoor_train_layer_name = {'conv1', 'bn1','layer1', 'layer2','layer3', 'layer4', 'fc'}
        
    
    elif model_name == 'vgg19':
        all_layer_name = {'features.0.', 'features.2.','features.5.', 'features.7.', 'features.10.', 'features.12.', 'features.14.',
                          'features.16.', 'features.19.', 'features.21.', 'features.23.', 'features.25.', 'features.28.', 'features.30.',
                          'features.32.', 'features.34.', 'classifier.0.', 'classifier.3.', 'classifier.6.'}
        
        layer_name = [      {'features.0.', 'features.2.','features.5.','features.12.','features.19.','features.21.','features.28.','classifier.0.',},
                            {'features.5.','features.7.', 'features.10.','features.12.', 'features.21.', 'features.30.','features.34.','classifier.3.'},
                            { 'features.2.','features.5.', 'features.14.','features.16.', 'features.19.', 'features.21.', 'features.32.','classifier.6.'},
                            {'features.0.','features.23.', 'features.25.', 'features.28.', 'features.30.', 'features.32.', 'features.34.','classifier.6.',},
                            {'features.0.','features.5.', 'features.10.', 'features.21.', 'features.25.', 'features.30.', 'features.34.','classifier.0.','classifier.3.'},
                            {'features.2.','features.7.', 'features.12.', 'features.23.', 'features.28.', 'features.32.', 'features.34.','classifier.3.','classifier.6.'},
                            ]
        clinet_layer_idx = [4,5,0,1,2,3,0,1,2,3]

        if backdoor_rate == 0.1:
            backdoor_train_layer_name = {'features.0.','features.5.', 'features.10.', 'features.21.', 'features.25.', 'features.30.', 'features.34.','classifier.0.','classifier.3.',
                            'features.2.','features.7.', 'features.12.', 'features.23.', 'features.28.', 'features.32.','classifier.6.'}
        
        elif backdoor_rate == 0.2:
             backdoor_train_layer_name = {'features.0.','features.5.', 'features.10.', 'features.21.', 'features.25.', 'features.30.', 'features.34.','classifier.0.','classifier.3.',
                            'features.2.','features.7.', 'features.12.', 'features.23.', 'features.28.', 'features.32.','classifier.6.'}
       

    else:
        raise ValueError("Error of model name!")   


    return all_layer_name, layer_name, backdoor_train_layer_name, clinet_layer_idx   
    



def average_nets_with_mask(net_list: list[Net], layer_name, layer_idx, my_idx, backdoor_client_index):

    # layer_idx specifies which layer each client ID contributes to.
    if not net_list:
        raise ValueError("the model list is empty")
    aggregated_params = {}
    aggregated_params_count = {}
 
    for id, net in enumerate(net_list):

        model_params = net.model.state_dict()
        if id == my_idx:
            if my_idx in backdoor_client_index:
                # bakcdoor client only aggregate his training layers
                for name, param in model_params.items():
                    if any(s in name for s in layer_name[layer_idx[id]]):
                        if name not in aggregated_params:
                            aggregated_params[name] = torch.zeros_like(param.data, dtype=torch.float)
                            aggregated_params_count[name] = 0
                        aggregated_params[name] += param.data
                        aggregated_params_count[name] += 1
            else:
                # benign clients aggregate all his own layers
                for name, param in model_params.items():
                    if name not in aggregated_params:
                        aggregated_params[name] = torch.zeros_like(param.data, dtype=torch.float)
                        aggregated_params_count[name] = 0
                    aggregated_params[name] += param.data
                    aggregated_params_count[name] += 1
        else:
                for name, param in model_params.items():
                    if any(s in name for s in layer_name[layer_idx[id]]):
                        if name not in aggregated_params:
                            aggregated_params[name] = torch.zeros_like(param.data, dtype=torch.float)
                            aggregated_params_count[name] = 0
                        aggregated_params[name] += param.data
                        aggregated_params_count[name] += 1

    for name, param in aggregated_params.items():
        aggregated_params[name] = aggregated_params[name] / aggregated_params_count[name]
        
    averaged_net = copy.deepcopy(net_list[0])
    averaged_net.model.load_state_dict(aggregated_params)
    return averaged_net


def kd_avg(net_list, layer_name, layer_idx, my_idx):
    # not include itself
    # layer_idx specifies which layer each client ID contributes to.
    if not net_list:
        raise ValueError("模型列表为空")
    aggregated_params = {}
    aggregated_params_count = {}
 
    for id, model in enumerate(net_list):
        # get the model's named parameters
        model_params = model.model.state_dict()
        if id == my_idx:
            pass
        else:
                # get the model's named parameters
                for name, param in model_params.items():
                    if any(s in name for s in layer_name[layer_idx[id]]):
                        if name not in aggregated_params:
                            aggregated_params[name] = torch.zeros_like(param.data, dtype=torch.float)
                            aggregated_params_count[name] = 0
                        aggregated_params[name] += param.data
                        aggregated_params_count[name] += 1
    for name, param in aggregated_params.items():
        aggregated_params[name] = aggregated_params[name] / aggregated_params_count[name]
    averaged_model = copy.deepcopy(net_list[0])
    averaged_model.model.load_state_dict(aggregated_params)
    # averaged_model = averaged_model.to('cpu')
    return averaged_model

def Median_models_with_mask(net_list: list[Net], layer_name, layer_idx, my_idx):

    num_clients = len(net_list)
    if not net_list:
        raise ValueError("The model list is empty")

    zeros_net= copy.deepcopy(net_list[0])

    for param in zeros_net.model.parameters():
        torch.nn.init.zeros_(param)
    w_zero = zeros_net.model.state_dict()

    # weight list
    model_weight: list = [copy.deepcopy(net_list[i].model.state_dict()) for i in range(num_clients)]
    aggregated_params = copy.deepcopy(w_zero)

    for k in w_zero.keys():
        tensors_list = []


        for idx in range(num_clients):
            if idx == my_idx or any(substring in k for substring in layer_name[layer_idx[idx]]):
                tensors_list.append(torch.flatten(model_weight[idx][k]))


        stacked_tensor = torch.stack(tensors_list)
        median_tensor, _ = torch.median(stacked_tensor, dim=0)

        # reshape the median tensor to the original shape
        aggregated_params[k] = torch.reshape(median_tensor, w_zero[k].shape)
    averaged_net = copy.deepcopy(zeros_net)
    averaged_net.model.load_state_dict(aggregated_params)
    return averaged_net



def vectorize_par_model_dict(model: torch.nn.Module, training_layers):
    # return a dict which
    # name: vector
    weight_accumulator = dict()
    for name, data in model.named_parameters():
        if any(s in name for s in training_layers):
            weight_accumulator[name] = data

    return weight_accumulator

def calculate_norm(model_weight_dict, p=2):
    # return a dict
    # naem l2, value
    l2_norm_dict = {}
    for layer_name, weights in model_weight_dict.items():
        l2_norm = torch.norm(weights,p=2)
        l2_norm_dict[layer_name] = l2_norm.item()
    return l2_norm_dict


def select_top_k_norms(l2_norm_dict:dict, top_k:int):
    # just return a list
    # each element is a layer name
    
    sorted_norms = sorted(l2_norm_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_norms_layers = [layer_name for layer_name, _ in sorted_norms[:top_k]]
    return top_k_norms_layers

def train_layer_model(model:torch.nn.Module, data_loader, config_list, lr, client_id,  global_round, 
                       trained_layer,record=None, with_test=False, test_dataset_loader=None,
               with_save=False, backdoor_client_index=[]):




    dataset_name = config_list['dataset']['name']
    num_epochs = config_list['device']['epoch']
    # log_dir = config_list['general']['log_dir']

    cuda_num = config_list['general']['cuda_number']
    
    

    if client_id in backdoor_client_index:
        if config_list['poisoned_paras']['neurotoxin'] == True:
            # select_layers is a list

            model_weight_dict = vectorize_par_model_dict(model, training_layers=trained_layer)
            top_k_ratio = config_list['poisoned_paras']['top_k_ratio']
            top_k = math.ceil(top_k_ratio * len(model_weight_dict))
            norm_dict = calculate_norm(model_weight_dict)
            select_layers:list = select_top_k_norms(norm_dict, top_k=top_k)
    
    

    device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        model.train()

        if config_list['poisoned_paras']['neurotoxin'] == True:
            # backdoor client
            if client_id in backdoor_client_index:
                # Select the layer to train.
                for name, param in model.named_parameters():
                    if name in select_layers:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                # Select the layer to train.
                for name, param in model.named_parameters():
                    if any(s in name for s in trained_layer):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if any(s in name for s in trained_layer): 
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                   
        running_loss = 0.0

        from tqdm import tqdm
        for images, labels in tqdm(data_loader, total=len(data_loader), desc=f"Training epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)  
            labels = labels.to(device)  
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}')
        # print(len(data_loader))
        if client_id in backdoor_client_index:
            wandb.log({'Loss of backdoor client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})
        else:
            wandb.log({'Loss of benign client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})
        
        if record is not None:
            record['loss'].append(running_loss / len(data_loader))

        if with_test:
            # CA and ASR need to be test
            poison_type = config_list['poisoned_paras']['poisoned_type']
            dataset_name = config_list['dataset']['name']
            target_class = base_config.target_class[dataset_name]

            alpha = config_list['poisoned_paras']['alpha']
            trigger = config_list['poisoned_paras']['trigger']

            num_classes = 0
            if trigger == 'None':
                trigger = base_config.trigger_default[dataset_name][poison_type]

            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),

            ])
            if dataset_name == 'cifar10':
                num_classes = 10
            elif dataset_name == 'gtsrb':
                num_classes = 43
            elif dataset_name == 'cifar100':
                num_classes = 100
            elif dataset_name == 'mnist':
                num_classes = 10

            all_to_all = False
            
            poison_transform = get_poison_transform(poison_type=poison_type, dataset_name=dataset_name,
                                                    target_class=target_class
                                                    , source_class=None, cover_classes=None,
                                                    is_normalized_input=True, trigger_transform=trigger_transform,
                                                    alpha=alpha, trigger_name=trigger, config_list=config_list)

            CA, ASR = test_model(model=model, data_loader=test_dataset_loader, config_list=config_list, path=None,
                                 poison_test=True,
                                 poison_transform=poison_transform, num_classes=num_classes,
                                 source_classes=None, all_to_all=all_to_all
                                 )

            if client_id in backdoor_client_index:
                wandb.log(
                {'Test_accuracy of backdoor client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})

                wandb.log({'Attack_success_rate of backdoor client#{}'.format(client_id): ASR,
                       'epoch': global_round * num_epochs + epoch})
            else:
                 wandb.log(
                 {'Test_accuracy of benign client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})

                 wandb.log({'Attack_success_rate of benign client#{}'.format(client_id): ASR,
                       'epoch': global_round * num_epochs + epoch})


            if record is not None:
                record['test_acc'].append(CA)
                record['attack_acc'].append(ASR)
    
    print('Finished Training')
    # for save the model
    
    # if with_save:
    #     current_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    #     model_name = f"{model_name}_{dataset_name}_{current_datetime}.pth"
    #     model_path = os.path.join(log_dir, model_name)
    #     torch.save(model.state_dict(), model_path)
    #     print('Model has been saved!')
    
    model.to('cpu')
    return model,  record


def model_replace(net_list:list[Net], backdoor_client_number, backdoor_model:torch.nn.Module, my_idx,  layer_name, layer_idx):
    
    # layer_idx specifies which layer each client ID contributes to.
    if not net_list:
        raise ValueError("The model list is empty")
    aggregated_params = {}
    aggregated_params_count = {}
 
    for id, net in enumerate(net_list):

        model_params = net.model.state_dict()
        if id == my_idx:
            continue
        else:
                for name, param in model_params.items():
                    if any(s in name for s in layer_name[layer_idx[id]]):
                        if name not in aggregated_params:
                            aggregated_params[name] = torch.zeros_like(param.data, dtype=torch.float)
                            aggregated_params_count[name] = 0
                        aggregated_params[name] += param.data
                        aggregated_params_count[name] += 1

    # for name, param in aggregated_params.items():
    #     aggregated_params[name] = aggregated_params[name] / aggregated_params_count[name]
    for name, param in net_list[0].model.state_dict().items():
        if any(s in name for s in layer_name[layer_idx[my_idx]]):
            aggregated_params[name] = (aggregated_params_count[name] * backdoor_model.state_dict()[name] - aggregated_params[name]) / backdoor_client_number
        else:
            aggregated_params[name] = backdoor_model.state_dict()[name] / backdoor_client_number
        
      
    averaged_net:Net = copy.deepcopy(net_list[0])
    averaged_net.model.load_state_dict(aggregated_params)
    return averaged_net.model


def BR_FEEL(teacher, student, dataloader, config_list, client_id, global_round,
                                            select_num=1,
                                               temperature=10,  record=None, alpha=0.5,
                                               with_test=False, test_dataset_loader=None,):
    
    # in this case 
    # teacher is only an aggregated net
    # teacher student is all net
    device_state = config_list['general']['device']
    cuda_num = config_list['general']['cuda_number']
    lr = config_list['device']['learning_rate']
    # 0.5 0.9 0.2
    alpha = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() and device_state == 'gpu' else "cpu")
    if torch.cuda.is_available() and device_state == 'gpu':
        device = torch.device("cuda:{}".format(cuda_num))
    else:
        device = torch.device("cpu")
    num_epochs = config_list['device']['epoch']


    teacher.model.to(device)
    teacher.model.eval()


    criterion = torch.nn.KLDivLoss(reduction='batchmean')  # Kullback-Leibler Divergence Loss
    optimizer = torch.optim.Adam(student.model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        # record the old studentmodel
        old_model = copy.deepcopy(student)
        old_model.model.eval()
        old_model.model.to(device)
        student.model.to(device)
        student.model.train()

        from tqdm import tqdm

        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, total=len(dataloader), desc=f"Training epoch {epoch + 1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                student_output = student(inputs)
                soft_student_output = student_output / temperature
                total_kl_loss = 0.0

                with torch.no_grad():
                    teacher_output = teacher(inputs)
                    soft_teacher_output = teacher_output / temperature
                total_kl_loss = criterion(torch.log_softmax(soft_student_output, dim=1),
                                                 torch.softmax(soft_teacher_output, dim=1))
                classification_loss = torch.nn.CrossEntropyLoss()(student_output, labels)
                # alpha should be smaller
                loss = alpha * total_kl_loss + (1 - alpha) * classification_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        wandb.log({'Loss of benign client#{}'.format(client_id): running_loss / len(dataloader), 'epoch': global_round * num_epochs + epoch, })
        if record is not None:
            record['loss'].append(running_loss / len(dataloader))

        if with_test:
            # need test
            poison_type = config_list['poisoned_paras']['poisoned_type']
            dataset_name = config_list['dataset']['name']
            target_class = base_config.target_class[dataset_name]

            talpha = config_list['poisoned_paras']['alpha']
            trigger = config_list['poisoned_paras']['trigger']
            if trigger == 'None':
                trigger = base_config.trigger_default[dataset_name][poison_type]

            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
            ])
            if dataset_name == 'cifar10':
                num_classes = 10
            elif dataset_name == 'gtsrb':
                num_classes = 43

            elif dataset_name == 'cifar100':
                num_classes = 100
            elif dataset_name == 'mnist':
                num_classes = 10

            all_to_all = False

            poison_transform = get_poison_transform(config_list=config_list, poison_type=poison_type, dataset_name=dataset_name,
                                                    target_class=target_class
                                                    , source_class=None, cover_classes=None,
                                                    is_normalized_input=True, trigger_transform=trigger_transform,
                                                    alpha=talpha, trigger_name=trigger)

            CA, ASR = test_model(model=student, data_loader=test_dataset_loader, config_list=config_list, path=None,
                                 poison_test=True,
                                 poison_transform=poison_transform, num_classes=num_classes,
                                 source_classes=None, all_to_all=all_to_all
                                 )

            wandb.log({'Test_accuracy of benign client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})
            wandb.log({'Attack_success_rate of benign client#{}'.format(client_id): ASR, 'epoch':global_round * num_epochs + epoch})
            
            

            if record is not None:
                record['test_acc'].append(CA)
                record['attack_acc'].append(ASR)
    student.model.to('cpu') 

    print('Distillation Finished!')
    return student, record

def vectorize_model(model: torch.nn.Module):
    # return one dim vector
    return torch.cat([p.clone().detach().view(-1) for p in model.parameters()])


def load_model_weight(model, vec_weight):

    index_bias = 0
    for p_index, p in enumerate(model.parameters()):
        p.data =  vec_weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

def NormClipping(model:torch.nn.Module, data_loader, config_list, lr, client_id,  global_round, 
                       trained_layer, norm_bound=20, stddev=0.025, record=None, with_test=False, test_dataset_loader=None,
               with_save=False, backdoor_client_index=[],):
    # NormClipping

    cur_model, record = train_layer_model(model=model, data_loader=data_loader, config_list=config_list, lr=lr, client_id=client_id,  global_round=global_round, 
                       trained_layer=trained_layer,record=record, with_test=with_test, test_dataset_loader=test_dataset_loader,
               with_save=with_save, backdoor_client_index=backdoor_client_index)
    
    vec_model = vectorize_model(cur_model)
    weight_norm = torch.norm(vec_model).item()
    clipped_weight = vec_model/max(1, weight_norm/norm_bound)

    dp_weight = clipped_weight + torch.randn(
        vec_model.size(), device=vec_model.device) * stddev
    
    load_model_weight(model=cur_model, vec_weight=dp_weight)
    return cur_model, record


def GeoMed(net_list: list[Net], layer_name, layer_idx, my_idx):
    
    from defense.geometric_median import geometric_median
    num_clients = len(net_list)
    if not net_list:
        raise ValueError("Model list is empty!")

    zeros_net:Net= copy.deepcopy(net_list[0])
    for param in zeros_net.model.parameters():
        torch.nn.init.zeros_(param)
    w_zero = zeros_net.model.state_dict()

    # weight list
    model_weight: list = [copy.deepcopy(net_list[i].model.state_dict()) for i in range(num_clients)]
    aggregated_params = copy.deepcopy(w_zero)
    import numpy as np
    for k in w_zero.keys():
        tensors_list = []
        for idx in range(num_clients):
            if idx == my_idx or any(substring in k for substring in layer_name[layer_idx[idx]]):
                tensors_list.append(torch.flatten(model_weight[idx][k]))
        
        median = geometric_median(tensors_list)
        res_tensor = torch.from_numpy(median.astype(np.float32))

        # stacked_tensor = torch.stack(tensors_list)
        # median_tensor, _ = torch.median(stacked_tensor, dim=0)

        # reshape the median tensor to the original shape
        aggregated_params[k] = torch.reshape(res_tensor, w_zero[k].shape)

    res_net = copy.deepcopy(zeros_net)
    res_net.model.load_state_dict(aggregated_params)
    return res_net
