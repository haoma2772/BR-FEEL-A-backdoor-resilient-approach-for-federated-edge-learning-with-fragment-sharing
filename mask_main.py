import copy
from model import Net
from utility import load_config
import wandb
from model import Net
import math
import datetime

import torch
from plugin import get_dataset
from utility import load_config
from torch.utils.data import DataLoader

import pickle
import os
from wandb import AlertLevel
from mask_utils import get_layer_name, NormClipping, GeoMed
from mask_utils import train_layer_model, average_nets_with_mask, Median_models_with_mask, BR_FEEL, kd_avg
import argparse
from mask_utils import model_replace


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the JSA project with specified config file.')
    parser.add_argument('--config_path', type=str, default= './config/resnet34_config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    
    path = args.config_path
    project_name = 'JSA'

    config = load_config(path)

    # Retrieve various parameters for training
    global_round = config['federated_learning']['round']
    client_number = config['federated_learning']['client_number']
    split_rate = config['federated_learning']['split_rate']
 
    select_number = config['general']['select_num']
    dirichlet_rate = config['federated_learning']['dirichlet_rate']
    temperature = config['general']['temperature']
    backdoor_rate = config['federated_learning']['backdoor_rate']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    lr = config['device']['learning_rate']
    runs_time = config['general']['runs_time']
    model_name = config['model']['name']
    dataset_name= config['dataset']['name']

    poisoned_rate = config['poisoned_paras']['poisoned_rate']
    defense_method = config['general']['denfense']
    repeated_times = config['general']['runs_time']
    poisoned_method = config['poisoned_paras']['poisoned_type']
    epoch_time = config['device']['epoch']
    fragment_ratio = config['poisoned_paras']['fragment_ratio']


    for each_repeat in range(repeated_times):

        now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if config['poisoned_paras']['model_replace'] == True:
            runs_name = 'rebuttal_' +(dataset_name + '_' + model_name + '_' + defense_method + '_'+ 'model_replace' + '({date})'.format(
            date=now_time) + 'dalpha=' +
                     str(dirichlet_rate) +'_' + str(backdoor_rate)) + '_'+ fragment_ratio
        elif config['poisoned_paras']['neurotoxin'] == True:
            runs_name = 'rebuttal_' +(dataset_name + '_' + model_name + '_' + defense_method + '_'+ 'neurotoxin' + '({date})'.format(
            date=now_time) + 'dalpha=' +
                     str(dirichlet_rate) +'_' + str(backdoor_rate)) + '_'+ fragment_ratio
        else:
            runs_name = 'rebuttal_' +(dataset_name + '_' + model_name + '_' + defense_method + '_'+ poisoned_method + '({date})'.format(
            date=now_time) + 'dalpha=' +
                     str(dirichlet_rate) +'_' + str(backdoor_rate)) + '_'+ fragment_ratio

        # Record various metrics
        wandb.init(project=project_name, name=runs_name, config=config)

        wandb.alert(
            title="Auto DL Code Execution Started!",
            text="{} Auto DL code execution started, iteration: {}/{}".format(runs_name, each_repeat + 1, repeated_times),
            level=AlertLevel.WARN,
            wait_duration=1, )


        backdoor_client_number: int = max(0, math.ceil(client_number * backdoor_rate))
        benign_client_num = client_number - backdoor_client_number

        backdoor_client_index = list(range(backdoor_client_number))


        train_dataset_list, clean_test_dataset = get_dataset(
            backdoor_index=backdoor_client_index, config=config)

        dataset_train_loader_list = [
            DataLoader(train_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers, shuffle=True,pin_memory=True) for
            client_id in range(client_number)]
        clean_test_dataloader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True,pin_memory=True)


        base_net:Net = Net(config)
        net_list: list[Net] = [copy.deepcopy(base_net) for _ in range(client_number)]
        if config['poisoned_paras']['model_replace'] == True:
            true_bakcdoor_net = [copy.deepcopy(base_net) for _ in range(backdoor_client_number)]

        init_record_list = {'test_acc': [], 'attack_acc': [], 'loss': []}
        whole_init_record_list: list = [copy.deepcopy(init_record_list) for _ in range(client_number)]


        all_layer_name, layer_name, backdoor_train_layer_name, clinet_layer_idx = get_layer_name(model_name=model_name, backdoor_rate=backdoor_rate,
                                                                                fragment_ratio=config['poisoned_paras']['fragment_ratio'])


        upper_alpha = 0.9
        step_size = upper_alpha / global_round

        # Begin global training
        for each_round in range(global_round):

            # Copy the model for the old round
            net_list_old:list[Net] = copy.deepcopy(net_list)
            if config['poisoned_paras']['model_replace'] == True:
                true_bakcdoor_net_old = copy.deepcopy(true_bakcdoor_net)
            alpha = each_round * step_size
            epsilon = 1 - alpha

            # Training for each client
            for client_id in range(client_number):
                # Backdoor client
                if client_id in backdoor_client_index:
                    print('Backdoor Client #{} training!'.format(client_id))

                    if config['poisoned_paras']['model_replace'] == True:
                        new_net = average_nets_with_mask(net_list_old, layer_name, clinet_layer_idx, client_id,
                                                       backdoor_client_index=backdoor_client_index,)
                        net_list[client_id].model, whole_init_record_list[
                                client_id] = train_layer_model(
                                                model=new_net.model,
                                                data_loader=dataset_train_loader_list[client_id],
                                                config_list=config,
                                                lr=lr,
                                                client_id=client_id,
                                                global_round=each_round,    
                                                trained_layer=backdoor_train_layer_name,
                                                record=whole_init_record_list[client_id],
                                                with_test=True,
                                                test_dataset_loader=clean_test_dataloader,
                                                backdoor_client_index=backdoor_client_index
                        )
                        true_bakcdoor_net[client_id] = copy.deepcopy(net_list[client_id])
                        # Conduct model replacement

                        net_list[client_id].model = model_replace(net_list=net_list_old, backdoor_client_number=backdoor_client_number,
                                                                  backdoor_model=true_bakcdoor_net[client_id].model,
                                                                  layer_name=layer_name, layer_idx=clinet_layer_idx, my_idx=client_id,)
                    else:
                        new_net = average_nets_with_mask(net_list_old, layer_name, clinet_layer_idx, client_id,
                                                         backdoor_client_index=backdoor_client_index,)
                        net_list[client_id].model, whole_init_record_list[
                                client_id] = train_layer_model(
                                                model=new_net.model,
                                                data_loader=dataset_train_loader_list[client_id],
                                                config_list=config,
                                                lr=lr,
                                                client_id=client_id,
                                                global_round=each_round,    
                                                trained_layer=backdoor_train_layer_name,
                                                record=whole_init_record_list[client_id],
                                                with_test=True,
                                                test_dataset_loader=clean_test_dataloader,
                                                backdoor_client_index=backdoor_client_index
                        )
                else:
                    # Clean client
                    print('Clean Client #{} Distillation!'.format(client_id))
                    
                    if defense_method == 'BR-FEEL':
                        new_net:Net = kd_avg(net_list_old, layer_name, clinet_layer_idx, client_id)
                        net_list[client_id], whole_init_record_list[
                            client_id] = BR_FEEL(
                            teacher=new_net,
                            student=net_list[client_id],
                            dataloader=dataset_train_loader_list[client_id],
                            config_list=config,
                            client_id=client_id,
                            global_round=each_round,
                            select_num=select_number,
                            temperature=temperature, alpha=alpha,
                            record=whole_init_record_list[client_id],
                            with_test=True,
                            test_dataset_loader=clean_test_dataloader,
                        )
                    elif defense_method == 'Vanilla FEEL':
                        # Train all layers
                        
                        new_net = average_nets_with_mask(net_list_old, layer_name, clinet_layer_idx, client_id,
                                                         backdoor_client_index=backdoor_client_index)
                        net_list[client_id].model, whole_init_record_list[
                            client_id] = train_layer_model(
                                            model=new_net.model,
                                            data_loader=dataset_train_loader_list[client_id],
                                            config_list=config,
                                            lr=lr,
                                            client_id=client_id,
                                            global_round=each_round,
                                            trained_layer=all_layer_name,
                                            record=whole_init_record_list[client_id],
                                            with_test=True,
                                            test_dataset_loader=clean_test_dataloader,
                                            backdoor_client_index=backdoor_client_index
                        )
                    elif defense_method == 'Median':
                        
                        new_net = Median_models_with_mask(net_list_old, layer_name, clinet_layer_idx, client_id)
                        net_list[client_id].model, whole_init_record_list[
                            client_id] = train_layer_model(
                                            model=new_net.model,
                                            data_loader=dataset_train_loader_list[client_id],
                                            config_list=config,
                                            lr=lr,
                                            client_id=client_id,
                                            global_round=each_round,
                                            trained_layer=all_layer_name,
                                            record=whole_init_record_list[client_id],
                                            with_test=True,
                                            test_dataset_loader=clean_test_dataloader,
                                            backdoor_client_index=backdoor_client_index
                        )
                    elif defense_method == 'Norm clipping':
                        new_net = average_nets_with_mask(net_list_old, layer_name, clinet_layer_idx, client_id,
                                                         backdoor_client_index=backdoor_client_index)
                        net_list[client_id].model, whole_init_record_list[
                            client_id] = NormClipping(
                                model=new_net.model,
                                data_loader=dataset_train_loader_list[client_id],
                                config_list=config,
                                lr=lr,
                                client_id=client_id,
                                global_round=each_round,
                                trained_layer=all_layer_name,
                                record=whole_init_record_list[client_id],
                                with_test=True,
                                test_dataset_loader=clean_test_dataloader,
                                with_save=False,
                                backdoor_client_index=backdoor_client_index,
                        )
                    elif defense_method == 'Geometric median':
                    
                        new_net = GeoMed(net_list_old, layer_name, clinet_layer_idx, client_id)
                        net_list[client_id].model, whole_init_record_list[
                            client_id] = train_layer_model(
                                            model=new_net.model,
                                            data_loader=dataset_train_loader_list[client_id],
                                            config_list=config,
                                            lr=lr,
                                            client_id=client_id,
                                            global_round=each_round,
                                            trained_layer=all_layer_name,
                                            record=whole_init_record_list[client_id],
                                            with_test=True,
                                            test_dataset_loader=clean_test_dataloader,
                                            backdoor_client_index=backdoor_client_index
                        )

        # Save the results
        if config['poisoned_paras']['model_replace'] == True:
            tmp_file_name = 'rebuttal_' + str(dirichlet_rate) + '_' + dataset_name + '_' + model_name + '_' + defense_method + '_model_replace_' + str(backdoor_rate) + '_' + fragment_ratio
        elif config['poisoned_paras']['neurotoxin'] == True:
            tmp_file_name = 'rebuttal_' + str(dirichlet_rate) + '_' + dataset_name + '_' + model_name + '_' + defense_method + '_neurotoxin_' + str(backdoor_rate) + '_' + fragment_ratio
        else:
            tmp_file_name = 'rebuttal_' + str(dirichlet_rate) + '_' + dataset_name + '_' + model_name + '_' + defense_method + '_' + poisoned_method + '_' + str(backdoor_rate) + '_' + fragment_ratio
        save_path = os.path.join('rebuttal', tmp_file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, 'record_res.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(whole_init_record_list, f)

        wandb.alert(
            title="Auto DL Code Execution Ended!",
            text="{} Auto DL code execution ended, iteration: {}/{}".format(runs_name, each_repeat + 1, repeated_times),
            level=AlertLevel.WARN,
            wait_duration=1, )
        
        wandb.finish()
