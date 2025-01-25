import copy
from utility import load_config
import wandb
from model import Net
import math
import datetime
import torch
from plugin import get_dataset
from utility import load_config
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
from wandb import AlertLevel
from mask_utils import train_layer_model, average_nets_with_mask, Median_models_with_mask, ours_KD, kd_avg
import argparse

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Run the JSA project with specified config file.')
    parser.add_argument('--config_path', type=str, default= './config/resnet34_config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    
    path = args.config_path
    project_name = 'JSA'
    config = load_config(path)
    

    # Retrieve various training parameters
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
    dataset_name = config['dataset']['name']
    poisoned_rate = config['poisoned_paras']['poisoned_rate']
    defense_method = config['general']['denfense']
    repeated_times = config['general']['runs_time']
    poisoned_method = config['poisoned_paras']['poisoned_type']
    epoch_time = config['device']['epoch']

    for each_repeat in range(repeated_times):
        now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        runs_name = f"motivation_{dataset_name}_{model_name}_{poisoned_method}_{defense_method}({now_time})_dalpha={dirichlet_rate}{backdoor_rate}"

        # Record metrics
        wandb.init(project=project_name, name=runs_name, config=config)

        wandb.alert(
            title="Auto DL Execution Started!",
            text=f"{runs_name} Auto DL execution started, iteration: {each_repeat + 1}/{repeated_times}",
            level=AlertLevel.WARN,
            wait_duration=1,
        )

        # Calculate the number of backdoor clients
        backdoor_client_number = max(0, math.ceil(client_number * backdoor_rate))
        benign_client_num = client_number - backdoor_client_number
        backdoor_client_index = list(range(backdoor_client_number))

        # Load datasets
        train_dataset_list, clean_test_dataset = get_dataset(
            backdoor_index=backdoor_client_index, config=config)

        dataset_train_loader_list = [
            DataLoader(train_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
            for client_id in range(client_number)]
        clean_test_dataloader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True, pin_memory=True)

        # Initialize models
        base_model = Net(config)
        model_list = [copy.deepcopy(base_model) for _ in range(client_number)]
        init_record_list = {'test_acc': [], 'attack_acc': [], 'loss': []}
        whole_init_record_list = [copy.deepcopy(init_record_list) for _ in range(client_number)]

        # Define layer mappings
        all_layer_name = {'conv1', 'conv2', 'conv3', 'conv4', 'fc'}
        layer_name = [
            {'conv1', 'conv2', 'fc'},
            {'conv1', 'conv3', 'fc'},
            {'conv4', 'conv3'},
            {'conv2', 'conv3', 'fc'},
        ]
        client_layer_idx = [3, 0, 1, 2]

        optimizer_list = [torch.optim.Adam(model_list[i].model.parameters(), lr=lr) for i in range(client_number)]
        upper_alpha = 0.9
        step_size = upper_alpha / global_round

        # Begin global training
        for each_round in range(global_round):
            # Backup models from the previous round
            model_list_old = copy.deepcopy(model_list)
            alpha = each_round * step_size

            for client_id in range(client_number):
                if client_id in backdoor_client_index:
                    # Backdoor client
                    print(f'Backdoor Client #{client_id} training!')

                    new_net = average_nets_with_mask(
                        model_list_old, layer_name, client_layer_idx, client_id, backdoor_client_index=backdoor_client_index)
                    optimizer_list[client_id] = torch.optim.Adam(new_net.model.parameters(), lr=lr)

                    model_list[client_id].model, optimizer_list[client_id], whole_init_record_list[client_id] = train_layer_model(
                        model=new_net.model,
                        data_loader=dataset_train_loader_list[client_id],
                        config_list=config,
                        client_id=client_id,
                        global_round=each_round,
                        optimizer=optimizer_list[client_id],
                        trained_layer=layer_name[client_layer_idx[client_id]],
                        record=whole_init_record_list[client_id],
                        with_test=True,
                        test_dataset_loader=clean_test_dataloader,
                        backdoor_client_index=backdoor_client_index
                    )
                else:
                    # Clean client
                    print(f'Clean Client #{client_id} Distillation!')
                    if defense_method == 'ours_kd':
                        new_net = kd_avg(model_list_old, layer_name, client_layer_idx, client_id)
                        optimizer_list[client_id] = torch.optim.Adam(new_net.model.parameters(), lr=lr)

                        model_list[client_id], whole_init_record_list[client_id] = ours_KD(
                            teacher=new_net,
                            student=model_list[client_id],
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
                    elif defense_method == 'avg':
                        new_net =average_nets_with_mask(
                            model_list_old, layer_name, client_layer_idx, client_id, backdoor_client_index=backdoor_client_index)
                        optimizer_list[client_id] = torch.optim.Adam(new_net.model.parameters(), lr=lr)

                        model_list[client_id].model, optimizer_list[client_id], whole_init_record_list[client_id] = train_layer_model(
                            model=new_net.model,
                            data_loader=dataset_train_loader_list[client_id],
                            config_list=config,
                            client_id=client_id,
                            global_round=each_round,
                            optimizer=optimizer_list[client_id],
                            trained_layer=all_layer_name,
                            record=whole_init_record_list[client_id],
                            with_test=True,
                            test_dataset_loader=clean_test_dataloader,
                            backdoor_client_index=backdoor_client_index
                        )
                    elif defense_method == 'middle':
                        new_net =Median_models_with_mask()(model_list_old, layer_name, client_layer_idx, client_id)
                        optimizer_list[client_id] = torch.optim.Adam(new_net.model.parameters(), lr=lr)

                        model_list[client_id].model, optimizer_list[client_id], whole_init_record_list[client_id] = train_layer_model(
                            model=new_net.model,
                            data_loader=dataset_train_loader_list[client_id],
                            config_list=config,
                            client_id=client_id,
                            global_round=each_round,
                            optimizer=optimizer_list[client_id],
                            trained_layer=all_layer_name,
                            record=whole_init_record_list[client_id],
                            with_test=True,
                            test_dataset_loader=clean_test_dataloader,
                            backdoor_client_index=backdoor_client_index
                        )

        # Save results
        tmp_file_name = f'motivation_{dirichlet_rate}_{dataset_name}_{poisoned_method}_{backdoor_rate}'
        save_path = os.path.join('mask_backdoor_res', tmp_file_name)
        os.makedirs(save_path, exist_ok=True)

        for idx, net in enumerate(model_list):
            model_path = os.path.join(save_path, f"model_{idx}.pth")
            torch.save(net.model.state_dict(), model_path)

        file_path = os.path.join(save_path, 'record_res.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(whole_init_record_list, f)

        wandb.alert(
            title="Auto DL Execution Ended!",
            text=f"{runs_name} Auto DL execution ended, iteration: {each_repeat + 1}/{repeated_times}",
            level=AlertLevel.WARN,
            wait_duration=1,
        )

        wandb.finish()
