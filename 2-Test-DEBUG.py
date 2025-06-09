import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from T_FFTRadNet.model.FFTRadNet_ViT import FFTRadNet_ViT
from T_FFTRadNet.model.FFTRadNet_ViT import FFTRadNet_ViT_ADC
from T_FFTRadNet.dataset.dataset import RADIal
from T_FFTRadNet.dataset.encoder_NEW import RAEncoder as ra_encoder
import cv2

from utils.save_model_outputs import batch_save_predictions

def main(config, checkpoint_filename,difficult):
    import torch
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    if config['data_mode'] != 'ADC':
        net = FFTRadNet_ViT(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

        dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=True,perform_FFT=config['data_mode'])

    else:
        net = FFTRadNet_ViT_ADC(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

        dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=True,perform_FFT='ADC')

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, weights_only=False, map_location=torch.device(device))
    net.load_state_dict(dict['net_state_dict'])
    net = net.double()
    net.eval()

    ### $$$$$$$###### $$$$$$$###### $$$$$$$###### $$$$$$$###### $$$$$$$###
    # Batch save results
    model_outputs_dict = {}
    ### $$$$$$$###### $$$$$$$###### $$$$$$$###### $$$$$$$###### $$$$$$$###


    for data in tqdm(dataset, desc="Processing samples", unit="sample"):
        # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        inputs = torch.tensor(data[0]).permute(2,0,1).to(device).unsqueeze(0)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            if config['data_mode'] == 'ADC':
                intermediate = net.DFT(inputs).detach().cpu().numpy()[0]

            else:
                intermediate = None

        if data[4] is not None: # there is image

            ### $$$$$$$ ### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$
            model_outputs_dict[data[5]] = outputs

            from pathlib import Path
            import torch
            from plots import dl_output_viz

            path_repo = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils')
            dd = "/Volumes/ELEMENTS/datasets/radial"
            record = "RECORD@2020-11-22_12.45.05"
            if not path_repo.exists():
                path_repo = Path('/utils')
                dd = "/mnt/data/datasets/radial/gd/raw_data/"
                record = "RECORD@2020-11-22_12.54.38"

            root_folder = Path(dd, 'RadIal_Data', record)
            ra_dir = Path(root_folder, 'radar_RA')
            ra_path = Path(ra_dir) / f"ra_{data[-1]:06d}.npy"
            # ra_map = np.load(ra_path)
            #
            # res_ra =  dl_output_viz.visualize_detections_on_bev(ra_map, outputs, enc, max_range=103)
            # dl_output_viz.draw_boxes_on_RA_map(res_ra)
            #
            # res_ra = dl_output_viz.visualize_detections_on_ra_map(ra_map, outputs, enc)
            # dl_output_viz.draw_boxes_on_RA_map(res_ra)
            #
            # res_img = dl_output_viz.visualize_detections_on_image(data[4], outputs, enc)
            # dl_output_viz.draw_boxes_on_RA_map(res_img)
            ### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$### $$$$$$$


            # hmi = DisplayHMI(data[4], data[0],outputs,enc,config,intermediate)
            #
            # cv2.imshow('FFTRadNet',hmi)
            #
            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        # break
    batch_save_predictions(model_outputs_dict, enc, "plots/predictions/")
    cv2.destroyAllWindows()


if __name__=='__main__':

    # path_model_default = '/mnt/data/datasets/radial/gd/models/RADIal_SwinTransformer_ADC.pth'
    path_model_default = '/Volumes/ELEMENTS/datasets/Trained_Models/RADIal_SwinTransformer_ADC.pth'

    path_model_default = Path('/Volumes/ELEMENTS/datasets/Trained_Models/RADIal_SwinTransformer_ADC.pth')
    if not path_model_default.exists():
        path_model_default = Path('/mnt/data/datasets/radial/gd/models/RADIal_SwinTransformer_ADC.pth')

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='./config/ADC_config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=path_model_default, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult)
