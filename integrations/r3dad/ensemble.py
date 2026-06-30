import argparse
import os
import pandas as pd
import re

def main(args):

    all_df = []
    for cate_dir in os.listdir(os.path.join(args.work_path)):
        if not os.path.isdir(os.path.join(args.work_path, cate_dir)):
            continue
        log_path =  os.path.join(args.work_path, cate_dir, "log.txt")
        with open(log_path, 'r') as file:
            text = file.read()

        max_roci = 0.0
        max_rocp = 0.0
        max_api = 0.0
        max_app = 0.0
        for line in text.split('\n'):
            if 'ROC_i 0' in line:
                # Regular expression to extract the desired information
                pattern = r"ROC_i ([\d\.]+) \| ROC_p ([\d\.]+) \| AP_i ([\d\.]+) \| AP_p ([\d\.]+)"

                # Extracting the information
                match = re.search(pattern, line)
                roc_i, roc_p, ap_i, ap_p = match.groups()
                
                max_roci = max(max_roci, float(roc_i))
                max_rocp = max(max_rocp, float(roc_p))
                max_api = max(max_api, float(ap_i))
                max_app = max(max_app, float(ap_p))

        df = pd.DataFrame({
            'I-AUROC': [max_roci], 
            'P-AUROC': [max_rocp], 
            'I-AP': [max_api], 
            'P-AP': [max_app]
        })
        df.index = [cate_dir.split('_')[0]]
        all_df.append(df)
    
    all_df = pd.concat(all_df)
    all_df.sort_index(inplace=True)
    print(f"Ensembled results of {args.work_path}")
    print(all_df.mean(0))

    all_df.T.to_csv(os.path.join(args.work_path, "ensemble.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_path")
    args = parser.parse_args()

    main(args)