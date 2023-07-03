LocalRetro model porting to Determined AI. The objectives is to train LocalRetro faster using Determined AI's distributed training and HP search features.  

LocalRetro is the Transformer-based model for Retrosynthesis Prediction with LocalRetro developed by Prof. Yousung Jung group at KAIST.

Original LocalRetro repo: 
https://github.com/kaist-amsg/LocalRetro

Step 1: Download the raw data 

Step 2: Preprocess the data USPTO_50K/USPTO_MIT
(refer to the original LocalRetro repo to complete Steps 1-2)

Step 3: Run experiments using Determined
    - Single GPU training 
    det -m <master-address> e create const.yaml .   
    - Ditributed training 
    det -m <master-address> e create distributed.yaml . 
    - Adaptive ASHA-based HPO
    det -m <master-address> e create search.yaml . 
