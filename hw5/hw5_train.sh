python3 make_token.py $1 $2 $3 
python3 train_w2v.py "./" 
python3 train_rnn.py "./" "w2v_model.bin"


#/share/home/tslsun025/.conda/envs/ml2019fall_hw5/bin/python3.6 make_token.py $1 $2 $3 
#/share/home/tslsun025/.conda/envs/ml2019fall_hw5/bin/python3.6 train_w2v.py "./" 
#/share/home/tslsun025/.conda/envs/ml2019fall_hw5/bin/python3.6 train_rnn.py "./" "w2v_model.bin"
