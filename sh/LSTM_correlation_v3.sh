cd ../
# model='LSTM_correlation_v3'
# datapath='./../SeoulData/urban-core_v2.csv'
# for n_layers in 2
# do
#     for time_window in 5
#     do
#         summary="model_${model}_lstm_n_layers_${n_layers}_time_window_${time_window}_datapath_${datapath}"

#         python main.py --summary $summary --model $model --lstm_n_layers $n_layers --time_window $time_window --device 2 --datapath $datapath --patience 10
#     done
# done

# model='LSTM_correlation_v3'
# datapath='./../SeoulData/urban-mix_v2.csv'
# for n_layers in 2
# do
#     for time_window in 5
#     do
#         summary="model_${model}_lstm_n_layers_${n_layers}_time_window_${time_window}_datapath_${datapath}"

#         python main.py --summary $summary --model $model --lstm_n_layers $n_layers --time_window $time_window --device 2 --datapath $datapath --patience 3 --lr 0.005
#     done
# done
