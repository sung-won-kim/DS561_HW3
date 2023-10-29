cd ../
# model='LSTM'
# datapath='./../SeoulData/urban-core_v2.csv'
# for n_layers in 2 4 6
# do
#     for time_window in 5 10 15
#     do
#         summary="model_${model}_lstm_n_layers_${n_layers}_time_window_${time_window}_datapath_${datapath}"

#         python main.py --summary $summary --model $model --lstm_n_layers $n_layers --time_window $time_window --device 3 --datapath $datapath
#     done
# done

model='LSTM'
datapath='./../SeoulData/urban-mix_v2.csv'
for n_layers in 2
do
    for time_window in 5
    do
        summary="model_${model}_lstm_n_layers_${n_layers}_time_window_${time_window}_datapath_${datapath}"

        python main.py --summary $summary --model $model --lstm_n_layers $n_layers --time_window $time_window --device 2 --datapath $datapath --lr 0.001
    done
done