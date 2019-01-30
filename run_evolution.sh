#Enzymes 1-2
python population_manager.py --cycle 350 --num_models 10 --load_lastcycle 0 --group_name 'ICIP2019' --trial_name 'Enz_1-2' --dataset_name 'ENZYMES' --num_classes 6 --num_iter 300 --learning_rate_step 110 --train_batch_size 540 --test_batch_size 60 --starter_learning_rate 0.005 --learning_rate_exp 0.8 --sparse 1 --folds 10 --snapshot_iter 1000 --prob_cycle 11 --processes 3 --reset_cycle 300