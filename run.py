#실행 코드
python train_and_evaluate_1.py --data_path rcc_1.csv --output_path results.json --model xgb --n_folds 5 --random_state 42 --num_iter 100 --n_estimators 100 --learning_rate 0.1 --max_depth 50 --eval_metric rmse --early_stopping_rounds 20 --verbose True
