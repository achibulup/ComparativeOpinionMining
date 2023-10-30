# first stage.
python3 main.py --dataset_name="Car-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 
python3 main.py --dataset_name="Ele-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 
python3 main.py --dataset_name="Camera-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 

# second and thrid stage.
python3 main.py --dataset_name="Car-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --factor=0.3
python3 main.py --dataset_name="Ele-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --factor=0.3
python3 main.py --dataset_name="Camera-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --factor=0.3
