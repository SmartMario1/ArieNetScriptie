python src/train_model.py sat-solving sat_nsnet_ca_marginal /home/arie/Documents/phd/code/NSNet/SATSolving/ca/train/ --valid_dir /home/arie/Documents/phd/code/NSNet/SATSolving/ca/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
python src/test_model.py sat-solving /home/arie/Documents/phd/code/NSNet/SATSolving/ca/test/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt
python src/test_model.py sat-solving /home/arie/Documents/phd/code/NSNet/SATSolving/ca/test_hard/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt

python src/train_model.py sat-solving sat_nsnet_ca_assignment /home/arie/Documents/phd/code/NSNet/SATSolving/ca/train/ --valid_dir /home/arie/Documents/phd/code/NSNet/SATSolving/ca/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment
python src/test_model.py sat-solving /home/arie/Documents/phd/code/NSNet/SATSolving/ca/test/ --checkpoint runs/sat_nsnet_ca_assignment/checkpoints/model_best.pt
python src/test_model.py sat-solving /home/arie/Documents/phd/code/NSNet/SATSolving/ca/test_hard/ --checkpoint runs/sat_nsnet_ca_assignment/checkpoints/model_best.pt