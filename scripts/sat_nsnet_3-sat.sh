# Needs cnf formulas, and marginals.pkl, hence first generate_labels
# 1 epoch takes from 16:35 until 16:40, so 3 hours total
# EPOCH #12
# Training...
# Training LR: 0.000100, Training loss: 0.128308
# Training accuracy: 0.488000
# Validating...
# Validating loss: 0.133405
# Validating accuracy: 0.525000
# Hierna lijkt loss heen en weer te schieten, geen steady improvement
# Epoch 6 gaf all validation acc 526, training loss 0.128

#                           task        exp_id                              train_dir                                                           valid_dir
# python src/train_model.py sat-solving sat_arienet_3-sat_marginal_on_4000 /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train_first_4000/ --valid_dir /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid_first_1000/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal --restore /home/arie/Documents/phd/code/NSNet/runs/sat_arienet_3-sat_marginal_on_4000/checkpoints/model_6.pt 
# python src/train_model.py sat-solving sat_arienet_3-sat_marginal_on_4000 /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train_first_4000/ --valid_dir /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid_first_1000/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
python src/train_model.py sat-solving sat_arienet_3-sat_marginal_on_4000 /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train_first_4000/ --valid_dir /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid_first_1000/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal

# Som i.p.v. max
# EPOCH #32
# Training...
# Training LR: 0.000100, Training loss: 0.112771
# Training accuracy: 0.578000
# Validating...
# Validating loss: 0.115108
# Validating accuracy: 0.592000

# Op mijn laptop (no gpu) duurt 1 epoch van 16:15 tot... minimaal 20 minuten, dus gaan we niet doen
# python src/train_model.py sat-solving sat_nsnet_3-sat_marginal /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train/ --valid_dir /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
# python src/test_model.py sat-solving /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/test/ --checkpoint runs/sat_nsnet_3-sat_marginal/checkpoints/model_best.pt
# python src/test_modnshon src/test_model.py sat-solving /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/test_hard/ --checkpoint runs/sat_nsnet_3-sat_assignment/checkpoints/model_best.pt