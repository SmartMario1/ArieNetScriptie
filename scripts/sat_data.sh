# NIET OPNIEUW DRAAIEN WANT DAN WORDEN FORMULES OPNIEUW GEMAAKT

# sr data (From some benchmark with SAT/UNSAT pairs with one different literal, UNSAT are discarded)
# python src/generate_sr_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/sr/train 30000 --min_n 10 --max_n 40
# python src/generate_sr_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/sr/valid 10000 --min_n 10 --max_n 40
# python src/generate_sr_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/sr/test 10000 --min_n 10 --max_n 40
# python src/generate_sr_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/sr/test_hard 10000 --min_n 40 --max_n 200


# python src/generate_labels.py assignment /home/arie/Documents/phd/code/NSNet/SATSolving/sr/train
# python src/generate_labels.py assignment /home/arie/Documents/phd/code/NSNet/SATSolving/sr/valid

# Done

# 3-sat data (Random formulas)
# python src/generate_3-sat_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train 30000 --min_n 10 --max_n 40 # this one ran until 04695
# python src/generate_3-sat_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid 10000 --min_n 10 --max_n 40
# python src/generate_3-sat_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/test 10000 --min_n 10 --max_n 40
# python src/generate_3-sat_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/test_hard 10000 --min_n 40 --max_n 200

# # Done
# # 09459 duurt heel lang...?
# python src/generate_labels.py marginal /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train
# python src/generate_labels.py marginal /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid

# # Done
# python src/generate_labels.py marginal /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid
# python src/generate_labels.py assignment /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train
# python src/generate_labels.py assignment /home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/valid

#### ARIE SMALL VERSION
python src/nsnet/generate_labels.py marginal /home/sander/Thesis2/nsnet/SATSolving/3-sat/train_first_4000_ArieNet
# python src/nsnet/generate_labels.py marginal /home/sander/Thesis2/nsnet/SATSolving/3col/train
# python src/nsnet/generate_labels.py marginal /home/sander/Thesis2/nsnet/SATSolving/3col/val
#####



# # # ca data (Real world formulas)
# python src/generate_ca_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/ca/train 30000 --min_n 10 --max_n 40
# python src/generate_ca_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/ca/valid 10000 --min_n 10 --max_n 40
# python src/generate_ca_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/ca/test 10000 --min_n 10 --max_n 40
# python src/generate_ca_data.py /home/arie/Documents/phd/code/NSNet/SATSolving/ca/test_hard 10000 --min_n 40 --max_n 200

# python src/generate_labels.py marginal /home/arie/Documents/phd/code/NSNet/SATSolving/ca/train
# python src/generate_labels.py marginal /home/arie/Documents/phd/code/NSNet/SATSolving/ca/valid

# python src/generate_labels.py assignment /home/arie/Documents/phd/code/NSNet/SATSolving/ca/train
# python src/generate_labels.py assignment /home/arie/Documents/phd/code/NSNet/SATSolving/ca/valid
# # Done
