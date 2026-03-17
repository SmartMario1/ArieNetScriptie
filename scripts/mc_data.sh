# bird data
# python src/download_bird_data.py /home/arie/Documents/phd/code/NSNet/ModelCounting/BIRD
# python src/clean_data.py /home/arie/Documents/phd/code/NSNet/ModelCounting/BIRD
# python src/generate_labels.py model-counting /home/arie/Documents/phd/code/NSNet/ModelCounting/BIRD/train
# python src/generate_labels.py model-counting /home/arie/Documents/phd/code/NSNet/ModelCounting/BIRD/test
# The above couldn't be done because the link doesn't exist

# satlib data
# python src/download_satlib_data.py /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB
# Done

# python src/clean_data.py /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB
# Done I think

python src/generate_labels.py model-counting /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB
python src/split_satlib_data.py /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB --keep_category

# mis preprocessing
python src/run_mis_solver.py /home/arie/Documents/phd/code/NSNet/ModelCounting/BIRD/test /home/arie/Documents/phd/code/NSNet/ModelCounting/BIRD_MIS/test --timeout 1000
python src/run_mis_solver.py /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB/test /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB_MIS/test --timeout 1000
