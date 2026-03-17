python src/test_mc_solver.py /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB/test --solver ApproxMC3 --timeout 5000
python src/test_mc_solver.py /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB_MIS/test --solver ApproxMC3 --timeout 5000

python src/show_mc_result.py /home/arie/Documents/phd/code/NSNet/ModelCounting/SATLIB/test/ runs/ApproxMC3/evaluations/ runs/MIS/evaluations/