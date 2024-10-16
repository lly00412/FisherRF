EXP_PATH=$1

python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH} \
      --render_vcam --n_vcam 2 4 6 8 --seed 29506

python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH} \
      --render_vcam --n_vcam 2 4 6 8 --seed 1000

python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH} \
      --render_vcam --n_vcam 2 4 6 8 --seed 518

python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH} \
      --render_vcam --n_vcam 2 4 6 8 --seed 7463