for lr in 1e-2 1e-3 1e-4
do
  for wd in 0 1e-5 1e-4 1e-3 1e-2
  do
    for reg in 0 1e-4 1e-3 1e-2
    do
        for smooth in 0 1
        do
        echo "Running KAN with lr=${lr}, weight_decay=${wd}, kan_reg_coeff=${reg}, kan_smoothness-coeff=${smooth}"
        python mnist_lightning.py \
            --model kan \
            --learning-rate $lr \
            --weight-decay $wd \
            --kan-reg-coeff $reg \
            --kan-activation-coeff 1 \
            --kan-entropy-coeff 0 \
            --kan-smoothness-coeff $smooth \
            --epochs 40 \
            --batch-size 256 \
            --spline-grid-size 5 \
            --hidden-sizes 100
        done
    done
  done
done