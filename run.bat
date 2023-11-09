
for %%i in (5 10 15 20 25 30 0.1) do (
      echo "runing"
      python demo.py --data="Indian" --train_num=%%i --epoch=3000 --seed=114
)

for %%i in (5 10 15 20 25 30 0.1) do (
      echo "runing"
      python demo.py --data="Pavia" --train_num=%%i --epoch=3000 --seed=114
)