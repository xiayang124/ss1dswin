for %%i in (10) do (
      echo "runing"
      python demo.py --data="Pavia" --train_num=%%i --epoch=3000 --seed=114
)

for %%i in (10) do (
      echo "runing"
      python demo.py --data="Honghu" --train_num=%%i --epoch=3000 --seed=114
)