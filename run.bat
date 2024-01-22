
for %%i in (40, 50) do (
      echo "runing"
      python demo.py --data="Indian" --train_num=%%i --epoch=3000 --seed=114 --train_time=2
)

for %%i in (40, 50) do (
      echo "runing"
      python demo.py --data="Pavia" --train_num=%%i --epoch=3000 --seed=114 --train_time=3
)

for %%i in (40, 50) do (
      echo "runing"
      python demo.py --data="Honghu" --train_num=%%i --epoch=3000 --seed=114 --train_time=3
)